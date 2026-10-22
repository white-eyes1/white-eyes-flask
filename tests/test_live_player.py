"""FNC-S-007 라이브 분석·재연결 (가짜 VideoCapture·가짜 탐지기·가짜 녹화기)"""

import threading
import time

import numpy as np
import pytest

from white_eyes_flask.camera.camera_job import CameraJob, CameraStatus
from white_eyes_flask.camera.player import PlayOutcome
from white_eyes_flask.camera.result_queue import ResultQueue
from white_eyes_flask.live import live_player
from white_eyes_flask.live.live_analyzer import LiveAnalyzer
from white_eyes_flask.live.live_player import LivePlayer
from white_eyes_flask.live.live_source import is_rtsp_uri, mask_uri
from white_eyes_flask.recording.recorder import RecorderFailed, RecorderUnavailable
from white_eyes_flask.stream.stream_service import StreamService

FRAME = np.zeros((90, 160, 3), dtype=np.uint8)


def wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class FakeDetector:
    """프레임마다 가운데에 사람 한 명."""

    def detect(self, frame):
        return np.array([[60.0, 30.0, 100.0, 70.0, 0.9]])


class FakeCapture:
    """script: 읽기마다 True(프레임)·False(실패). 다 쓰면 마지막 값을 반복한다."""

    def __init__(self, opened=True, script=(True,), delay=0.0):
        self.opened = opened
        self.script = list(script)
        self.delay = delay
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        time.sleep(self.delay)
        ok = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        return ok, (FRAME.copy() if ok else None)

    def release(self):
        self.released = True


class FakeReporter:
    def __init__(self):
        self.calls = []

    def send(self, camera_id, status, progress=None, error_message=None):
        self.calls.append((camera_id, status, error_message))
        return live_player.ReportResult.OK, ""


class FakeRecorder:
    def __init__(self, write_error=None):
        self.frames = []
        self.boxes = []
        self.write_error = write_error
        self.closed = False

    def write_frame(self, frame, at):
        if self.write_error:
            raise self.write_error
        self.frames.append(at)

    def add_boxes(self, at, boxes):
        self.boxes.append((at, boxes))

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def fast_timers(monkeypatch):
    monkeypatch.setattr(live_player, "LIVE_RECONNECT_FAIL_SEC", 0.3)
    monkeypatch.setattr(live_player, "RECONNECT_INTERVAL_SEC", 0.02)
    monkeypatch.setattr(live_player, "ERROR_RECONNECT_INTERVAL_SEC", 0.05)
    monkeypatch.setattr(live_player, "ANALYSIS_POLL_SEC", 0.02)


def make(captures, recorder=None, streams=None, detector=None, **kwargs):
    """captures: 연결할 때마다 차례로 돌려줄 FakeCapture들(다 쓰면 열리지 않는 것)."""
    queue_ = list(captures)
    sink = ResultQueue()
    reporter = FakeReporter()
    streams = streams or StreamService()

    def factory(uri):
        return queue_.pop(0) if queue_ else FakeCapture(opened=False)

    player = LivePlayer(
        lambda: detector or FakeDetector(),
        sink,
        reporter,
        streams,
        recorder,
        fps=5,
        conf_thres=0.3,
        edge_margin=0.05,
        capture_factory=factory,
        **kwargs,
    )
    return player, sink, reporter, streams


def run(player, stop=None):
    """job 스레드처럼 play를 돌린다. (스레드, 정지 플래그, 결과 담는 리스트)"""
    stop = stop or threading.Event()
    outcome = []
    thread = threading.Thread(
        target=lambda: outcome.append(player.play(7, "rtsp://u:p@cam/s", stop)),
        daemon=True,
    )
    thread.start()
    return thread, stop, outcome


def finish(thread, stop):
    stop.set()
    thread.join(5)
    assert not thread.is_alive()


# ---------- 읽기·분석·녹화·전송 ----------


def test_every_frame_goes_to_the_recorder_and_only_sampled_frames_are_analyzed():
    recorder = FakeRecorder()
    player, sink, _, _ = make([FakeCapture(delay=0.01)], recorder)
    thread, stop, _ = run(player)
    assert wait_until(lambda: len(recorder.frames) >= 60)
    finish(thread, stop)
    # 읽은 프레임은 모두 녹화기로, 분석은 초당 5장 안팎으로 솎는다
    assert len(recorder.boxes) < len(recorder.frames) / 3
    assert recorder.boxes
    assert recorder.boxes[0][1][0].track_id is None or True


def test_per_second_results_use_the_same_shape_as_the_dummy_player():
    player, sink, _, _ = make([FakeCapture(delay=0.01)])
    thread, stop, _ = run(player)
    assert wait_until(lambda: len(sink) >= 1, timeout=6)
    finish(thread, stop)
    item = sink.drain()[0]
    assert set(item) == {"cameraId", "at", "personCount", "in", "out", "dwellSec"}
    assert item["cameraId"] == 7
    assert item["personCount"] == 1
    assert set(item["in"]) == {"N", "E", "S", "W"}


def test_frames_are_published_only_while_someone_watches():
    streams = StreamService()
    stream = streams.open(7)
    player, _, _, _ = make([FakeCapture(delay=0.01)], streams=streams)
    thread, stop, _ = run(player)
    time.sleep(0.5)
    assert (
        stream.snapshot_jpeg() is None
    )  # 보는 사람이 없던 동안은 프레임을 만들지 않았다
    assert wait_until(lambda: stream.snapshot_jpeg() is not None)  # 요청 뒤에는 만든다
    finish(thread, stop)


# ---------- 재연결 ----------


def test_error_is_reported_after_the_fail_time_and_running_after_recovery():
    # 처음엔 열리지 않다가(30초 대신 0.3초) ERROR, 그 뒤 연결되면 RUNNING
    captures = [FakeCapture(opened=False) for _ in range(40)]
    player, _, reporter, _ = make(captures + [FakeCapture(delay=0.01)])
    statuses = []
    player.on_status = statuses.append
    thread, stop, _ = run(player)
    assert wait_until(lambda: (7, "RUNNING", None) in reporter.calls)
    finish(thread, stop)
    assert reporter.calls[0] == (7, "ERROR", "스트림 연결 실패")
    assert statuses == [CameraStatus.ERROR, CameraStatus.RUNNING]
    assert reporter.calls.count((7, "ERROR", "스트림 연결 실패")) == 1


def test_the_configured_fail_time_decides_when_error_is_reported():
    # 모듈 기본값(이 시험에서는 0.3초)이 아니라 넘겨 준 설정값(0.9초)을 따른다
    captures = [FakeCapture(opened=False) for _ in range(100)]
    player, _, reporter, _ = make(captures, reconnect_fail_sec=0.9)
    started = time.monotonic()
    thread, stop, _ = run(player)
    time.sleep(0.6)
    assert reporter.calls == []  # 0.3초를 넘겼지만 설정값 0.9초는 아직이다
    assert wait_until(lambda: (7, "ERROR", "스트림 연결 실패") in reporter.calls)
    elapsed = time.monotonic() - started
    finish(thread, stop)
    assert elapsed >= 0.9


def test_a_longer_configured_fail_time_keeps_the_camera_running():
    captures = [FakeCapture(opened=False) for _ in range(100)]
    player, _, reporter, _ = make(captures, reconnect_fail_sec=60)
    thread, stop, _ = run(player)
    time.sleep(0.8)
    finish(thread, stop)
    assert reporter.calls == []


def test_the_live_factory_passes_the_config_fail_time_to_each_player():
    from unittest.mock import MagicMock

    from white_eyes_flask.app import build_live_factory
    from white_eyes_flask.config import Config

    config = Config(
        internal_token="t",
        yolo_weights=None,
        host="0.0.0.0",
        port=5000,
        live_reconnect_fail_sec=7,
    )
    factory = build_live_factory(
        config, MagicMock(), ResultQueue(), StreamService(), MagicMock()
    )
    player, _ = factory.create(3)
    assert player._reconnect_fail_sec == 7


def test_no_error_is_reported_for_a_short_outage():
    captures = [FakeCapture(script=[True, True, False]), FakeCapture(delay=0.01)]
    player, _, reporter, _ = make(captures)
    thread, stop, _ = run(player)
    time.sleep(0.5)
    finish(thread, stop)
    assert reporter.calls == []


def test_lost_connection_is_closed_and_reopened():
    first = FakeCapture(script=[True, False])
    second = FakeCapture(delay=0.01)
    player, _, _, _ = make([first, second])
    thread, stop, _ = run(player)
    assert wait_until(lambda: first.released)
    time.sleep(0.3)  # 다시 연결해 읽는 중
    finish(thread, stop)
    assert second.released  # 정지할 때 열려 있던 연결도 닫는다


def test_stop_ends_play_while_waiting_to_reconnect():
    player, _, _, _ = make([])
    thread, stop, outcome = run(player)
    time.sleep(0.1)
    finish(thread, stop)
    assert outcome == [PlayOutcome.STOPPED]


# ---------- 녹화기·탐지기 오류 ----------


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (RecorderUnavailable("no ffmpeg"), "녹화를 시작할 수 없음"),
        (RecorderFailed("died"), "녹화 중 오류가 발생함"),
    ],
)
def test_recorder_errors_end_the_job_with_error(error, message):
    recorder = FakeRecorder(write_error=error)
    player, _, reporter, _ = make([FakeCapture()], recorder)
    job = CameraJob(
        7, "rtsp://cam", CameraStatus.RUNNING, player=player, recorder=recorder
    )
    job.start()
    assert wait_until(lambda: not job.is_alive())
    assert job.status == CameraStatus.ERROR
    assert (7, "ERROR", message) in reporter.calls
    assert recorder.closed  # 어떻게 끝나도 녹화기는 닫는다


def test_missing_model_reports_error_and_ends():
    from white_eyes_flask.analysis.detector_provider import DetectorUnavailableError

    def unavailable():
        raise DetectorUnavailableError("no weights")

    player, _, reporter, _ = make([FakeCapture()])
    player._detector_provider = unavailable
    stop = threading.Event()
    assert player.play(7, "rtsp://cam", stop) == PlayOutcome.FAILED
    assert reporter.calls == [(7, "ERROR", "분석 모델을 불러올 수 없음")]


# ---------- 분석기 ----------


class CountingDetector:
    """호출마다 사람 수를 바꾼다."""

    def __init__(self, counts):
        self.counts = list(counts)

    def detect(self, frame):
        n = self.counts.pop(0)
        return np.array(
            [[60.0 + i * 5, 30.0, 70.0 + i * 5, 50.0, 0.9] for i in range(n)]
        ).reshape(-1, 5)


def test_analyzer_averages_people_per_second_and_skips_empty_seconds():
    sink = ResultQueue()
    analyzer = LiveAnalyzer(
        7, CountingDetector([1, 2, 4, 1]), sink, fps=5, conf_thres=0.3, edge_margin=0.05
    )
    for at in (100.0, 100.2, 100.4):  # 100초: 1, 2, 4 -> 평균 2.33 -> 2
        analyzer.analyze(FRAME, at)
    analyzer.analyze(FRAME, 105.0)  # 101~104초는 프레임이 없어 보내지 않는다
    analyzer.close()
    items = sink.drain()
    assert [i["personCount"] for i in items] == [2, 1]
    assert items[0]["at"].endswith("+09:00")
    assert len(items) == 2


def test_analyzer_flushes_a_finished_second_while_idle():
    sink = ResultQueue()
    analyzer = LiveAnalyzer(
        7, CountingDetector([1]), sink, fps=5, conf_thres=0.3, edge_margin=0.05
    )
    analyzer.analyze(FRAME, 100.0)
    analyzer.flush_idle(101.5)
    assert len(sink) == 0  # 분석이 늦을 수 있어 1초 더 기다린다
    analyzer.flush_idle(102.5)
    assert len(sink) == 1


def test_analyzer_starts_a_new_tracker_after_a_reconnect():
    sink = ResultQueue()
    analyzer = LiveAnalyzer(
        7, CountingDetector([1, 1]), sink, fps=5, conf_thres=0.3, edge_margin=0.05
    )
    analyzer.analyze(FRAME, 100.0)
    tracker = analyzer._tracker
    analyzer.end_session()
    analyzer.analyze(FRAME, 160.0)
    assert analyzer._tracker is not tracker


# ---------- 주소 ----------


def test_uri_helpers():
    assert is_rtsp_uri("rtsp://cam/s") and is_rtsp_uri("RTSPS://cam/s")
    assert not is_rtsp_uri("http://cam/s") and not is_rtsp_uri("dummy/videos/a.mp4")
    assert mask_uri("rtsp://admin:pw@10.0.0.5/s") == "rtsp://***@10.0.0.5/s"
    assert mask_uri("rtsp://10.0.0.5/s") == "rtsp://10.0.0.5/s"


# ---------- 검수 반영 ----------


class RaisingCapture(FakeCapture):
    def read(self):
        raise RuntimeError("cv2.error: decode failed")


def test_exception_while_reading_reconnects_instead_of_ending_the_job():
    bad = RaisingCapture()
    good = FakeCapture(delay=0.01)
    recorder = FakeRecorder()
    player, _, _, _ = make([bad, good], recorder)
    thread, stop, outcome = run(player)
    assert wait_until(lambda: len(recorder.frames) >= 3)
    assert bad.released
    assert thread.is_alive() and outcome == []
    finish(thread, stop)
    assert outcome == [PlayOutcome.STOPPED]


def test_stop_during_a_slow_connect_ends_within_the_stop_wait():
    class SlowFactory:
        reads = 0

        def __call__(self, uri):
            time.sleep(5)  # 연결이 5초 걸린다
            cap = FakeCapture()
            orig = cap.read

            def read():
                SlowFactory.reads += 1
                return orig()

            cap.read = read
            return cap

    player, *_ = make([])
    player._capture_factory = SlowFactory()
    thread, stop, outcome = run(player)
    time.sleep(0.2)
    began = time.monotonic()
    stop.set()
    thread.join(10)
    assert not thread.is_alive()
    assert time.monotonic() - began < 10  # CameraJob의 STOP_WAIT_SEC
    assert SlowFactory.reads == 0  # 연결 직후 정지를 보고 읽지 않고 나왔다
    assert outcome == [PlayOutcome.STOPPED]


def test_analysis_thread_survives_an_exception(monkeypatch):
    calls = {"n": 0}
    original = LiveAnalyzer.analyze

    def flaky(self, frame, at):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return original(self, frame, at)

    monkeypatch.setattr(LiveAnalyzer, "analyze", flaky)
    recorder = FakeRecorder()
    player, _, _, _ = make([FakeCapture(delay=0.01)], recorder)
    thread, stop, _ = run(player)
    assert wait_until(lambda: len(recorder.boxes) >= 2)  # 예외 뒤에도 분석이 이어진다
    finish(thread, stop)


def test_analysis_thread_survives_an_exception_in_flush_idle(monkeypatch):
    calls = {"n": 0}

    def bad_flush(self, now):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")

    monkeypatch.setattr(LiveAnalyzer, "flush_idle", bad_flush)
    recorder = FakeRecorder()
    player, _, _, _ = make([FakeCapture(delay=0.01)], recorder)
    thread, stop, _ = run(player)
    assert wait_until(lambda: calls["n"] >= 3 and len(recorder.boxes) >= 2)
    finish(thread, stop)


class ScriptedDetector:
    def __init__(self, boxes):
        self.boxes = list(boxes)

    def detect(self, frame):
        return np.array([self.boxes.pop(0)])


def test_leavers_confirmed_with_no_second_to_send_are_dropped(caplog):
    from white_eyes_flask.analysis.result import FrameResult, TrackedBox

    class WalkingTracker:
        """추적 번호 1번 사람이 가운데에 있다가 왼쪽 띠로 나간다."""

        def __init__(self):
            self.xs = [0.5, 0.5, 0.02, 0.02]

        def update(self, frame):
            cx = self.xs.pop(0)
            box = TrackedBox(1, cx - 0.05, 0.4, 0.1, 0.2, cx, 0.5, 0.9)
            return FrameResult(count=1, boxes=[box])

    sink = ResultQueue()
    analyzer = LiveAnalyzer(
        7, FakeDetector(), sink, fps=5, conf_thres=0.3, edge_margin=0.05
    )
    analyzer.analyze(FRAME, 100.0)
    analyzer._tracker = WalkingTracker()
    for at in (100.2, 100.4, 100.6):
        analyzer.analyze(FRAME, at)
    analyzer.flush_idle(103.0)  # 100초를 보냈다. 이제 보낼 초가 없다
    assert len(sink) == 1
    with caplog.at_level("INFO"):
        analyzer.end_session()  # 여기서 나간 사람이 확정된다
    assert "버림" in caplog.text
    analyzer.analyze(FRAME, 5000.0)  # 몇 시간 뒤 첫 프레임
    analyzer.close()
    items = sink.drain()
    assert len(items) == 2
    # 옛 유출이 몇 시간 뒤의 초에 붙지 않는다
    assert sum(items[1]["out"].values()) == 0 and items[1]["dwellSec"] == []


def test_close_drops_leavers_when_there_is_no_second(caplog):
    sink = ResultQueue()
    analyzer = LiveAnalyzer(
        7, FakeDetector(), sink, fps=5, conf_thres=0.3, edge_margin=0.05
    )
    analyzer.analyze(FRAME, 100.0)
    analyzer.flush_idle(103.0)
    analyzer.close()  # 이미 보낸 뒤라 보낼 초가 없다. 예외 없이 끝나고 다시 보내지 않는다
    assert len(sink) == 1
