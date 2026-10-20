"""FNC-S-009 더미 분석 준비 (분석 + 캐시 + 상태 보고를 엮는 DummyPreparer)"""

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from white_eyes_flask.analysis.detector_provider import DetectorUnavailableError
from white_eyes_flask.analysis.dummy_cache import DummyCache
from white_eyes_flask.camera.preparer import DummyPreparer, PrepareOutcome

SOURCE = "dummy/videos/clip.mp4"
WIDTH, HEIGHT = 160, 120


class FakeReporter:
    def __init__(self):
        self.calls = []
        self._lock = threading.Lock()

    def report(self, camera_id, status, progress=None, error_message=None):
        with self._lock:
            self.calls.append((camera_id, status, progress, error_message))
        return True

    def statuses(self, camera_id=None):
        return [c[1] for c in self.calls if camera_id in (None, c[0])]


class CountingDetector:
    def __init__(self, delay=0.0):
        self.calls = 0
        self.delay = delay
        self.started = threading.Event()

    def detect(self, frame):
        self.calls += 1
        self.started.set()
        if self.delay:
            time.sleep(self.delay)
        return np.zeros((0, 5), dtype=np.float32)


def make_video(path: Path, frames: int = 60, fps: float = 30.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT)
    )
    for _ in range(frames):
        writer.write(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
    writer.release()


@pytest.fixture
def media(tmp_path: Path) -> Path:
    root = tmp_path / "media"
    make_video(root / SOURCE)
    return root


def make_preparer(tmp_path, media, detector=None, reporter=None, provider=None):
    cache = DummyCache(cache_dir=tmp_path / "cache", media_root=media)
    detector = detector or CountingDetector()
    reporter = reporter or FakeReporter()
    preparer = DummyPreparer(
        media_root=media,
        cache=cache,
        detector_provider=provider or (lambda: detector),
        reporter=reporter,
        fps=5,
        conf_thres=0.3,
        edge_margin=0.05,
    )
    return preparer, cache, detector, reporter


def test_prepare_reports_preparing_then_progress_then_running(tmp_path, media):
    preparer, cache, _, reporter = make_preparer(tmp_path, media)
    outcome = preparer.prepare(7, SOURCE, threading.Event())
    assert outcome == PrepareOutcome.DONE
    assert reporter.calls[0] == (7, "PREPARING", 0, None)
    assert reporter.calls[-1] == (7, "RUNNING", 100, None)
    progress = [c[2] for c in reporter.calls[1:-1]]
    assert progress == sorted(set(progress))
    assert all(c[1] == "PREPARING" for c in reporter.calls[:-1])


def test_prepare_creates_cache_and_box_files(tmp_path, media):
    preparer, cache, _, _ = make_preparer(tmp_path, media)
    assert preparer.has_cache(SOURCE) is False
    preparer.prepare(7, SOURCE, threading.Event())
    assert preparer.has_cache(SOURCE) is True
    assert cache.result_path(SOURCE).is_file()
    assert cache.boxes_path(SOURCE).is_file()


def test_running_is_reported_only_after_the_files_exist(tmp_path, media):
    seen = {}

    class CheckingReporter(FakeReporter):
        def report(self, camera_id, status, progress=None, error_message=None):
            if status == "RUNNING":
                seen["has_cache"] = cache.has(SOURCE)
            return super().report(camera_id, status, progress, error_message)

    preparer, cache, _, _ = make_preparer(tmp_path, media, reporter=CheckingReporter())
    preparer.prepare(7, SOURCE, threading.Event())
    assert seen == {"has_cache": True}


def test_cache_made_while_waiting_skips_the_analysis(tmp_path, media):
    preparer, cache, detector, reporter = make_preparer(tmp_path, media)
    preparer.prepare(7, SOURCE, threading.Event())
    detector_calls = detector.calls
    reporter.calls.clear()
    outcome = preparer.prepare(8, SOURCE, threading.Event())
    assert outcome == PrepareOutcome.DONE
    assert detector.calls == detector_calls  # 다시 분석하지 않음
    assert reporter.statuses() == ["RUNNING"]


def test_two_cameras_on_the_same_video_analyze_it_once(tmp_path, media):
    detector = CountingDetector(delay=0.01)
    preparer, _, _, reporter = make_preparer(tmp_path, media, detector=detector)
    outcomes = {}

    def run(camera_id):
        outcomes[camera_id] = preparer.prepare(camera_id, SOURCE, threading.Event())

    first = threading.Thread(target=run, args=(1,))
    first.start()
    assert detector.started.wait(timeout=5)
    second = threading.Thread(target=run, args=(2,))
    second.start()
    first.join(timeout=30)
    second.join(timeout=30)
    assert outcomes == {1: PrepareOutcome.DONE, 2: PrepareOutcome.DONE}
    assert detector.calls == 10  # 60프레임 30fps = 2초 = 슬롯 10개, 한 번만 분석
    assert reporter.statuses(1)[-1] == "RUNNING"
    assert reporter.statuses(2)[-1] == "RUNNING"


def test_stop_during_analysis_cancels_and_removes_files(tmp_path, media):
    detector = CountingDetector(delay=0.05)
    preparer, cache, _, reporter = make_preparer(tmp_path, media, detector=detector)
    stop = threading.Event()
    result = {}
    thread = threading.Thread(
        target=lambda: result.setdefault("outcome", preparer.prepare(7, SOURCE, stop))
    )
    thread.start()
    assert detector.started.wait(timeout=5)
    stop.set()
    thread.join(timeout=10)
    assert result["outcome"] == PrepareOutcome.STOPPED
    assert preparer.has_cache(SOURCE) is False
    leftovers = [p for p in tmp_path.rglob("*") if p.is_file() and p.suffix != ".mp4"]
    assert leftovers == []
    assert "RUNNING" not in reporter.statuses() and "ERROR" not in reporter.statuses()


def test_stop_while_waiting_for_another_camera_returns_stopped(tmp_path, media):
    detector = CountingDetector(delay=0.05)
    preparer, _, _, _ = make_preparer(tmp_path, media, detector=detector)
    first = threading.Thread(
        target=lambda: preparer.prepare(1, SOURCE, threading.Event())
    )
    first.start()
    assert detector.started.wait(timeout=5)
    stop = threading.Event()
    result = {}
    second = threading.Thread(
        target=lambda: result.setdefault("outcome", preparer.prepare(2, SOURCE, stop))
    )
    second.start()
    time.sleep(0.2)
    stop.set()
    second.join(timeout=5)
    assert result["outcome"] == PrepareOutcome.STOPPED
    first.join(timeout=30)


def test_missing_model_reports_error_without_analysis(tmp_path, media):
    def provider():
        raise DetectorUnavailableError("가중치 없음")

    preparer, cache, _, reporter = make_preparer(tmp_path, media, provider=provider)
    outcome = preparer.prepare(7, SOURCE, threading.Event())
    assert outcome == PrepareOutcome.FAILED
    assert reporter.calls[-1] == (7, "ERROR", None, "분석 모델을 불러올 수 없음")
    assert cache.has(SOURCE) is False


def test_unreadable_video_reports_error(tmp_path, media):
    (media / SOURCE).write_bytes(b"")
    preparer, cache, _, reporter = make_preparer(tmp_path, media)
    outcome = preparer.prepare(7, SOURCE, threading.Event())
    assert outcome == PrepareOutcome.FAILED
    assert reporter.calls[-1] == (7, "ERROR", None, "영상 파일을 읽을 수 없음")
    assert cache.has(SOURCE) is False


def test_missing_video_reports_error(tmp_path, media):
    (media / SOURCE).unlink()
    preparer, _, _, reporter = make_preparer(tmp_path, media)
    assert preparer.prepare(7, SOURCE, threading.Event()) == PrepareOutcome.FAILED
    assert reporter.calls[-1][1:] == ("ERROR", None, "영상 파일을 읽을 수 없음")


def test_unexpected_error_reports_generic_error_and_cleans_up(tmp_path, media, caplog):
    def provider():
        raise RuntimeError("예상 못 한 오류")

    preparer, cache, _, reporter = make_preparer(tmp_path, media, provider=provider)
    assert preparer.prepare(7, SOURCE, threading.Event()) == PrepareOutcome.FAILED
    assert reporter.calls[-1] == (7, "ERROR", None, "분석 중 오류가 발생함")
    assert "예상 못 한 오류" in caplog.text  # 원인은 로그에만 남김
    assert cache.has(SOURCE) is False


def test_failing_status_report_does_not_stop_the_analysis(tmp_path, media):
    class BrokenReporter:
        def report(self, *args, **kwargs):
            raise RuntimeError("보고 실패")

    preparer, cache, _, _ = make_preparer(tmp_path, media, reporter=BrokenReporter())
    outcome = preparer.prepare(7, SOURCE, threading.Event())
    assert outcome == PrepareOutcome.DONE
    assert cache.has(SOURCE) is True
