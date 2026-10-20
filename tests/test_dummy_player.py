"""FNC-S-010 더미 재생·반복 등록 (DummyPlayer)"""

import logging
import threading
import time
from datetime import datetime

import pytest

from white_eyes_flask.analysis.dummy_cache import DummyCache
from white_eyes_flask.camera.dummy_player import (
    DummyPlayer,
    PlayOutcome,
    RegistrationWorker,
)
from white_eyes_flask.camera.kst import KST
from white_eyes_flask.camera.recording_registrar import RegisterResult
from white_eyes_flask.camera.result_queue import ResultQueue

SOURCE = "dummy/videos/clip.mp4"
CAMERA = 7
ZERO = {"N": 0, "E": 0, "S": 0, "W": 0}


def epoch(hour, minute, second, micro=0) -> float:
    return datetime(2026, 10, 12, hour, minute, second, micro, tzinfo=KST).timestamp()


def second_result(sec, count=None, dwell=None):
    return {
        "sec": sec,
        "personCount": sec * 10 if count is None else count,
        "in": {**ZERO, "N": sec},
        "out": dict(ZERO),
        "dwellSec": dwell or [],
    }


class FakeClock:
    """대기하면 시계가 그만큼 흐르는 가짜 시계. max_waits번 기다리면 정지시킨다."""

    def __init__(self, start: float, max_waits: int):
        self.now = start
        self.max_waits = max_waits
        self.waits = []

    def clock(self) -> float:
        return self.now

    def wait(self, stop: threading.Event, timeout: float) -> bool:
        self.waits.append(timeout)
        if len(self.waits) >= self.max_waits:
            stop.set()
            return True
        self.now += timeout
        return stop.is_set()


class FakeRegistrar:
    def __init__(self, results=None):
        self.results = list(results or [])
        self.calls = []
        self.block = None

    def register(self, camera_id, started_ms, video_path, box_path):
        if self.block is not None:
            self.block.wait(timeout=5)
        self.calls.append((camera_id, started_ms, video_path, box_path))
        return self.results.pop(0) if self.results else RegisterResult.OK


class FakeReporter:
    def __init__(self):
        self.calls = []

    def report(self, camera_id, status, progress=None, error_message=None):
        self.calls.append((camera_id, status, progress, error_message))
        return True


@pytest.fixture
def cache(tmp_path):
    return DummyCache(cache_dir=tmp_path / "cache", media_root=tmp_path / "media")


def publish(cache, seconds, duration=None):
    with cache.writer(SOURCE, fps=5) as writer:
        writer.commit(seconds, duration)


def make_player(cache, fake, registrar=None, sink=None, reporter=None, **kwargs):
    registrar = registrar or FakeRegistrar()
    sink = sink if sink is not None else ResultQueue(maxlen=1000)
    reporter = reporter or FakeReporter()
    player = DummyPlayer(
        cache,
        registrar,
        sink,
        reporter,
        clock=fake.clock,
        waiter=fake.wait,
        inline_registration=kwargs.pop("inline_registration", True),
        **kwargs,
    )
    return player, registrar, sink, reporter


def run(player, stop=None):
    return player.play(CAMERA, SOURCE, stop or threading.Event())


def wait_until(condition, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def ats(items):
    return [item["at"] for item in items]


def test_first_result_is_sent_at_the_next_second_boundary(cache):
    publish(cache, [second_result(s) for s in range(5)], duration=5.0)
    fake = FakeClock(epoch(14, 3, 5, 300000), max_waits=2)
    player, _, sink, _ = make_player(cache, fake)
    run(player)
    items = sink.drain()
    assert ats(items) == ["2026-10-12T14:03:06+09:00"]
    assert fake.waits[0] == pytest.approx(0.7, abs=0.01)  # 5.3초 -> 6.0초 경계까지
    assert items[0]["personCount"] == 0  # k = 0


def test_one_result_per_second_with_the_matching_cache_entry(cache):
    publish(cache, [second_result(s) for s in range(5)], duration=5.0)
    fake = FakeClock(epoch(14, 3, 5), max_waits=5)
    player, _, sink, _ = make_player(cache, fake)
    run(player)
    items = sink.drain()
    assert ats(items) == [f"2026-10-12T14:03:{s:02d}+09:00" for s in range(6, 10)]
    assert [item["personCount"] for item in items] == [0, 10, 20, 30]
    assert items[1] == {
        "cameraId": CAMERA,
        "at": "2026-10-12T14:03:07+09:00",
        "personCount": 10,
        "in": {**ZERO, "N": 1},
        "out": ZERO,
        "dwellSec": [],
    }


def test_at_minus_started_at_equals_the_position_in_the_video(cache):
    publish(cache, [second_result(s) for s in range(4)], duration=4.0)
    fake = FakeClock(epoch(14, 3, 5, 800000), max_waits=4)
    player, registrar, sink, _ = make_player(cache, fake)
    run(player)
    assert wait_until(lambda: len(registrar.calls) == 1)
    started_ms = registrar.calls[0][1]
    start = datetime.fromtimestamp(started_ms / 1000, KST)
    assert start.microsecond == 0  # 초 경계에 맞춘 S
    for item in sink.drain():
        at = datetime.fromisoformat(item["at"])
        assert (at - start).total_seconds() == item["personCount"] / 10


def test_recording_is_registered_once_per_repetition_with_paths(cache):
    publish(cache, [second_result(s) for s in range(3)], duration=3.0)
    fake = FakeClock(epoch(14, 3, 0), max_waits=8)
    player, registrar, _, _ = make_player(cache, fake)
    run(player)
    assert len(registrar.calls) >= 2
    first, second = registrar.calls[:2]
    assert first[0] == CAMERA
    assert first[2:] == (SOURCE, "dummy/boxes/clip.json.gz")
    assert second[1] - first[1] == 3000  # 3초짜리 영상이라 3초 뒤에 다음 반복


def test_repetition_restarts_when_k_reaches_the_video_length(cache):
    publish(cache, [second_result(s) for s in range(3)], duration=3.0)
    fake = FakeClock(epoch(14, 3, 0), max_waits=8)
    player, _, sink, _ = make_player(cache, fake)
    run(player)
    assert [item["personCount"] for item in sink.drain()] == [0, 10, 20, 0, 10, 20, 0]


def test_fractional_video_length_rounds_the_repetition_up_to_whole_seconds(cache):
    # 2.5초 영상: k=0,1,2는 재생(2는 0.5초만 남은 마지막 조각), k=3 >= 2.5이면 새 반복
    publish(cache, [second_result(s) for s in range(3)], duration=2.5)
    fake = FakeClock(epoch(14, 3, 0), max_waits=8)
    player, _, sink, _ = make_player(cache, fake)
    run(player)
    assert [item["personCount"] for item in sink.drain()] == [0, 10, 20, 0, 10, 20, 0]


def test_video_shorter_than_a_second_restarts_every_second(cache):
    publish(cache, [second_result(0, count=4)], duration=0.5)
    fake = FakeClock(epoch(14, 3, 0), max_waits=5)
    player, registrar, sink, _ = make_player(cache, fake)
    run(player)
    assert [item["personCount"] for item in sink.drain()] == [4, 4, 4, 4]
    assert wait_until(lambda: len(registrar.calls) == 4)


def test_missing_duration_falls_back_to_the_number_of_seconds(cache):
    publish(cache, [second_result(s) for s in range(2)])  # durationSec 없음
    fake = FakeClock(epoch(14, 3, 0), max_waits=6)
    player, _, sink, _ = make_player(cache, fake)
    run(player)
    assert [item["personCount"] for item in sink.drain()] == [0, 10, 0, 10, 0]


def test_late_wakeup_fills_in_the_missed_seconds_in_order(cache):
    publish(cache, [second_result(s) for s in range(10)], duration=10.0)
    fake = FakeClock(epoch(14, 3, 0), max_waits=3)
    player, _, sink, _ = make_player(cache, fake)
    # 첫 깨어남 뒤 3.2초 멈췄다가 깨어남
    original_wait = fake.wait

    def slow_wait(stop, timeout):
        if len(fake.waits) == 1:
            fake.now += 3.2
        return original_wait(stop, timeout)

    player = DummyPlayer(
        cache,
        FakeRegistrar(),
        sink,
        FakeReporter(),
        clock=fake.clock,
        waiter=slow_wait,
    )
    run(player)
    items = sink.drain()
    assert ats(items) == [f"2026-10-12T14:03:{s:02d}+09:00" for s in range(1, 6)]
    assert [item["personCount"] for item in items] == [0, 10, 20, 30, 40]


def test_catch_up_across_a_repetition_boundary_registers_the_new_repetition(cache):
    publish(cache, [second_result(s) for s in range(3)], duration=3.0)
    fake = FakeClock(epoch(14, 3, 0), max_waits=3)
    registrar = FakeRegistrar()
    sink = ResultQueue(maxlen=100)
    original_wait = fake.wait

    def slow_wait(stop, timeout):
        if len(fake.waits) == 1:
            fake.now += 3.0
        return original_wait(stop, timeout)

    player = DummyPlayer(
        cache,
        registrar,
        sink,
        FakeReporter(),
        clock=fake.clock,
        waiter=slow_wait,
        inline_registration=True,
    )
    run(player)
    assert [item["personCount"] for item in sink.drain()] == [0, 10, 20, 0, 10]
    assert len(registrar.calls) == 2


def test_very_long_stall_skips_ahead_with_a_warning(cache, caplog):
    publish(cache, [second_result(s) for s in range(100)], duration=100.0)
    fake = FakeClock(epoch(14, 3, 0), max_waits=3)
    original_wait = fake.wait

    def stalled(stop, timeout):
        if len(fake.waits) == 1:
            fake.now += 60.0
        return original_wait(stop, timeout)

    player = DummyPlayer(
        cache,
        FakeRegistrar(),
        sink := ResultQueue(maxlen=1000),
        FakeReporter(),
        clock=fake.clock,
        waiter=stalled,
    )
    with caplog.at_level(logging.WARNING):
        run(player)
    assert len(sink.drain()) <= 7  # 60초치를 한꺼번에 쏟지 않는다
    assert "건너" in caplog.text


def test_same_second_or_clock_going_back_sends_nothing_twice(cache):
    publish(cache, [second_result(s) for s in range(5)], duration=5.0)
    fake = FakeClock(epoch(14, 3, 0), max_waits=5)
    original_wait = fake.wait

    def going_back(stop, timeout):
        stopped = original_wait(stop, timeout)
        if len(fake.waits) == 2:
            fake.now -= 1.5  # 시계가 되돌아감
        return stopped

    player = DummyPlayer(
        cache,
        FakeRegistrar(),
        sink := ResultQueue(maxlen=100),
        FakeReporter(),
        clock=fake.clock,
        waiter=going_back,
    )
    run(player)
    seen = ats(sink.drain())
    assert len(seen) == len(set(seen))


def test_stop_ends_the_loop_and_returns_stopped(cache):
    publish(cache, [second_result(0)], duration=1.0)
    fake = FakeClock(epoch(14, 3, 0), max_waits=100)
    player, _, _, _ = make_player(cache, fake)
    stop = threading.Event()
    stop.set()
    assert run(player, stop) == PlayOutcome.STOPPED
    assert fake.waits == []


def test_play_returns_stopped_when_stopped_while_waiting(cache):
    publish(cache, [second_result(0)], duration=1.0)
    fake = FakeClock(epoch(14, 3, 0), max_waits=2)
    player, _, _, _ = make_player(cache, fake)
    assert run(player) == PlayOutcome.STOPPED


def test_real_clock_stops_promptly(cache):
    publish(cache, [second_result(0)], duration=1.0)
    player = DummyPlayer(cache, FakeRegistrar(), ResultQueue(maxlen=10), FakeReporter())
    stop = threading.Event()
    result = {}
    thread = threading.Thread(target=lambda: result.setdefault("o", run(player, stop)))
    thread.start()
    time.sleep(0.2)
    started = time.time()
    stop.set()
    thread.join(timeout=3)
    assert result["o"] == PlayOutcome.STOPPED
    assert time.time() - started < 1.0


@pytest.mark.parametrize(
    "content",
    [
        "{깨진 json",
        '{"version": 1, "fps": 5, "seconds": []}',
        '{"version": 1, "fps": 5, "seconds": [{"sec": 1}]}',
        '{"version": 1, "fps": 5}',
        '{"version": 1, "fps": 5, "seconds": [{"sec": 0, "personCount": "x"}]}',
    ],
)
def test_broken_cache_reports_error_and_is_removed(cache, content):
    publish(cache, [second_result(0)], duration=1.0)
    cache.result_path(SOURCE).write_text(content, encoding="utf-8")
    fake = FakeClock(epoch(14, 3, 0), max_waits=3)
    player, _, sink, reporter = make_player(cache, fake)
    assert run(player) == PlayOutcome.FAILED
    assert reporter.calls == [(CAMERA, "ERROR", None, "캐시 파일을 읽을 수 없음")]
    assert cache.has(SOURCE) is False  # 다음 시작 때 다시 준비하게 한다
    assert sink.drain() == []


def test_missing_cache_reports_error(cache):
    fake = FakeClock(epoch(14, 3, 0), max_waits=3)
    player, _, _, reporter = make_player(cache, fake)
    assert run(player) == PlayOutcome.FAILED
    assert reporter.calls[0][1:] == ("ERROR", None, "캐시 파일을 읽을 수 없음")


def test_registration_does_not_block_the_per_second_results(cache):
    publish(cache, [second_result(s) for s in range(10)], duration=10.0)
    fake = FakeClock(epoch(14, 3, 0), max_waits=5)
    registrar = FakeRegistrar()
    registrar.block = threading.Event()  # 관리자 앱이 응답하지 않는 상황
    player, _, sink, _ = make_player(
        cache, fake, registrar=registrar, inline_registration=False
    )
    try:
        run(player)
        assert len(sink.drain()) == 4  # 등록이 막혀 있어도 결과는 매초 나간다
    finally:
        registrar.block.set()


# --- 녹화 등록 작업 스레드(RegistrationWorker) ---


def make_worker(registrar, **kwargs):
    stop = kwargs.pop("stop", threading.Event())
    worker = RegistrationWorker(
        registrar,
        CAMERA,
        SOURCE,
        "dummy/boxes/clip.json.gz",
        stop,
        retry_interval=kwargs.pop("retry_interval", 0.01),
        retry_window=kwargs.pop("retry_window", 60.0),
    )
    worker.start()
    return worker


def test_worker_submit_does_not_block_while_the_server_is_slow():
    registrar = FakeRegistrar()
    registrar.block = threading.Event()
    worker = make_worker(registrar)
    try:
        started = time.time()
        worker.submit(1_000)
        worker.submit(2_000)
        assert time.time() - started < 0.5
    finally:
        registrar.block.set()
        worker.shutdown()


def test_worker_retries_a_transient_failure_with_the_same_start():
    registrar = FakeRegistrar([RegisterResult.RETRY, RegisterResult.RETRY])
    worker = make_worker(registrar)
    try:
        worker.submit(5_000)
        assert wait_until(lambda: len(registrar.calls) == 3)
        assert {call[1] for call in registrar.calls} == {5_000}
    finally:
        worker.shutdown()


def test_worker_does_not_retry_a_rejected_registration(caplog):
    registrar = FakeRegistrar([RegisterResult.REJECTED])
    with caplog.at_level(logging.ERROR):
        worker = make_worker(registrar)
        try:
            worker.submit(5_000)
            assert wait_until(lambda: "녹화 등록 거절" in caplog.text)
            time.sleep(0.1)
            assert len(registrar.calls) == 1
        finally:
            worker.shutdown()


def test_worker_keeps_the_repetition_order_even_when_retrying():
    registrar = FakeRegistrar([RegisterResult.RETRY])
    worker = make_worker(registrar)
    try:
        worker.submit(1_000)
        worker.submit(2_000)
        assert wait_until(lambda: len(registrar.calls) == 3)
        assert [call[1] for call in registrar.calls] == [1_000, 1_000, 2_000]
    finally:
        worker.shutdown()


def test_worker_gives_up_after_the_retry_window_and_moves_on(caplog):
    registrar = FakeRegistrar([RegisterResult.RETRY] * 10_000)
    with caplog.at_level(logging.ERROR):
        worker = make_worker(registrar, retry_window=0.1)
        try:
            worker.submit(1_000)
            assert wait_until(lambda: "녹화 등록 포기" in caplog.text)
            registrar.results.clear()  # 이제 서버가 살아남
            worker.submit(2_000)
            assert wait_until(lambda: registrar.calls[-1][1] == 2_000)
        finally:
            worker.shutdown()


def test_worker_stops_retrying_after_shutdown():
    registrar = FakeRegistrar([RegisterResult.RETRY] * 10_000)
    worker = make_worker(registrar)
    worker.submit(1_000)
    assert wait_until(lambda: len(registrar.calls) >= 2)
    worker.shutdown()
    count = len(registrar.calls)
    time.sleep(0.15)
    assert len(registrar.calls) == count


def test_worker_stops_retrying_when_the_camera_is_stopped():
    stop = threading.Event()
    registrar = FakeRegistrar([RegisterResult.RETRY] * 10_000)
    worker = make_worker(registrar, stop=stop)
    try:
        worker.submit(1_000)
        assert wait_until(lambda: len(registrar.calls) >= 2)
        stop.set()
        time.sleep(0.1)
        count = len(registrar.calls)
        time.sleep(0.15)
        assert len(registrar.calls) == count
    finally:
        worker.shutdown()


def test_worker_drops_the_oldest_pending_registration_when_full(caplog):
    registrar = FakeRegistrar()
    registrar.block = threading.Event()  # 첫 요청이 응답하지 않아 뒤가 밀림
    worker = RegistrationWorker(
        registrar,
        CAMERA,
        SOURCE,
        "dummy/boxes/clip.json.gz",
        threading.Event(),
        retry_interval=0.01,
        max_pending=2,
    )
    worker.start()
    try:
        with caplog.at_level(logging.WARNING):
            worker.submit(1_000)
            time.sleep(0.1)  # 작업 스레드가 1_000을 집어 가서 막혀 있음
            for ms in (2_000, 3_000, 4_000, 5_000):
                worker.submit(ms)
        assert "가장 오래된 것을 버림" in caplog.text
    finally:
        registrar.block.set()
        assert wait_until(lambda: len(registrar.calls) >= 3)
        worker.shutdown()
    sent = [call[1] for call in registrar.calls]
    assert sent[0] == 1_000
    assert 2_000 not in sent and 3_000 not in sent  # 가장 오래된 둘은 버려짐
    assert 5_000 in sent


def test_worker_retry_window_includes_the_time_spent_waiting_in_the_queue(caplog):
    registrar = FakeRegistrar()
    registrar.block = threading.Event()
    worker = make_worker(registrar, retry_window=0.15)
    try:
        with caplog.at_level(logging.ERROR):
            worker.submit(1_000)
            time.sleep(0.05)
            worker.submit(2_000)  # 첫 요청 뒤에서 기다리는 동안 기한이 흐른다
            time.sleep(0.3)
            registrar.block.set()
            assert wait_until(lambda: "녹화 등록 포기" in caplog.text)
        assert [call[1] for call in registrar.calls] == [1_000]  # 2_000은 보내지 않음
    finally:
        registrar.block.set()
        worker.shutdown()


class _BrokenCache(DummyCache):
    def invalidate(self, source_uri):
        raise PermissionError("삭제 권한 없음")


def test_error_is_reported_even_if_the_broken_cache_cannot_be_removed(tmp_path):
    broken = _BrokenCache(cache_dir=tmp_path / "c", media_root=tmp_path / "m")
    publish(broken, [second_result(0)], duration=1.0)
    broken.result_path(SOURCE).write_text("{깨진 json", encoding="utf-8")
    fake = FakeClock(epoch(14, 3, 0), max_waits=3)
    player, _, _, reporter = make_player(broken, fake)
    assert run(player) == PlayOutcome.FAILED
    assert reporter.calls == [(CAMERA, "ERROR", None, "캐시 파일을 읽을 수 없음")]


def test_stop_in_the_middle_of_catching_up_sends_no_more_seconds(cache):
    publish(cache, [second_result(s) for s in range(10)], duration=10.0)
    fake = FakeClock(epoch(14, 3, 0), max_waits=3)
    stop = threading.Event()

    class StoppingSink(ResultQueue):
        def put(self, item):
            super().put(item)
            stop.set()  # 첫 결과를 넣자마자 정지가 들어옴

    original_wait = fake.wait

    def stall(stop_event, timeout):
        if len(fake.waits) == 1:
            fake.now += 3.0
        return original_wait(stop_event, timeout)

    sink = StoppingSink(maxlen=100)
    player = DummyPlayer(
        cache,
        FakeRegistrar(),
        sink,
        FakeReporter(),
        clock=fake.clock,
        waiter=stall,
        inline_registration=True,
    )
    run(player, stop)
    assert len(sink.drain()) == 1
