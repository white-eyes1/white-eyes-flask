"""FNC-S-006 1초 결과 전송·재전송 (ResultSender)"""

import logging
import threading
import time

import pytest

from white_eyes_flask.camera.kst import format_at
from white_eyes_flask.camera.result_client import SendResult
from white_eyes_flask.camera.result_queue import ResultQueue
from white_eyes_flask.camera.result_sender import ResultSender

T0 = 1_760_000_000  # 임의의 기준 시각(초)
RETRY_SEC = 300


def item(camera_id, epoch, count=1):
    return {
        "cameraId": camera_id,
        "at": format_at(epoch),
        "personCount": count,
        "in": {"N": 0, "E": 0, "S": 0, "W": 0},
        "out": {"N": 0, "E": 0, "S": 0, "W": 0},
        "dwellSec": [],
    }


class FakeClient:
    """보낸 묶음을 기록하고, 정해 둔 결과를 차례로 돌려준다(다 쓰면 OK)."""

    def __init__(self, results=()):
        self.batches = []
        self.results = list(results)
        self.on_send = None

    def send(self, items):
        self.batches.append(list(items))
        if self.on_send:
            self.on_send(len(self.batches))
        result = self.results.pop(0) if self.results else SendResult.OK
        if isinstance(result, Exception):
            raise result
        return result


class Clock:
    def __init__(self, now=T0):
        self.now = now

    def __call__(self):
        return self.now


def make(client=None, **kwargs):
    queue = ResultQueue()
    clock = kwargs.pop("clock", None) or Clock()
    client = client or FakeClient()
    sender = ResultSender(
        queue, client, retry_buffer_sec=RETRY_SEC, clock=clock, **kwargs
    )
    return sender, queue, client, clock


def test_new_results_are_sent_in_one_request_in_arrival_order():
    sender, queue, client, _ = make()
    first, second = item(1, T0 - 1), item(2, T0 - 1)
    queue.put(first)
    queue.put(second)
    sender.tick()
    assert client.batches == [[first, second]]
    assert len(queue) == 0
    assert sender.pending == 0


def test_nothing_is_sent_when_there_is_nothing():
    sender, _, client, _ = make()
    sender.tick()
    assert client.batches == []


def test_failed_results_are_kept_and_sent_again_with_new_ones():
    sender, queue, client, _ = make(FakeClient([SendResult.RETRY]))
    old = item(1, T0 - 2)
    queue.put(old)
    sender.tick()
    assert sender.pending == 1
    new = item(1, T0 - 1)
    queue.put(new)
    sender.tick()
    assert client.batches == [[old], [old, new]]  # 순서를 지키고 내용은 그대로
    assert sender.pending == 0


def test_rejected_batch_is_dropped_and_logged_with_its_size(caplog):
    sender, queue, client, _ = make(FakeClient([SendResult.REJECTED]))
    queue.put(item(1, T0 - 2))
    queue.put(item(1, T0 - 1))
    with caplog.at_level(logging.ERROR):
        sender.tick()
    assert sender.pending == 0
    assert "2" in caplog.text
    queue.put(item(1, T0))
    sender.tick()
    assert len(client.batches[-1]) == 1  # 버린 것은 다시 오지 않는다


def test_results_older_than_the_retry_window_are_dropped_not_sent():
    sender, queue, client, _ = make()
    stale = item(1, T0 - RETRY_SEC - 1)
    edge = item(1, T0 - RETRY_SEC)  # 딱 300초 전은 아직 보관
    fresh = item(1, T0 - 1)
    for result in (stale, edge, fresh):
        queue.put(result)
    sender.tick()
    assert client.batches == [[edge, fresh]]


def test_buffered_results_expire_while_the_server_is_down():
    sender, queue, client, clock = make(FakeClient([SendResult.RETRY]))
    queue.put(item(1, T0))
    sender.tick()
    assert sender.pending == 1
    clock.now = T0 + RETRY_SEC + 1  # 5분 넘게 꺼져 있었다
    sender.tick()
    assert sender.pending == 0
    assert len(client.batches) == 1  # 만료된 것은 다시 보내지 않는다


def test_two_minutes_of_outage_are_filled_when_the_server_comes_back():
    results = [SendResult.RETRY] * 120
    sender, queue, client, clock = make(FakeClient(results))
    expected = []
    for second in range(120):
        clock.now = T0 + second
        result = item(7, T0 + second)
        expected.append(result)
        queue.put(result)
        sender.tick()
    clock.now = T0 + 120
    sender.tick()  # 서버가 돌아옴
    assert client.batches[-1] == expected
    assert sender.pending == 0


def test_expiry_is_checked_again_between_batches():
    sender, queue, client, clock = make(batch_size=2)
    for epoch in (T0 - RETRY_SEC + 1, T0 - RETRY_SEC + 1, T0 - RETRY_SEC + 1, T0):
        queue.put(item(1, epoch))
    # 첫 묶음을 보내는 동안 2초가 지나 두 번째 묶음의 앞 항목(T0-299)이 만료된다(기준 T0-298)
    client.on_send = lambda n: setattr(clock, "now", T0 + 2) if n == 1 else None
    sender.tick()
    assert len(client.batches[0]) == 2
    assert [r["at"] for r in client.batches[1]] == [format_at(T0)]


def test_large_backlog_is_split_into_batches_of_at_most_batch_size():
    sender, queue, client, _ = make(batch_size=2000, max_batches_per_tick=10)
    for index in range(4500):
        queue.put(item(1, T0 - 1, count=index))
    sender.tick()
    assert [len(b) for b in client.batches] == [2000, 2000, 500]
    flat = [r["personCount"] for batch in client.batches for r in batch]
    assert flat == list(range(4500))


def test_requests_per_tick_are_capped_and_the_rest_waits_for_the_next_tick():
    sender, queue, client, _ = make(batch_size=2, max_batches_per_tick=2)
    for index in range(7):
        queue.put(item(1, T0 - 1, count=index))
    sender.tick()
    assert [len(b) for b in client.batches] == [2, 2]
    assert sender.pending == 3
    sender.tick()
    assert [len(b) for b in client.batches] == [2, 2, 2, 1]


def test_a_failure_stops_the_tick_without_trying_later_batches():
    sender, queue, client, _ = make(
        FakeClient([SendResult.OK, SendResult.RETRY]), batch_size=2
    )
    for index in range(6):
        queue.put(item(1, T0 - 1, count=index))
    sender.tick()
    assert len(client.batches) == 2  # 두 번째에서 실패하고 멈춤
    assert sender.pending == 4


def test_a_rejected_batch_does_not_stop_the_following_batches():
    sender, queue, client, _ = make(
        FakeClient([SendResult.REJECTED, SendResult.OK]), batch_size=2
    )
    for index in range(4):
        queue.put(item(1, T0 - 1, count=index))
    sender.tick()
    assert len(client.batches) == 2
    assert sender.pending == 0


def test_results_with_an_unreadable_time_are_dropped_and_the_rest_sent(caplog):
    sender, queue, client, _ = make()
    good = item(1, T0 - 1)
    queue.put({**item(1, T0 - 1), "at": "어제"})
    queue.put({**item(1, T0 - 1), "at": "2026-10-12T14:03:05"})  # 시간대 없음
    queue.put({"cameraId": 1})  # at이 없음
    queue.put(good)
    with caplog.at_level(logging.WARNING):
        sender.tick()
    assert client.batches == [[good]]
    assert "버림" in caplog.text


def test_buffer_is_capped_and_the_oldest_are_dropped(caplog):
    sender, queue, client, clock = make(
        FakeClient([SendResult.RETRY] * 3), max_buffer=3
    )
    for index in range(5):
        queue.put(item(1, T0 - 1, count=index))
    with caplog.at_level(logging.WARNING):
        sender.tick()
    assert sender.pending == 3
    queue.put(item(1, T0 - 1, count=5))
    sender.tick()
    assert [r["personCount"] for r in client.batches[-1]] == [3, 4, 5]
    assert "버림" in caplog.text


def test_results_are_not_sent_twice_after_success():
    sender, queue, client, _ = make()
    queue.put(item(1, T0 - 1))
    sender.tick()
    sender.tick()
    assert len(client.batches) == 1


# ---- 스레드 ----------------------------------------------------------------


def wait_for(condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def real_clock_sender(client, interval=0.05, **kwargs):
    queue = ResultQueue()
    sender = ResultSender(
        queue, client, retry_buffer_sec=RETRY_SEC, interval=interval, **kwargs
    )
    return sender, queue


def test_thread_sends_results_every_interval():
    client = FakeClient()
    sender, queue = real_clock_sender(client)
    sender.start()
    try:
        queue.put(item(1, int(time.time())))
        assert wait_for(lambda: len(client.batches) >= 1)
        queue.put(item(1, int(time.time())))
        assert wait_for(lambda: len(client.batches) >= 2)
    finally:
        sender.stop()


def test_thread_survives_an_unexpected_error_and_keeps_the_results(caplog):
    client = FakeClient([RuntimeError("깨짐")])
    sender, queue = real_clock_sender(client)
    queue.put(item(1, int(time.time())))
    sender.start()
    try:
        assert wait_for(lambda: len(client.batches) >= 2)  # 실패 뒤 다시 보냄
    finally:
        sender.stop()
    assert "깨짐" in caplog.text or "전송 중 오류" in caplog.text
    assert client.batches[0] == client.batches[1]


def test_stop_makes_a_last_attempt_to_send_what_is_left():
    client = FakeClient()
    sender, queue = real_clock_sender(client, interval=30.0)
    sender.start()
    assert wait_for(lambda: not sender._stop.is_set())  # 스레드가 대기에 들어감
    queue.put(item(1, int(time.time())))
    sender.stop()
    assert len(client.batches) == 1


def test_stop_ends_the_thread_even_if_the_server_keeps_failing():
    client = FakeClient([SendResult.RETRY] * 100)
    sender, queue = real_clock_sender(client)
    queue.put(item(1, int(time.time())))
    sender.start()
    time.sleep(0.15)
    sender.stop(timeout=2.0)
    assert not sender.is_running


def test_start_twice_does_not_start_a_second_thread():
    sender, _ = real_clock_sender(FakeClient())
    sender.start()
    try:
        threads_before = threading.active_count()
        sender.start()
        assert threading.active_count() == threads_before
    finally:
        sender.stop()


def test_stop_before_start_is_harmless():
    sender, _ = real_clock_sender(FakeClient())
    sender.stop()
    assert not sender.is_running


def test_can_be_started_again_after_stop():
    client = FakeClient()
    sender, queue = real_clock_sender(client)
    sender.start()
    sender.stop()
    sender.start()
    try:
        queue.put(item(1, int(time.time())))
        assert wait_for(lambda: len(client.batches) >= 1)
    finally:
        sender.stop()


def test_expired_results_are_removed_before_the_buffer_cap_is_applied():
    sender, queue, client, _ = make(FakeClient([SendResult.RETRY]), max_buffer=3)
    fresh = [item(1, T0 - 1, count=index) for index in range(3)]
    for result in fresh:
        queue.put(result)
    for index in range(5):
        queue.put(item(1, T0 - RETRY_SEC - 10, count=100 + index))  # 이미 만료
    sender.tick()
    assert client.batches == [fresh]  # 멀쩡한 결과가 상한 때문에 밀려나지 않는다


def test_a_result_that_cannot_be_turned_into_json_does_not_block_the_others(caplog):
    sender, queue, client, _ = make()
    good = item(1, T0 - 1)
    queue.put({**item(1, T0 - 1), "dwellSec": {1, 2}})  # set은 JSON이 되지 않는다
    queue.put(good)
    with caplog.at_level(logging.WARNING):
        sender.tick()
    assert client.batches == [[good]]
    assert "버림" in caplog.text


def test_a_time_out_of_range_does_not_break_the_tick():
    sender, queue, client, _ = make()
    good = item(1, T0 - 1)
    queue.put({**item(1, T0 - 1), "at": "0001-01-01T00:00:00+09:00"})
    queue.put({**item(1, T0 - 1), "at": "9999-12-31T23:59:59-23:59"})
    queue.put(good)
    sender.tick()
    assert client.batches == [[good]]


def test_the_last_attempt_sends_only_one_request_even_with_a_backlog():
    sender, queue, client, _ = make(batch_size=2, max_batches_per_tick=5)
    for index in range(10):
        queue.put(item(1, T0 - 1, count=index))
    sender.tick(final=True)
    assert len(client.batches) == 1


def test_start_from_many_threads_at_once_runs_a_single_sender_thread():
    def sender_threads():
        return [t for t in threading.enumerate() if t.name == "result-sender"]

    before = len(sender_threads())
    sender, _ = real_clock_sender(FakeClient())
    starters = [threading.Thread(target=sender.start) for _ in range(8)]
    for starter in starters:
        starter.start()
    for starter in starters:
        starter.join(timeout=2)
    try:
        assert len(sender_threads()) == before + 1
    finally:
        sender.stop()


def test_batch_size_cannot_exceed_the_api_limit():
    ResultSender(ResultQueue(), FakeClient(), retry_buffer_sec=300, batch_size=2000)
    with pytest.raises(ValueError):
        ResultSender(ResultQueue(), FakeClient(), retry_buffer_sec=300, batch_size=2001)


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError):
        ResultSender(ResultQueue(), FakeClient(), retry_buffer_sec=0)
    with pytest.raises(ValueError):
        ResultSender(ResultQueue(), FakeClient(), retry_buffer_sec=300, batch_size=0)
    with pytest.raises(ValueError):
        ResultSender(ResultQueue(), FakeClient(), retry_buffer_sec=300, interval=0)
