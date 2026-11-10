import logging
import threading

from white_eyes_flask.camera.result_queue import ResultQueue


def test_items_come_out_in_the_order_they_went_in():
    queue = ResultQueue(maxlen=10)
    for n in range(3):
        queue.put({"n": n})
    assert queue.drain() == [{"n": 0}, {"n": 1}, {"n": 2}]
    assert len(queue) == 0


def test_drain_can_take_a_limited_number():
    queue = ResultQueue(maxlen=10)
    for n in range(5):
        queue.put({"n": n})
    assert queue.drain(2) == [{"n": 0}, {"n": 1}]
    assert len(queue) == 3


def test_oldest_items_are_dropped_with_a_warning_when_full(caplog):
    queue = ResultQueue(maxlen=3)
    with caplog.at_level(logging.WARNING):
        for n in range(5):
            queue.put({"n": n})
    assert [item["n"] for item in queue.drain()] == [2, 3, 4]
    assert queue.dropped == 2
    assert "전송 큐가 가득" in caplog.text


def test_nothing_is_lost_with_concurrent_put_and_drain():
    queue = ResultQueue(maxlen=100000)
    drained = []
    stop = threading.Event()

    def drainer():
        while not stop.is_set():
            drained.extend(queue.drain())

    thread = threading.Thread(target=drainer)
    thread.start()
    try:
        for n in range(2000):
            queue.put({"n": n})
    finally:
        stop.set()
        thread.join()
    drained.extend(queue.drain())
    assert [item["n"] for item in drained] == list(range(2000))
