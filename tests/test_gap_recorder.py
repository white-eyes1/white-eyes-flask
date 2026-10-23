"""FNC-S-008 스트림이 끊겼다 복구되면 녹화 파일을 새로 시작하는 녹화기(GapSplittingRecorder)"""

import logging
import threading
import time

import pytest

from white_eyes_flask.recording import gap_recorder
from white_eyes_flask.recording.gap_recorder import GapSplittingRecorder
from white_eyes_flask.recording.recorder import RecorderUnavailable


class FakeRecorder:
    """LiveRecorder 대역. 받은 호출을 기록한다."""

    created: list["FakeRecorder"] = []

    def __init__(self, close_block=None, close_error=None, write_error=None):
        self.frames = []
        self.boxes = []
        self.closed = threading.Event()
        self.close_started = threading.Event()
        self._block = close_block
        self._close_error = close_error
        self._write_error = write_error
        FakeRecorder.created.append(self)

    def write_frame(self, frame, at):
        if self._write_error:
            raise self._write_error
        self.frames.append(at)

    def add_boxes(self, at, boxes):
        self.boxes.append(at)

    def close(self):
        self.close_started.set()
        if self._block is not None:
            self._block.wait(5)
        self.closed.set()
        if self._close_error:
            raise self._close_error


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture(autouse=True)
def reset_created():
    FakeRecorder.created = []


def make(clock, gap=5.0, **fake_kwargs):
    recorder = GapSplittingRecorder(
        7, lambda: FakeRecorder(**fake_kwargs), gap_sec=gap, clock=clock
    )
    return recorder


def test_the_first_frame_starts_one_recorder_and_close_closes_it():
    clock = Clock()
    recorder = make(clock)
    recorder.write_frame("f", 1.0)
    recorder.close()
    assert len(FakeRecorder.created) == 1
    assert FakeRecorder.created[0].frames == [1.0]
    assert FakeRecorder.created[0].closed.is_set()


def test_a_gap_below_the_limit_keeps_the_same_file():
    clock = Clock()
    recorder = make(clock)
    recorder.write_frame("f", 1.0)
    clock.advance(4.9)
    recorder.write_frame("f", 5.9)
    recorder.close()
    assert len(FakeRecorder.created) == 1
    assert FakeRecorder.created[0].frames == [1.0, 5.9]


@pytest.mark.parametrize("gap", [5.0, 5.1, 74.0])
def test_a_gap_at_or_above_the_limit_starts_a_new_file(gap):
    clock = Clock()
    recorder = make(clock)
    recorder.write_frame("f", 1.0)
    clock.advance(gap)
    recorder.write_frame("f", 1.0 + gap)
    old, new = FakeRecorder.created
    assert old.closed.wait(2)  # 끊기기 전 파일은 마무리된다
    assert old.frames == [1.0] and new.frames == [1.0 + gap]
    assert not new.closed.is_set()
    recorder.close()
    assert new.closed.is_set()


def test_the_gap_is_measured_between_consecutive_frames_not_from_the_start():
    clock = Clock()
    recorder = make(clock)
    for step in range(20):  # 총 40초 동안 2초 간격으로 계속 들어옴
        recorder.write_frame("f", float(step))
        clock.advance(2.0)
    recorder.close()
    assert len(FakeRecorder.created) == 1


def test_reading_resumes_without_waiting_for_the_old_file_to_close():
    clock = Clock()
    release = threading.Event()
    created = []

    def factory():
        # 첫 번째 녹화기만 닫는 데 오래 걸린다
        rec = FakeRecorder(close_block=release if not created else None)
        created.append(rec)
        return rec

    recorder = GapSplittingRecorder(7, factory, gap_sec=5.0, clock=clock)
    recorder.write_frame("f", 1.0)
    clock.advance(10)
    started = time.monotonic()
    recorder.write_frame("f", 11.0)  # 이전 파일이 아직 닫히는 중이어도 막히지 않는다
    assert time.monotonic() - started < 1.0
    old, new = created
    assert old.close_started.wait(2) and not old.closed.is_set()
    assert new.frames == [11.0]
    release.set()
    recorder.close()
    assert old.closed.is_set()


def test_boxes_go_to_the_current_file_and_are_ignored_before_the_first_frame():
    clock = Clock()
    recorder = make(clock)
    recorder.add_boxes(0.5, [])  # 아직 녹화가 시작되지 않았다
    recorder.write_frame("f", 1.0)
    recorder.add_boxes(1.0, [])
    clock.advance(10)
    recorder.write_frame("f", 11.0)
    recorder.add_boxes(11.0, [])
    recorder.close()
    old, new = FakeRecorder.created
    assert old.boxes == [1.0]
    assert new.boxes == [11.0]


def test_close_is_idempotent_and_later_frames_are_ignored():
    clock = Clock()
    recorder = make(clock)
    recorder.write_frame("f", 1.0)
    recorder.close()
    recorder.close()
    recorder.write_frame("f", 2.0)
    assert len(FakeRecorder.created) == 1
    assert FakeRecorder.created[0].frames == [1.0]


def test_close_without_any_frame_does_nothing():
    clock = Clock()
    recorder = make(clock)
    recorder.close()
    assert FakeRecorder.created == []


def test_a_failure_closing_the_old_file_is_logged_and_does_not_stop_recording(caplog):
    clock = Clock()
    created = []

    def factory():
        rec = FakeRecorder(
            close_error=RuntimeError("닫기 실패") if not created else None
        )
        created.append(rec)
        return rec

    recorder = GapSplittingRecorder(7, factory, gap_sec=5.0, clock=clock)
    recorder.write_frame("f", 1.0)
    clock.advance(10)
    with caplog.at_level(logging.ERROR):
        recorder.write_frame("f", 11.0)
        assert created[0].closed.wait(2)
        recorder.close()
    assert created[1].frames == [11.0] and created[1].closed.is_set()
    assert any("끊김 앞 녹화 파일을 닫다 예외" in r.message for r in caplog.records)


def test_close_does_not_wait_forever_for_an_old_file_that_never_closes(
    monkeypatch, caplog
):
    monkeypatch.setattr(gap_recorder, "CLOSE_JOIN_SEC", 0.2)
    release = threading.Event()
    created = []

    def factory():
        rec = FakeRecorder(close_block=release if not created else None)
        created.append(rec)
        return rec

    clock = Clock()
    recorder = GapSplittingRecorder(7, factory, gap_sec=5.0, clock=clock)
    recorder.write_frame("f", 1.0)
    clock.advance(10)
    recorder.write_frame("f", 11.0)
    assert created[0].close_started.wait(2)
    with caplog.at_level(logging.WARNING):
        started = time.monotonic()
        recorder.close()
        elapsed = time.monotonic() - started
    release.set()
    assert elapsed < 1.0
    assert any("다 닫지 못한 채 정지" in r.message for r in caplog.records)


def test_a_new_file_that_cannot_start_raises_like_the_first_one():
    clock = Clock()
    created = []

    def factory():
        rec = FakeRecorder(
            write_error=RecorderUnavailable("ffmpeg 없음") if created else None
        )
        created.append(rec)
        return rec

    recorder = GapSplittingRecorder(7, factory, gap_sec=5.0, clock=clock)
    recorder.write_frame("f", 1.0)
    clock.advance(10)
    with pytest.raises(RecorderUnavailable):
        recorder.write_frame("f", 11.0)
    recorder.close()


def test_many_gaps_do_not_pile_up_unfinished_closers():
    clock = Clock()
    recorder = make(clock)
    for step in range(6):
        recorder.write_frame("f", float(step * 10))
        clock.advance(10)
    recorder.close()
    assert len(FakeRecorder.created) == 6
    assert all(r.closed.is_set() for r in FakeRecorder.created)
