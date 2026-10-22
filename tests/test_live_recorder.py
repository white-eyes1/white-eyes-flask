"""FNC-S-008 라이브 녹화: LiveRecorder(가짜 ffmpeg 프로세스로 검증)"""

import gzip
import json
import logging
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from white_eyes_flask.analysis.result import TrackedBox
from white_eyes_flask.camera.kst import KST
from white_eyes_flask.recording import recorder as recorder_module
from white_eyes_flask.recording.ffmpeg import RecorderSettings
from white_eyes_flask.recording.recorder import LiveRecorder, RecorderUnavailable

FRAME = np.zeros((4, 6, 3), dtype=np.uint8)


def at(hour, minute, second=0.0) -> float:
    moment = datetime(2026, 10, 12, hour, minute, 0, tzinfo=KST)
    return moment.timestamp() + second


class FakeStdin:
    def __init__(self, gate: threading.Event | None = None, fail: bool = False):
        self.chunks: list[bytes] = []
        self.closed = False
        self.on_close = None  # stdin이 닫히면 ffmpeg가 마지막 파일을 닫는 것을 흉내
        self._gate = gate
        self._fail = fail

    def write(self, data: bytes) -> None:
        if self._fail:
            raise BrokenPipeError
        if self._gate is not None:
            self._gate.wait(5)
        self.chunks.append(data)

    def close(self) -> None:
        self.closed = True
        if self.on_close is not None:
            self.on_close()


class FakeProc:
    """ffmpeg 흉내. 시험이 segment 파일과 list 줄을 직접 만든다."""

    def __init__(self, command: list[str], stdin: FakeStdin, returncode=0) -> None:
        self.command = command
        self.stdin = stdin
        self._returncode = returncode
        self.terminated = False
        self.killed = False
        self.hang = False

    @property
    def tmp_dir(self) -> Path:
        return Path(self.command[-1]).parent

    def segment(self, index: int) -> Path:
        pattern = self.command[-1]
        return Path(pattern.replace("%05d", f"{index:05d}"))

    def finish(self, index: int, partial_line: bool = False) -> None:
        """index번째 파일을 닫고 list에 올린다."""
        path = self.segment(index)
        path.write_bytes(b"mp4-%d" % index)
        listing = Path(self.command[self.command.index("-segment_list") + 1])
        line = f"{path.name},0.000,1.000"
        with open(listing, "a", encoding="utf-8") as stream:
            stream.write(line if partial_line else line + "\n")

    def wait(self, timeout=None):
        if self.hang and not self.terminated:
            raise subprocess.TimeoutExpired("ffmpeg", timeout)
        return self._returncode

    def poll(self):
        return self._returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


class FakePublisher:
    def __init__(self) -> None:
        self.items = []

    def submit(self, recording) -> None:
        self.items.append(recording)


class Harness:
    def __init__(self, tmp_path: Path, **proc_kwargs) -> None:
        self.tmp = tmp_path / "tmp"
        self.publisher = FakePublisher()
        self.proc: FakeProc | None = None
        self._proc_kwargs = proc_kwargs
        self.recorder = LiveRecorder(
            7,
            RecorderSettings(tmp_dir=str(self.tmp), analyze_fps=5),
            self.publisher,
            popen=self._popen,
        )

    def _popen(self, command, stdin=None, stdout=None, stderr=None):
        stdin_kwargs = {
            key: self._proc_kwargs[key]
            for key in ("gate", "fail")
            if key in self._proc_kwargs
        }
        self.proc = FakeProc(
            command,
            FakeStdin(**stdin_kwargs),
            self._proc_kwargs.get("returncode", 0),
        )
        self.proc.hang = self._proc_kwargs.get("hang", False)
        return self.proc


@pytest.fixture
def harness(tmp_path):
    return Harness(tmp_path)


def box(track_id=1, x=0.5) -> TrackedBox:
    return TrackedBox(track_id, x, 0.5, 0.1, 0.2, x + 0.05, 0.6, 0.9)


def read_boxes(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def test_a_short_recording_becomes_one_file_that_ends_at_the_last_frame(harness):
    rec = harness.recorder
    rec.write_frame(FRAME, at(14, 3, 27))
    rec.add_boxes(at(14, 3, 30), [box(1)])
    rec.write_frame(FRAME, at(14, 3, 50))
    rec.add_boxes(at(14, 3, 50), [box(2)])
    harness.proc.stdin.on_close = lambda: harness.proc.finish(0)
    rec.close()
    (item,) = harness.publisher.items
    assert item.camera_id == 7
    assert item.started_ms == int(at(14, 3, 27) * 1000)
    assert item.ended_ms == int(at(14, 3, 50) * 1000)
    body = read_boxes(item.boxes)
    assert body["fps"] == 5
    assert [f["t"] for f in body["frames"]] == [3.0, 23.0]
    assert body["frames"][0]["b"][0][0] == 1


def test_a_file_is_published_when_the_minute_changes(harness):
    rec = harness.recorder
    rec.write_frame(FRAME, at(14, 3, 50))
    rec.add_boxes(at(14, 3, 50), [box(1)])
    rec.write_frame(FRAME, at(14, 4, 5))
    rec.add_boxes(at(14, 4, 5), [box(2)])
    harness.proc.finish(0)  # 14:04 정각에 첫 파일이 닫혔다
    rec.poll_segments()
    (first,) = harness.publisher.items
    assert first.started_ms == int(at(14, 3, 50) * 1000)
    assert first.ended_ms == int(at(14, 4) * 1000)
    assert [f["t"] for f in read_boxes(first.boxes)["frames"]] == [0.0]
    assert first.video.name.endswith("_00000.mp4")


def test_second_file_starts_on_the_minute_and_keeps_its_own_boxes(harness):
    rec = harness.recorder
    rec.write_frame(FRAME, at(14, 3, 50))
    rec.add_boxes(at(14, 3, 50), [box(1)])
    rec.write_frame(FRAME, at(14, 4, 5))
    rec.add_boxes(at(14, 4, 5), [box(2)])
    rec.write_frame(FRAME, at(14, 4, 20))
    rec.add_boxes(at(14, 4, 20), [box(3)])
    harness.proc.finish(0)
    rec.poll_segments()
    harness.proc.stdin.on_close = lambda: harness.proc.finish(1)
    rec.close()
    first, second = harness.publisher.items
    assert second.started_ms == int(at(14, 4) * 1000)
    assert second.ended_ms == int(at(14, 4, 20) * 1000)
    frames = read_boxes(second.boxes)["frames"]
    assert [f["t"] for f in frames] == [5.0, 20.0]
    assert [f["b"][0][0] for f in frames] == [2, 3]
    assert second.video.name.endswith("_00001.mp4")


def test_a_full_minute_file_ends_on_the_minute_boundary(harness):
    rec = harness.recorder
    rec.write_frame(FRAME, at(14, 3))
    rec.write_frame(FRAME, at(14, 3, 59.5))
    rec.write_frame(FRAME, at(14, 4, 0.5))
    harness.proc.finish(0)
    rec.poll_segments()
    (item,) = harness.publisher.items
    assert item.started_ms == int(at(14, 3) * 1000)
    assert item.ended_ms == int(at(14, 4) * 1000)


def test_a_half_written_list_line_is_read_on_the_next_poll(harness):
    rec = harness.recorder
    rec.write_frame(FRAME, at(14, 3, 10))
    harness.proc.finish(0, partial_line=True)
    rec.poll_segments()
    assert harness.publisher.items == []
    listing = Path(
        harness.proc.command[harness.proc.command.index("-segment_list") + 1]
    )
    with open(listing, "a", encoding="utf-8") as stream:
        stream.write("\n")
    rec.poll_segments()
    assert len(harness.publisher.items) == 1


def test_each_file_is_published_only_once(harness):
    rec = harness.recorder
    rec.write_frame(FRAME, at(14, 3, 10))
    harness.proc.finish(0)
    rec.poll_segments()
    rec.poll_segments()
    assert len(harness.publisher.items) == 1


def test_frames_reach_ffmpeg_in_order(harness):
    rec = harness.recorder
    first = np.full((4, 6, 3), 1, dtype=np.uint8)
    second = np.full((4, 6, 3), 2, dtype=np.uint8)
    rec.write_frame(first, at(14, 3, 1))
    rec.write_frame(second, at(14, 3, 2))
    rec.close()
    assert harness.proc.stdin.chunks == [first.tobytes(), second.tobytes()]
    assert harness.proc.stdin.closed


def test_ffmpeg_is_given_the_frame_size_of_the_first_frame(harness):
    harness.recorder.write_frame(np.zeros((8, 12, 3), dtype=np.uint8), at(14, 3))
    command = harness.proc.command
    assert command[command.index("-s") + 1] == "12x8"
    harness.recorder.close()


def test_missing_ffmpeg_is_reported_clearly(tmp_path):
    def missing(*args, **kwargs):
        raise FileNotFoundError("ffmpeg")

    rec = LiveRecorder(
        7, RecorderSettings(tmp_dir=str(tmp_path)), FakePublisher(), popen=missing
    )
    with pytest.raises(RecorderUnavailable):
        rec.write_frame(FRAME, at(14, 3))
    rec.close()  # 시작하지 못했으면 닫을 것도 없다


def test_closing_without_any_frame_does_nothing(harness):
    harness.recorder.close()
    assert harness.proc is None
    assert harness.publisher.items == []


def test_closing_twice_is_harmless(harness):
    harness.recorder.write_frame(FRAME, at(14, 3))
    harness.recorder.close()
    harness.recorder.close()


def test_a_slow_ffmpeg_never_blocks_the_caller_and_frames_are_dropped(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(recorder_module, "QUEUE_FRAMES", 3)
    gate = threading.Event()
    harness = Harness(tmp_path, gate=gate)
    rec = harness.recorder
    started = time.monotonic()
    with caplog.at_level(logging.WARNING):
        for index in range(30):
            rec.write_frame(FRAME, at(14, 3, index * 0.1))
    assert time.monotonic() - started < 2.0
    assert any("프레임을 버림" in record.message for record in caplog.records)
    gate.set()
    rec.close()


def test_a_broken_pipe_does_not_stop_the_caller(tmp_path, caplog):
    harness = Harness(tmp_path, fail=True)
    rec = harness.recorder
    with caplog.at_level(logging.ERROR):
        for index in range(5):
            rec.write_frame(FRAME, at(14, 3, index))
        rec.close()
    assert any("쓸 수 없음" in record.message for record in caplog.records)


def test_an_ffmpeg_that_does_not_exit_is_terminated(tmp_path, monkeypatch):
    monkeypatch.setattr(recorder_module, "EXIT_WAIT_SEC", 0.1)
    harness = Harness(tmp_path, hang=True)
    harness.recorder.write_frame(FRAME, at(14, 3))
    harness.recorder.close()
    assert harness.proc.terminated


def test_a_failed_ffmpeg_is_logged_with_its_error_output(tmp_path, caplog):
    harness = Harness(tmp_path, returncode=1)
    harness.recorder.write_frame(FRAME, at(14, 3))
    with caplog.at_level(logging.ERROR):
        harness.recorder.close()
    assert any("오류로 끝남" in record.message for record in caplog.records)


def test_a_file_closed_while_recording_ends_on_the_minute_boundary(harness):
    # 분이 바뀌고 다음 프레임이 아직 안 들어왔어도 완성된 1분 파일이다
    rec = harness.recorder
    rec.write_frame(FRAME, at(14, 3, 20))
    harness.proc.finish(0)
    rec.poll_segments()
    (item,) = harness.publisher.items
    assert item.ended_ms == int(at(14, 4) * 1000)


def test_boxes_are_assigned_by_time_even_if_they_arrive_out_of_order(harness):
    rec = harness.recorder
    rec.write_frame(FRAME, at(14, 3, 50))
    rec.add_boxes(at(14, 4, 3), [box(3)])  # 다음 분 박스가 먼저 도착
    rec.add_boxes(at(14, 3, 55), [box(1)])
    harness.proc.finish(0)
    rec.poll_segments()
    (first,) = harness.publisher.items
    frames = read_boxes(first.boxes)["frames"]
    assert [f["b"][0][0] for f in frames] == [1]
    rec.add_boxes(at(14, 4, 10), [box(4)])
    rec.write_frame(FRAME, at(14, 4, 12))
    harness.proc.stdin.on_close = lambda: harness.proc.finish(1)
    rec.close()
    second = harness.publisher.items[1]
    frames = read_boxes(second.boxes)["frames"]
    assert [f["b"][0][0] for f in frames] == [3, 4]
    assert [f["t"] for f in frames] == [3.0, 10.0]


def test_boxes_that_arrive_after_their_file_was_published_are_dropped(harness):
    rec = harness.recorder
    rec.write_frame(FRAME, at(14, 3, 50))
    harness.proc.finish(0)
    rec.poll_segments()
    rec.add_boxes(at(14, 3, 58), [box(9)])  # 앞 파일이 이미 나간 뒤에 도착
    rec.write_frame(FRAME, at(14, 4, 5))
    rec.add_boxes(at(14, 4, 5), [box(2)])
    harness.proc.stdin.on_close = lambda: harness.proc.finish(1)
    rec.close()
    frames = read_boxes(harness.publisher.items[1].boxes)["frames"]
    assert [f["b"][0][0] for f in frames] == [2]
    assert all(f["t"] >= 0 for f in frames)


def test_a_rewritten_segment_list_does_not_publish_a_file_twice(harness):
    rec = harness.recorder
    rec.write_frame(FRAME, at(14, 3, 10))
    harness.proc.finish(0)
    rec.poll_segments()
    listing = Path(
        harness.proc.command[harness.proc.command.index("-segment_list") + 1]
    )
    listing.write_text("", encoding="utf-8")  # ffmpeg가 목록을 처음부터 다시 썼다
    harness.proc.finish(0)
    harness.proc.finish(1)
    rec.poll_segments()
    names = [item.video.name for item in harness.publisher.items]
    assert len(names) == 2 and names[0].endswith("_00000.mp4")
    assert names[1].endswith("_00001.mp4")


def test_a_quoted_name_in_the_segment_list_is_read(harness):
    rec = harness.recorder
    rec.write_frame(FRAME, at(14, 3, 10))
    path = harness.proc.segment(0)
    path.write_bytes(b"mp4")
    listing = Path(
        harness.proc.command[harness.proc.command.index("-segment_list") + 1]
    )
    listing.write_text(f'"{path.name}",0.000,1.000\n', encoding="utf-8")
    rec.poll_segments()
    assert harness.publisher.items[0].video.name == path.name


def test_segment_numbers_beyond_five_digits_are_still_recognized(harness):
    rec = harness.recorder
    rec.write_frame(FRAME, at(14, 3, 10))
    path = Path(str(harness.proc.segment(0)).replace("00000", "100000"))
    path.write_bytes(b"mp4")
    listing = Path(
        harness.proc.command[harness.proc.command.index("-segment_list") + 1]
    )
    listing.write_text(f"{path.name},0.000,1.000\n", encoding="utf-8")
    rec.poll_segments()
    assert len(harness.publisher.items) == 1


def test_list_and_error_log_are_removed_after_a_clean_exit(harness):
    rec = harness.recorder
    rec.write_frame(FRAME, at(14, 3, 10))
    rec.close()
    names = sorted(p.suffix for p in harness.tmp.iterdir())
    assert ".log" not in names and ".csv" not in names


def test_list_and_error_log_are_kept_after_a_failed_exit(tmp_path):
    harness = Harness(tmp_path, returncode=1)
    harness.recorder.write_frame(FRAME, at(14, 3, 10))
    harness.recorder.close()
    assert ".log" in {p.suffix for p in harness.tmp.iterdir()}


def test_ffmpeg_is_killed_if_terminate_is_not_enough(tmp_path, monkeypatch):
    monkeypatch.setattr(recorder_module, "EXIT_WAIT_SEC", 0.05)
    monkeypatch.setattr(recorder_module, "KILL_WAIT_SEC", 0.05)
    harness = Harness(tmp_path, hang=True)
    harness.recorder.write_frame(FRAME, at(14, 3))
    original = FakeProc.terminate
    FakeProc.terminate = lambda self: None  # 종료 요청을 무시하는 프로세스
    try:
        harness.recorder.close()
    finally:
        FakeProc.terminate = original
    assert harness.proc.killed
