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
from white_eyes_flask.camera.recording_registrar import RegisterResult
from white_eyes_flask.recording import recorder as recorder_module
from white_eyes_flask.recording.ffmpeg import RecorderSettings
from white_eyes_flask.recording.publisher import RecordingPublisher
from white_eyes_flask.recording.recorder import (
    LiveRecorder,
    RecorderFailed,
    RecorderUnavailable,
)

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
        self.exited = False
        self.terminated = False
        self.killed = False
        self.hang = False

    @property
    def tmp_dir(self) -> Path:
        return Path(self.command[-1]).parent

    def segment(self, index: int) -> Path:
        pattern = self.command[-1]
        return Path(pattern.replace("%05d", f"{index:05d}"))

    def finish(
        self,
        index: int,
        start: float | None = None,
        end: float | None = None,
        partial_line: bool = False,
    ) -> None:
        """index번째 파일을 닫고 list에 올린다. start·end는 첫 프레임부터의 초(실제 ffmpeg처럼).

        시각을 주지 않으면 파일 이름만 적는다(시각을 못 읽을 때의 분 경계 가정 경로를 시험한다).
        """
        path = self.segment(index)
        path.write_bytes(b"mp4-%d" % index)
        listing = Path(self.command[self.command.index("-segment_list") + 1])
        line = path.name if start is None else f"{path.name},{start:.6f},{end:.6f}"
        with open(listing, "a", encoding="utf-8") as stream:
            stream.write(line if partial_line else line + "\n")

    def wait(self, timeout=None):
        if self.hang and not self.terminated:
            raise subprocess.TimeoutExpired("ffmpeg", timeout)
        self.exited = True
        return self._returncode

    def die(self, code: int = 137) -> None:
        """녹화 중에 ffmpeg가 죽은 것을 흉내."""
        self._returncode = code
        self.exited = True

    def poll(self):
        return self._returncode if self.exited else None

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
    def __init__(self, tmp_path: Path, grace: float = 0.0, **proc_kwargs) -> None:
        self.now = 1000.0  # 가짜 시계(초)
        self.tmp = tmp_path / "tmp"
        self.publisher = FakePublisher()
        self.proc: FakeProc | None = None
        self._proc_kwargs = proc_kwargs
        self.recorder = LiveRecorder(
            7,
            RecorderSettings(tmp_dir=str(self.tmp), analyze_fps=5),
            self.publisher,
            popen=self._popen,
            box_grace_sec=grace,
            clock=lambda: self.now,
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
    assert item.ended_ms == int(at(14, 3, 50) * 1000) + 100  # 마지막 프레임 길이만큼
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
    assert second.ended_ms == int(at(14, 4, 20) * 1000) + 100
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
    harness.proc.finish(0, 0.0, 1.0, partial_line=True)
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


def test_a_closed_file_waits_until_analysis_has_passed_the_minute(tmp_path):
    harness = Harness(tmp_path, grace=10.0)
    rec = harness.recorder
    rec.write_frame(FRAME, at(14, 3, 50))
    rec.add_boxes(at(14, 3, 50), [box(1)])
    harness.proc.finish(0)
    rec.poll_segments()
    assert harness.publisher.items == []  # 분석이 아직 14:04를 넘지 않았다
    rec.add_boxes(at(14, 3, 59.5), [box(2)])  # 분 끝 박스가 늦게 도착
    rec.poll_segments()
    assert harness.publisher.items == []
    rec.add_boxes(at(14, 4, 0.2), [box(3)])  # 분석이 경계를 넘었다
    rec.poll_segments()
    (first,) = harness.publisher.items
    ids = [f["b"][0][0] for f in read_boxes(first.boxes)["frames"]]
    assert ids == [1, 2]  # 늦게 도착한 분 끝 박스도 들어갔다
    rec.close()


def test_a_closed_file_is_published_after_the_grace_time_even_without_boxes(tmp_path):
    harness = Harness(tmp_path, grace=10.0)
    rec = harness.recorder
    rec.write_frame(FRAME, at(14, 3, 50))
    harness.proc.finish(0)
    rec.poll_segments()
    assert harness.publisher.items == []
    harness.now += 11
    rec.poll_segments()
    assert len(harness.publisher.items) == 1
    rec.close()


def test_files_are_published_in_order(tmp_path):
    harness = Harness(tmp_path, grace=10.0)
    rec = harness.recorder
    rec.write_frame(FRAME, at(14, 3, 50))
    harness.proc.finish(0)
    harness.proc.finish(1)
    rec.add_boxes(at(14, 5, 1), [box(1)])  # 둘째 파일 쪽은 이미 조건이 충족됐다
    rec.poll_segments()
    names = [item.video.name for item in harness.publisher.items]
    assert names == [harness.proc.segment(0).name, harness.proc.segment(1).name]
    rec.close()


def test_closing_publishes_files_that_are_still_waiting(tmp_path):
    harness = Harness(tmp_path, grace=10.0)
    rec = harness.recorder
    rec.write_frame(FRAME, at(14, 3, 50))
    harness.proc.stdin.on_close = lambda: harness.proc.finish(0)
    rec.close()
    assert len(harness.publisher.items) == 1


def test_the_last_file_is_never_zero_length(harness):
    rec = harness.recorder
    rec.write_frame(FRAME, at(14, 3, 20))  # 프레임이 한 장뿐
    harness.proc.stdin.on_close = lambda: harness.proc.finish(0)
    rec.close()
    (item,) = harness.publisher.items
    assert item.ended_ms > item.started_ms


def test_write_frame_raises_after_ffmpeg_dies_and_boxes_are_not_kept(harness, caplog):
    rec = harness.recorder
    rec.write_frame(FRAME, at(14, 3, 10))
    harness.proc.die()
    with caplog.at_level(logging.ERROR):
        rec._check_alive()
    assert any("녹화 중에 죽음" in record.message for record in caplog.records)
    with pytest.raises(RecorderFailed):
        rec.write_frame(FRAME, at(14, 3, 11))
    rec.add_boxes(at(14, 3, 12), [box(1)])
    assert rec._boxes == []
    rec.close()


def test_a_frame_of_another_size_is_resized_to_the_first_one(harness, caplog):
    rec = harness.recorder
    rec.write_frame(np.zeros((8, 12, 3), dtype=np.uint8), at(14, 3, 1))
    with caplog.at_level(logging.WARNING):
        rec.write_frame(np.zeros((16, 24, 3), dtype=np.uint8), at(14, 3, 2))
    rec.close()
    sizes = {len(chunk) for chunk in harness.proc.stdin.chunks}
    assert sizes == {8 * 12 * 3}  # 두 번째도 처음 크기로 들어갔다
    assert any("크기가 바뀌어" in record.message for record in caplog.records)


def test_old_boxes_are_dropped_when_no_file_ever_comes_out(
    harness, monkeypatch, caplog
):
    monkeypatch.setattr(recorder_module, "BOX_BACKLOG_LIMIT", 20)
    rec = harness.recorder
    rec.write_frame(FRAME, at(14, 3, 0))
    with caplog.at_level(logging.WARNING):
        for index in range(60):
            rec.add_boxes(at(14, 3, index * 0.1), [box(1)])
    assert len(rec._boxes) <= 20
    assert any("오래된 박스를 버림" in record.message for record in caplog.records)
    rec.close()


def test_close_is_fast_even_if_the_pipe_is_stuck(tmp_path):
    gate = threading.Event()  # 열리지 않는다: ffmpeg가 입력을 안 읽는 상태
    harness = Harness(tmp_path, gate=gate, hang=True)
    rec = harness.recorder
    rec.write_frame(FRAME, at(14, 3, 1))
    time.sleep(0.2)  # 쓰기 스레드가 write에서 막히게 한다
    started = time.monotonic()
    rec.close()
    assert time.monotonic() - started < 8.0
    assert harness.proc.terminated  # 끝나길 기다리지 않고 바로 종료시켰다
    gate.set()


def test_files_that_never_closed_are_not_published_and_are_reported(harness, caplog):
    rec = harness.recorder
    rec.write_frame(FRAME, at(14, 3, 10))
    harness.proc.segment(0).write_bytes(b"half")  # 목록에는 오르지 않은 파일
    with caplog.at_level(logging.WARNING):
        rec.close()
    assert harness.publisher.items == []
    assert any("닫히지 못한 녹화 파일" in record.message for record in caplog.records)


def test_file_times_come_from_the_segment_list_not_from_minute_boundaries(harness):
    # 실제 ffmpeg 관측: 분 경계 0.15초 전에 시작하면 경계에서 안 끊기고 파일이 하나로 이어진다
    rec = harness.recorder
    rec.write_frame(FRAME, at(14, 3, 59.85))
    rec.add_boxes(at(14, 3, 59.9), [box(1)])
    rec.write_frame(FRAME, at(14, 4, 1.0))
    rec.add_boxes(at(14, 4, 1.0), [box(2)])  # 경계 뒤의 박스도 이 파일 몫이다
    rec.write_frame(FRAME, at(14, 4, 3.8))
    rec.add_boxes(at(14, 4, 3.8), [box(3)])
    harness.proc.stdin.on_close = lambda: harness.proc.finish(0, 0.0, 4.2)
    rec.close()
    (item,) = harness.publisher.items
    assert item.started_ms == int(at(14, 3, 59.85) * 1000)
    assert item.ended_ms == int(at(14, 4, 3.8) * 1000) + 100
    frames = read_boxes(item.boxes)["frames"]
    assert [f["b"][0][0] for f in frames] == [1, 2, 3]
    assert [f["t"] for f in frames] == [0.05, 1.15, 3.95]


def test_a_file_cut_before_the_boundary_is_split_where_ffmpeg_cut_it(harness):
    # 실제 ffmpeg 관측: 첫 파일이 분 경계보다 0.8초 일찍 끊겼다
    rec = harness.recorder
    rec.write_frame(FRAME, at(14, 3, 57))
    rec.add_boxes(at(14, 3, 58), [box(1)])  # 첫 파일 구간(57.0~59.2)
    rec.add_boxes(at(14, 3, 59.5), [box(2)])  # 둘째 파일 구간(59.2~)
    rec.write_frame(FRAME, at(14, 4, 5))
    rec.add_boxes(at(14, 4, 5), [box(3)])
    harness.proc.finish(0, 0.0, 2.2)
    rec.poll_segments()
    harness.proc.stdin.on_close = lambda: harness.proc.finish(1, 2.2, 8.3)
    rec.close()
    first, second = harness.publisher.items
    assert first.ended_ms == second.started_ms == int(at(14, 3, 59.2) * 1000)
    assert [f["b"][0][0] for f in read_boxes(first.boxes)["frames"]] == [1]
    frames = read_boxes(second.boxes)["frames"]
    assert [f["b"][0][0] for f in frames] == [2, 3]
    assert frames[0]["t"] == 0.3
    # 이름은 ffmpeg가 끊은 시각에서 가장 가까운 분이다(둘째 파일이 14:04)
    assert "cam7_20261012_1403_" in first.boxes.name
    assert "cam7_20261012_1404_" in second.boxes.name


def test_unreadable_times_in_the_segment_list_fall_back_to_minute_boundaries(harness):
    rec = harness.recorder
    rec.write_frame(FRAME, at(14, 3, 50))
    rec.add_boxes(at(14, 3, 55), [box(1)])
    rec.write_frame(FRAME, at(14, 4, 5))
    harness.proc.finish(0, 5.0, 1.0)  # 끝이 시작보다 앞서 믿을 수 없다
    rec.poll_segments()
    (first,) = harness.publisher.items
    assert first.ended_ms == int(at(14, 4) * 1000)
    assert len(read_boxes(first.boxes)["frames"]) == 1
    rec.close()


class _Registrar:
    def __init__(self) -> None:
        self.calls = []

    def register(self, camera_id, started_ms, video_path, box_path, ended_ms=None):
        self.calls.append((video_path, box_path))
        return RegisterResult.OK


def _with_real_publisher(harness, tmp_path):
    registrar = _Registrar()
    publisher = RecordingPublisher(str(tmp_path / "media"), registrar)
    harness.recorder._publisher = publisher
    return publisher, registrar


def test_final_shared_paths_after_a_cut_before_the_boundary(tmp_path):
    # 첫 파일이 분 경계보다 0.8초 일찍 끊긴 경우: 둘째 파일은 14:04 이름이어야 하고 _2가 생기지 않는다
    harness = Harness(tmp_path)
    publisher, registrar = _with_real_publisher(harness, tmp_path)
    rec = harness.recorder
    rec.write_frame(FRAME, at(14, 3, 57))
    rec.write_frame(FRAME, at(14, 4, 5))
    harness.proc.finish(0, 0.0, 2.2)
    rec.poll_segments()
    harness.proc.stdin.on_close = lambda: harness.proc.finish(1, 2.2, 8.3)
    rec.close()
    publisher.run_due()
    folder = tmp_path / "media/recordings/live/cam7/20261012"
    names = sorted(p.name for p in folder.iterdir())
    assert names == [
        "cam7_20261012_1403.json.gz",
        "cam7_20261012_1403.mp4",
        "cam7_20261012_1404.json.gz",
        "cam7_20261012_1404.mp4",
    ]
    assert [video for video, _ in registrar.calls] == [
        "recordings/live/cam7/20261012/cam7_20261012_1403.mp4",
        "recordings/live/cam7/20261012/cam7_20261012_1404.mp4",
    ]


def test_final_shared_paths_just_before_midnight(tmp_path):
    # 23:59:58에 시작해 23:59:59.2에 끊기면 둘째 파일은 다음 날 00:00 이름과 폴더다
    harness = Harness(tmp_path)
    publisher, registrar = _with_real_publisher(harness, tmp_path)
    rec = harness.recorder
    rec.write_frame(FRAME, at(23, 59, 58))
    rec.write_frame(FRAME, at(23, 59, 59.5))
    harness.proc.finish(0, 0.0, 1.2)
    rec.poll_segments()
    harness.proc.stdin.on_close = lambda: harness.proc.finish(1, 1.2, 3.0)
    rec.close()
    publisher.run_due()
    videos = [video for video, _ in registrar.calls]
    assert videos == [
        "recordings/live/cam7/20261012/cam7_20261012_2359.mp4",
        "recordings/live/cam7/20261013/cam7_20261013_0000.mp4",
    ]


def test_a_failed_box_file_write_is_retried_and_the_video_is_not_lost(
    harness, monkeypatch
):
    real = recorder_module.write_box_file
    calls = []

    def flaky(path, fps, frames):
        calls.append(len(frames))
        if len(calls) == 1:
            raise OSError("disk full")
        return real(path, fps, frames)

    monkeypatch.setattr(recorder_module, "write_box_file", flaky)
    rec = harness.recorder
    rec.write_frame(FRAME, at(14, 3, 50))
    rec.add_boxes(at(14, 3, 55), [box(1)])
    harness.proc.finish(0, 0.0, 10.0)
    rec.poll_segments()
    assert harness.publisher.items == []  # 첫 쓰기 실패
    rec.poll_segments()  # 다음 확인 때 다시 시도한다
    (item,) = harness.publisher.items
    assert calls == [1, 1]  # 박스가 그대로 남아 있어 두 번째에도 같은 박스를 쓴다
    assert [f["b"][0][0] for f in read_boxes(item.boxes)["frames"]] == [1]
    rec.close()


def test_gives_up_after_repeated_box_file_failures_and_says_why(
    harness, monkeypatch, caplog
):
    def always_fail(path, fps, frames):
        raise OSError("disk full")

    monkeypatch.setattr(recorder_module, "write_box_file", always_fail)
    rec = harness.recorder
    rec.write_frame(FRAME, at(14, 3, 50))
    harness.proc.finish(0, 0.0, 10.0)
    with caplog.at_level(logging.WARNING):
        for _ in range(recorder_module.MAX_BOX_WRITE_ATTEMPTS + 2):
            rec.poll_segments()
        rec.close()
    assert harness.publisher.items == []
    messages = [record.message for record in caplog.records]
    assert any("번 쓰지 못해 게시하지 못함" in m for m in messages)
    # 닫히지 못한 파일과 구분된 문구
    assert any("좌표 파일을 쓰지 못해 게시하지 못한 녹화 파일" in m for m in messages)
    assert not any("닫히지 못한 녹화 파일" in m for m in messages)
