"""FNC-S-008 실제 ffmpeg로 녹화(ffmpeg가 있는 PC·서버에서만 돈다)"""

import gzip
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from white_eyes_flask.analysis.result import TrackedBox
from white_eyes_flask.recording.ffmpeg import RecorderSettings
from white_eyes_flask.recording.recorder import LiveRecorder

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg가 설치돼 있지 않음"
)


class FakePublisher:
    def __init__(self) -> None:
        self.items = []

    def submit(self, recording) -> None:
        self.items.append(recording)


def test_records_a_playable_faststart_mp4_and_a_box_file(tmp_path: Path):
    publisher = FakePublisher()
    recorder = LiveRecorder(
        7,
        RecorderSettings(width=160, height=90, fps=10, tmp_dir=str(tmp_path)),
        publisher,
    )
    now = time.time()
    for index in range(25):
        frame = np.full((90, 160, 3), index * 8 % 255, dtype=np.uint8)
        recorder.write_frame(frame, now + index * 0.1)
        time.sleep(0.1)
    recorder.close()
    assert len(publisher.items) >= 1
    data = publisher.items[0].video.read_bytes()
    assert data[4:8] == b"ftyp"
    # faststart: 구간 이동에 필요한 moov가 영상 데이터(mdat)보다 앞에 있다
    assert data.find(b"moov") != -1
    assert data.find(b"moov") < data.find(b"mdat")


def video_seconds(path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    out = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    return float(out.stdout.strip()) if out.stdout.strip() else None


def test_no_box_is_lost_and_file_times_match_the_video(tmp_path: Path):
    """분 경계 근처에서 시작해도(ffmpeg는 경계에서 끊지 않을 수 있다) 박스가 사라지지 않는다."""
    publisher = FakePublisher()
    recorder = LiveRecorder(
        7,
        RecorderSettings(width=160, height=90, fps=10, tmp_dir=str(tmp_path)),
        publisher,
        box_grace_sec=0.5,
    )
    # 다음 분 경계의 0.3초 전에 시작해 경계를 넘기며 녹화한다(최대 약 1분 기다림)
    now = time.time()
    time.sleep(max(0.0, (int(now // 60) + 1) * 60 - 0.3 - now))
    added = 0
    for index in range(40):
        at = time.time()
        recorder.write_frame(np.full((90, 160, 3), index * 5 % 255, dtype=np.uint8), at)
        recorder.add_boxes(at, [TrackedBox(1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.9)])
        added += 1
        time.sleep(0.1)
    recorder.close()
    assert publisher.items, "파일이 하나도 안 나왔다"
    in_files = 0
    for item in publisher.items:
        with gzip.open(item.boxes, "rt", encoding="utf-8") as stream:
            frames = json.load(stream)["frames"]
        in_files += len(frames)
        assert item.ended_ms > item.started_ms
        assert all(frame["t"] >= 0 for frame in frames)
        seconds = video_seconds(item.video)
        if seconds is not None:
            # 영상 길이와 기록한 구간이 1초 안에서 맞는다
            assert abs(seconds - (item.ended_ms - item.started_ms) / 1000) < 1.0
    assert in_files == added  # 박스가 사라지지 않았다
