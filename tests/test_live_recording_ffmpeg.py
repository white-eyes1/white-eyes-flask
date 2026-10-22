"""FNC-S-008 실제 ffmpeg로 녹화(ffmpeg가 있는 PC·서버에서만 돈다)"""

import shutil
import time
from pathlib import Path

import numpy as np
import pytest

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
