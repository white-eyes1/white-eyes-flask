"""FNC-S-008 4~6번: 공유 폴더로 옮기고 관리자 앱에 등록(재시도)"""

import logging
import time
from datetime import datetime
from pathlib import Path

import pytest

from white_eyes_flask.camera.kst import KST
from white_eyes_flask.camera.recording_registrar import RegisterResult
from white_eyes_flask.recording.publisher import (
    MAX_ATTEMPTS,
    FinishedRecording,
    RecordingPublisher,
)


def ms(hour, minute, second=0) -> int:
    moment = datetime(2026, 10, 12, hour, minute, second, tzinfo=KST)
    return int(moment.timestamp() * 1000)


class FakeRegistrar:
    def __init__(self, *results: RegisterResult) -> None:
        self.results = list(results)
        self.calls: list[tuple] = []

    def register(self, camera_id, started_ms, video_path, box_path, ended_ms=None):
        self.calls.append((camera_id, started_ms, video_path, box_path, ended_ms))
        return self.results.pop(0) if self.results else RegisterResult.OK


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def local(tmp_path: Path) -> Path:
    folder = tmp_path / "tmp"
    folder.mkdir()
    return folder


@pytest.fixture
def media(tmp_path: Path) -> Path:
    return tmp_path / "media"


def recording(local: Path, started=ms(14, 3, 27), ended=ms(14, 4)) -> FinishedRecording:
    video = local / "cam7_x_00000.mp4"
    boxes = local / "cam7_20261012_1403_00000.json.gz"
    video.write_bytes(b"video")
    boxes.write_bytes(b"boxes")
    return FinishedRecording(7, started, ended, video, boxes)


def publisher(media, registrar, clock, **kwargs) -> RecordingPublisher:
    return RecordingPublisher(str(media), registrar, clock=clock, **kwargs)


def test_moves_files_to_the_shared_folder_and_registers(media, local, clock):
    registrar = FakeRegistrar()
    pub = publisher(media, registrar, clock)
    item = recording(local)
    pub.submit(item)
    pub.run_due()
    folder = media / "recordings" / "live" / "cam7" / "20261012"
    assert (folder / "cam7_20261012_1403.mp4").read_bytes() == b"video"
    assert (folder / "cam7_20261012_1403.json.gz").read_bytes() == b"boxes"
    assert registrar.calls == [
        (
            7,
            item.started_ms,
            "recordings/live/cam7/20261012/cam7_20261012_1403.mp4",
            "recordings/live/cam7/20261012/cam7_20261012_1403.json.gz",
            item.ended_ms,
        )
    ]
    assert pub.pending == 0


def test_local_files_are_removed_after_moving(media, local, clock):
    pub = publisher(media, FakeRegistrar(), clock)
    pub.submit(recording(local))
    pub.run_due()
    assert list(local.iterdir()) == []


def test_no_part_files_are_left_in_the_shared_folder(media, local, clock):
    pub = publisher(media, FakeRegistrar(), clock)
    pub.submit(recording(local))
    pub.run_due()
    names = [p.name for p in (media / "recordings/live/cam7/20261012").iterdir()]
    assert sorted(names) == ["cam7_20261012_1403.json.gz", "cam7_20261012_1403.mp4"]


def test_registration_is_retried_every_minute_and_then_succeeds(media, local, clock):
    registrar = FakeRegistrar(RegisterResult.RETRY, RegisterResult.RETRY)
    pub = publisher(media, registrar, clock)
    pub.submit(recording(local))
    assert pub.run_due() == 60  # 첫 시도가 실패하면 1분 뒤
    assert len(registrar.calls) == 1
    clock.now += 30
    pub.run_due()
    assert len(registrar.calls) == 1  # 아직 때가 아님
    clock.now += 30
    pub.run_due()
    clock.now += 60
    assert pub.run_due() is None
    assert len(registrar.calls) == 3
    assert pub.pending == 0


def test_registration_retries_reuse_the_same_files(media, local, clock):
    registrar = FakeRegistrar(RegisterResult.RETRY)
    pub = publisher(media, registrar, clock)
    pub.submit(recording(local))
    pub.run_due()
    clock.now += 60
    pub.run_due()
    assert registrar.calls[0][2:4] == registrar.calls[1][2:4]


def test_gives_up_after_ten_failures_and_keeps_the_files(media, local, clock, caplog):
    registrar = FakeRegistrar(*[RegisterResult.RETRY] * 20)
    pub = publisher(media, registrar, clock)
    pub.submit(recording(local))
    with caplog.at_level(logging.ERROR):
        for _ in range(MAX_ATTEMPTS + 3):
            pub.run_due()
            clock.now += 60
    assert len(registrar.calls) == MAX_ATTEMPTS
    assert pub.pending == 0
    assert (media / "recordings/live/cam7/20261012/cam7_20261012_1403.mp4").exists()
    assert any("포기" in record.message for record in caplog.records)


def test_rejected_registration_is_not_retried(media, local, clock):
    registrar = FakeRegistrar(RegisterResult.REJECTED)
    pub = publisher(media, registrar, clock)
    pub.submit(recording(local))
    assert pub.run_due() is None
    assert len(registrar.calls) == 1


def test_move_failure_keeps_local_files_and_retries_next_minute(
    media, local, clock, tmp_path
):
    # 공유 폴더 자리에 파일이 있어 폴더를 만들 수 없다
    media.write_bytes(b"not a folder")
    registrar = FakeRegistrar()
    pub = publisher(media, registrar, clock)
    item = recording(local)
    pub.submit(item)
    assert pub.run_due() == 60
    assert item.video.exists() and item.boxes.exists()
    assert registrar.calls == []
    media.unlink()
    clock.now += 60
    assert pub.run_due() is None
    assert len(registrar.calls) == 1
    assert not item.video.exists()


def test_video_is_not_left_behind_when_the_box_file_cannot_be_copied(
    media, local, clock
):
    registrar = FakeRegistrar()
    pub = publisher(media, registrar, clock)
    item = recording(local)
    item.boxes.unlink()  # 좌표 파일이 없어 두 번째 복사가 실패한다
    pub.submit(item)
    pub.run_due()
    folder = media / "recordings/live/cam7/20261012"
    assert not (folder / "cam7_20261012_1403.mp4").exists()
    assert registrar.calls == []


def test_restart_in_the_same_minute_does_not_overwrite_the_old_files(
    media, local, clock
):
    registrar = FakeRegistrar()
    pub = publisher(media, registrar, clock)
    pub.submit(recording(local))
    pub.run_due()
    pub.submit(recording(local))
    pub.run_due()
    folder = media / "recordings/live/cam7/20261012"
    assert (folder / "cam7_20261012_1403.mp4").exists()
    assert (folder / "cam7_20261012_1403_2.mp4").exists()
    assert registrar.calls[1][2].endswith("cam7_20261012_1403_2.mp4")
    assert registrar.calls[1][3].endswith("cam7_20261012_1403_2.json.gz")


def test_worker_thread_publishes_without_blocking_the_caller(media, local):
    registrar = FakeRegistrar()
    pub = RecordingPublisher(str(media), registrar)
    pub.start()
    try:
        pub.submit(recording(local))
        for _ in range(100):
            if registrar.calls:
                break
            time.sleep(0.02)
        assert len(registrar.calls) == 1
    finally:
        pub.stop()


def test_publisher_can_be_started_again_after_stop(media, local):
    registrar = FakeRegistrar()
    pub = RecordingPublisher(str(media), registrar)
    pub.start()
    pub.stop()
    pub.start()
    try:
        pub.submit(recording(local))
        for _ in range(100):
            if registrar.calls:
                break
            time.sleep(0.02)
        assert len(registrar.calls) == 1
    finally:
        pub.stop()


def test_move_failures_are_retried_without_a_limit(media, local, clock):
    # 공유 폴더가 한참 안 돌아와도 로컬 파일을 포기하지 않고 계속 다시 옮긴다
    media.write_bytes(b"not a folder")
    registrar = FakeRegistrar()
    pub = publisher(media, registrar, clock)
    item = recording(local)
    pub.submit(item)
    for _ in range(MAX_ATTEMPTS + 10):
        pub.run_due()
        clock.now += 60
    assert pub.pending == 1
    assert item.video.exists() and item.boxes.exists()
    media.unlink()
    pub.run_due()
    assert len(registrar.calls) == 1


def test_one_failing_job_does_not_block_the_others(media, local, clock, caplog):
    class Exploding(FakeRegistrar):
        def register(self, camera_id, *args, **kwargs):
            if camera_id == 7:
                raise RuntimeError("boom")
            return super().register(camera_id, *args, **kwargs)

    registrar = Exploding()
    pub = publisher(media, registrar, clock)
    pub.submit(recording(local))  # cameraId=7: 등록에서 예외
    other_local = local.parent / "tmp2"
    other_local.mkdir()
    video = other_local / "v.mp4"
    boxes = other_local / "b.json.gz"
    video.write_bytes(b"v")
    boxes.write_bytes(b"b")
    pub.submit(FinishedRecording(8, ms(14, 3, 27), ms(14, 4), video, boxes))
    with caplog.at_level(logging.ERROR):
        pub.run_due()
    assert [call[0] for call in registrar.calls] == [8]
    assert any("게시 중 예외" in record.message for record in caplog.records)
