"""캠 작업이 더미 분석 준비(FNC-S-009)를 부르는 흐름"""

import threading
import time
from pathlib import Path

import pytest

from white_eyes_flask.camera.camera_job import CameraStatus
from white_eyes_flask.camera.camera_service import CameraService
from white_eyes_flask.camera.preparer import NullPreparer, PrepareOutcome

URI = "dummy/videos/a.mp4"


class FakePreparer:
    def __init__(self, cached=False, outcome=PrepareOutcome.DONE, hold=False):
        self.cached = cached
        self.outcome = outcome
        self.hold = hold  # True면 정지될 때까지 준비 중
        self.calls = []
        self.entered = threading.Event()

    def has_cache(self, source_uri):
        return self.cached

    def prepare(self, camera_id, source_uri, stop):
        self.calls.append((camera_id, source_uri))
        self.entered.set()
        if self.hold:
            stop.wait()
            return PrepareOutcome.STOPPED
        return self.outcome


@pytest.fixture
def media_root(tmp_path: Path) -> Path:
    video = tmp_path / URI
    video.parent.mkdir(parents=True)
    video.write_bytes(b"")
    return tmp_path


def make_service(media_root, preparer) -> CameraService:
    return CameraService(str(media_root), preparer=preparer)


def wait_until(condition, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def job_of(service: CameraService, camera_id: int):
    return service._get_job(camera_id)


def test_cached_video_starts_as_running_without_preparing(media_root):
    preparer = FakePreparer(cached=True)
    service = make_service(media_root, preparer)
    try:
        status = service.start(7, "DUMMY", URI)
        assert status == CameraStatus.RUNNING
        time.sleep(0.1)
        assert preparer.calls == []  # 두 번째 가동은 준비 없이 바로 RUNNING
    finally:
        service.shutdown()


def test_video_without_cache_starts_as_preparing_and_calls_the_preparer(media_root):
    preparer = FakePreparer(hold=True)
    service = make_service(media_root, preparer)
    try:
        assert service.start(7, "DUMMY", URI) == CameraStatus.PREPARING
        assert preparer.entered.wait(timeout=3)
        assert preparer.calls == [(7, URI)]
    finally:
        service.shutdown()


def test_job_becomes_running_when_preparation_is_done(media_root):
    service = make_service(media_root, FakePreparer(outcome=PrepareOutcome.DONE))
    try:
        service.start(7, "DUMMY", URI)
        assert wait_until(lambda: job_of(service, 7).status == CameraStatus.RUNNING)
        assert job_of(service, 7).is_alive()  # 재생(FNC-S-010) 자리를 정지까지 지킴
        # 이미 도는 캠에 다시 시작이 오면 현재 상태(RUNNING)를 돌려준다
        assert service.start(7, "DUMMY", URI) == CameraStatus.RUNNING
    finally:
        service.shutdown()


def test_job_ends_with_error_status_when_preparation_fails(media_root):
    service = make_service(media_root, FakePreparer(outcome=PrepareOutcome.FAILED))
    try:
        service.start(7, "DUMMY", URI)
        assert wait_until(lambda: not job_of(service, 7).is_alive())
        assert job_of(service, 7).status == CameraStatus.ERROR
    finally:
        service.shutdown()


def test_start_again_after_error_prepares_again(media_root):
    preparer = FakePreparer(outcome=PrepareOutcome.FAILED)
    service = make_service(media_root, preparer)
    try:
        service.start(7, "DUMMY", URI)
        assert wait_until(lambda: not job_of(service, 7).is_alive())
        preparer.outcome = PrepareOutcome.DONE
        assert service.start(7, "DUMMY", URI) == CameraStatus.PREPARING
        assert wait_until(lambda: job_of(service, 7).status == CameraStatus.RUNNING)
        assert len(preparer.calls) == 2
    finally:
        service.shutdown()


def test_stop_cancels_a_preparation_in_progress(media_root):
    preparer = FakePreparer(hold=True)
    service = make_service(media_root, preparer)
    service.start(7, "DUMMY", URI)
    assert preparer.entered.wait(timeout=3)
    started = time.time()
    service.stop(7)
    assert time.time() - started < 2
    assert service.job_count() == 0
    assert not [t for t in threading.enumerate() if t.name == "cam-7"]


def test_cache_check_uses_the_preparer(media_root):
    preparer = FakePreparer(cached=False)
    service = make_service(media_root, preparer)
    try:
        assert service.start(7, "DUMMY", URI) == CameraStatus.PREPARING
        service.stop(7)
        preparer.cached = True
        assert service.start(7, "DUMMY", URI) == CameraStatus.RUNNING
    finally:
        service.shutdown()


def test_null_preparer_keeps_the_job_preparing_until_stopped(media_root):
    service = make_service(media_root, NullPreparer())
    try:
        assert service.start(7, "DUMMY", URI) == CameraStatus.PREPARING
        time.sleep(0.1)
        assert job_of(service, 7).is_alive()
        assert job_of(service, 7).status == CameraStatus.PREPARING
    finally:
        service.shutdown()
