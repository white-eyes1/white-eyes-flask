"""FNC-S-008 정지 때 녹화 마무리: 작업이 어떻게 끝나든 CameraJob이 녹화기를 닫는다"""

import threading

from white_eyes_flask.camera.camera_job import CameraJob, CameraStatus
from white_eyes_flask.camera.player import PlayOutcome


class FakeRecorder:
    def __init__(self, fail: bool = False) -> None:
        self.closed = threading.Event()
        self.calls = 0
        self._fail = fail

    def close(self) -> None:
        self.calls += 1
        self.closed.set()
        if self._fail:
            raise RuntimeError("close failed")


class RaisingPlayer:
    def play(self, camera_id, source_uri, stop):
        raise RuntimeError("boom")


class FailingPlayer:
    def play(self, camera_id, source_uri, stop):
        return PlayOutcome.FAILED


def make_job(recorder, player=None) -> CameraJob:
    return CameraJob(
        7, "rtsp://cam", CameraStatus.RUNNING, player=player, recorder=recorder
    )


def test_stop_closes_the_recorder_before_the_thread_ends():
    recorder = FakeRecorder()
    job = make_job(recorder)
    job.start()
    assert not recorder.closed.is_set()  # 도는 동안은 열어 둔다
    assert job.stop(5) is True
    assert recorder.closed.is_set()
    assert recorder.calls == 1


def test_recorder_is_closed_when_the_player_raises():
    recorder = FakeRecorder()
    job = make_job(recorder, RaisingPlayer())
    job.start()
    job.stop(5)
    assert recorder.closed.is_set()
    assert job.status == CameraStatus.ERROR


def test_recorder_is_closed_when_the_player_reports_failure():
    recorder = FakeRecorder()
    job = make_job(recorder, FailingPlayer())
    job.start()
    job.stop(5)
    assert recorder.closed.is_set()


def test_a_recorder_that_fails_to_close_does_not_keep_the_thread_alive():
    job = make_job(FakeRecorder(fail=True))
    job.start()
    assert job.stop(5) is True


def test_job_without_a_recorder_still_stops():
    job = make_job(None)
    job.start()
    assert job.stop(5) is True
