import threading

from white_eyes_flask.camera.camera_job import CameraJob, CameraStatus

SOURCE_URI = "dummy/videos/a.mp4"


def make_job() -> CameraJob:
    return CameraJob(7, SOURCE_URI, CameraStatus.PREPARING)


def test_reading_status_waits_for_the_status_lock():
    job = make_job()
    seen: list[CameraStatus] = []
    with job._status_lock:
        reader = threading.Thread(target=lambda: seen.append(job.status))
        reader.start()
        reader.join(0.1)
        assert reader.is_alive()
    reader.join(2)
    assert seen == [CameraStatus.PREPARING]


def test_writing_status_waits_for_the_status_lock():
    job = make_job()
    with job._status_lock:
        writer = threading.Thread(
            target=lambda: setattr(job, "status", CameraStatus.RUNNING)
        )
        writer.start()
        writer.join(0.1)
        assert writer.is_alive()
    writer.join(2)
    assert job.status == CameraStatus.RUNNING


def test_status_keeps_the_value_it_was_given():
    job = make_job()
    assert job.status == CameraStatus.PREPARING
    job.status = CameraStatus.ERROR
    assert job.status == CameraStatus.ERROR
