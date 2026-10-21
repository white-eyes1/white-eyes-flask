"""FNC-S-002 캠 정지 (POST /analyzer/v1/cameras/{id}/stop)"""

import logging
import threading
import time
from pathlib import Path

import pytest

from white_eyes_flask import create_app
from white_eyes_flask.camera import camera_job, camera_service
from white_eyes_flask.camera.preparer import NullPreparer
from white_eyes_flask.config import Config

TOKEN = "test-token"
HEADERS = {"X-Internal-Token": TOKEN}
START_BODY = {"sourceType": "DUMMY", "sourceUri": "dummy/videos/a.mp4"}


def start_url(camera_id: int = 7) -> str:
    return f"/analyzer/v1/cameras/{camera_id}/start"


def stop_url(camera_id: int = 7) -> str:
    return f"/analyzer/v1/cameras/{camera_id}/stop"


def cam_threads(camera_id: int = 7) -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == f"cam-{camera_id}"]


@pytest.fixture
def app(tmp_path: Path):
    video = tmp_path / "dummy" / "videos" / "a.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"")
    config = Config(
        internal_token=TOKEN,
        yolo_weights=None,
        host="127.0.0.1",
        port=5000,
        media_root=str(tmp_path),
    )
    app = create_app(config, preparer=NullPreparer())
    yield app
    app.extensions["camera_service"].shutdown()


@pytest.fixture
def client(app):
    return app.test_client()


def test_stop_running_camera_returns_200_and_removes_job(client, app):
    client.post(start_url(), json=START_BODY, headers=HEADERS)
    res = client.post(stop_url(), headers=HEADERS)
    assert res.status_code == 200
    assert app.extensions["camera_service"].job_count() == 0


def test_stop_waits_until_job_thread_ends(client):
    client.post(start_url(), json=START_BODY, headers=HEADERS)
    assert len(cam_threads()) == 1
    client.post(stop_url(), headers=HEADERS)
    assert cam_threads() == []


def test_stop_without_job_returns_200(client):
    assert client.post(stop_url(), headers=HEADERS).status_code == 200


def test_stop_twice_returns_200_both_times(client):
    client.post(start_url(), json=START_BODY, headers=HEADERS)
    first = client.post(stop_url(), headers=HEADERS)
    second = client.post(stop_url(), headers=HEADERS)
    assert (first.status_code, second.status_code) == (200, 200)


def test_start_again_after_stop_creates_new_job(client, app):
    client.post(start_url(), json=START_BODY, headers=HEADERS)
    client.post(stop_url(), headers=HEADERS)
    res = client.post(start_url(), json=START_BODY, headers=HEADERS)
    assert res.status_code == 200
    assert app.extensions["camera_service"].job_count() == 1


def test_stop_affects_only_target_camera(client, app):
    client.post(start_url(7), json=START_BODY, headers=HEADERS)
    client.post(start_url(8), json=START_BODY, headers=HEADERS)
    client.post(stop_url(7), headers=HEADERS)
    assert app.extensions["camera_service"].job_count() == 1
    assert len(cam_threads(8)) == 1


def test_stop_without_token_is_401(client, app):
    client.post(start_url(), json=START_BODY, headers=HEADERS)
    assert client.post(stop_url()).status_code == 401
    assert app.extensions["camera_service"].job_count() == 1


def test_start_during_slow_stop_never_overlaps_threads(client, app, monkeypatch):
    def slow_to_end(self) -> None:
        self._stop_flag.wait()
        time.sleep(0.3)

    monkeypatch.setattr(camera_job.CameraJob, "_run", slow_to_end)
    client.post(start_url(), json=START_BODY, headers=HEADERS)

    stopper = threading.Thread(
        target=lambda: app.test_client().post(stop_url(), headers=HEADERS)
    )
    stopper.start()
    time.sleep(0.05)
    client.post(start_url(), json=START_BODY, headers=HEADERS)

    assert len(cam_threads()) == 1
    stopper.join()


def test_slow_job_still_returns_200_and_logs_warning(client, app, monkeypatch, caplog):
    release = threading.Event()

    def ignore_stop_flag(self) -> None:
        release.wait()

    monkeypatch.setattr(camera_job.CameraJob, "_run", ignore_stop_flag)
    monkeypatch.setattr(camera_service, "STOP_WAIT_SEC", 0.05)
    client.post(start_url(), json=START_BODY, headers=HEADERS)

    with caplog.at_level(logging.WARNING):
        res = client.post(stop_url(), headers=HEADERS)

    release.set()
    assert res.status_code == 200
    assert any("cameraId=7" in r.getMessage() for r in caplog.records)
