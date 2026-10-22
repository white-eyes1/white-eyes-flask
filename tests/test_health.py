import time

import pytest

from white_eyes_flask import create_app
from white_eyes_flask.config import Config

TOKEN = "test-token"


@pytest.fixture
def client():
    app = create_app(
        Config(internal_token=TOKEN, yolo_weights=None, host="127.0.0.1", port=5000)
    )
    return app.test_client()


def test_health_returns_up(client):
    res = client.get("/analyzer/v1/health", headers={"X-Internal-Token": TOKEN})
    assert res.status_code == 200
    assert res.get_json() == {"status": "UP", "cameras": [], "bufferSize": 0}


def test_health_without_token_is_401(client):
    assert client.get("/analyzer/v1/health").status_code == 401


def test_health_with_wrong_token_is_401(client):
    res = client.get("/analyzer/v1/health", headers={"X-Internal-Token": "wrong"})
    assert res.status_code == 401


def test_empty_configured_token_rejects_everything():
    app = create_app(
        Config(internal_token="", yolo_weights=None, host="127.0.0.1", port=5000)
    )
    res = app.test_client().get("/analyzer/v1/health", headers={"X-Internal-Token": ""})
    assert res.status_code == 401


# ---------- 캠 작업 현황(FNC-S-013) ----------

from white_eyes_flask.camera.camera_job import CameraStatus  # noqa: E402
from white_eyes_flask.camera.kst import format_at  # noqa: E402
from white_eyes_flask.camera.preparer import NullPreparer  # noqa: E402

HEADERS = {"X-Internal-Token": TOKEN}


@pytest.fixture
def cam_app(tmp_path):
    video = tmp_path / "dummy" / "videos" / "a.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"")
    app = create_app(
        Config(
            internal_token=TOKEN,
            yolo_weights=None,
            host="127.0.0.1",
            port=5000,
            media_root=str(tmp_path),
        ),
        preparer=NullPreparer(),
    )
    yield app
    app.extensions["camera_service"].shutdown()


def cameras(app):
    res = app.test_client().get("/analyzer/v1/health", headers=HEADERS)
    assert res.status_code == 200
    return res.get_json()["cameras"]


def start(app, camera_id):
    app.extensions["camera_service"].start(camera_id, "DUMMY", "dummy/videos/a.mp4")


def test_no_cameras_gives_an_empty_list(cam_app):
    assert cameras(cam_app) == []


def test_started_camera_is_listed_without_a_result_yet(cam_app):
    start(cam_app, 7)
    assert cameras(cam_app) == [
        {"cameraId": 7, "state": "PREPARING", "lastResultAt": None}
    ]


def test_state_follows_the_job(cam_app):
    start(cam_app, 7)
    job = cam_app.extensions["camera_service"]._get_job(7)
    job.status = CameraStatus.ERROR
    assert cameras(cam_app)[0]["state"] == "ERROR"
    job.status = CameraStatus.RUNNING
    assert cameras(cam_app)[0]["state"] == "RUNNING"


def test_last_result_at_is_the_time_the_result_was_queued(cam_app):
    start(cam_app, 7)
    time.sleep(0.05)
    queue = cam_app.extensions["result_queue"]
    queue.put({"cameraId": 7, "at": "x"})
    queue.put({"cameraId": 8, "at": "x"})  # 다른 캠의 결과는 섞이지 않는다
    last = queue.last_put_at(7)
    assert cameras(cam_app)[0]["lastResultAt"] == format_at(last)
    assert cameras(cam_app)[0]["lastResultAt"].endswith("+09:00")


def test_stopped_camera_leaves_the_list(cam_app):
    start(cam_app, 7)
    cam_app.extensions["camera_service"].stop(7)
    assert cameras(cam_app) == []


def test_restarted_camera_does_not_show_an_old_result_time(cam_app):
    start(cam_app, 7)
    cam_app.extensions["result_queue"].put({"cameraId": 7, "at": "x"})
    cam_app.extensions["camera_service"].stop(7)
    time.sleep(0.05)  # Windows 시계 해상도보다 길게
    start(cam_app, 7)
    assert cameras(cam_app)[0]["lastResultAt"] is None


def test_cameras_are_sorted_by_id(cam_app):
    for camera_id in (9, 2, 5):
        start(cam_app, camera_id)
    assert [c["cameraId"] for c in cameras(cam_app)] == [2, 5, 9]


def test_camera_list_needs_the_token(cam_app):
    start(cam_app, 7)
    assert cam_app.test_client().get("/analyzer/v1/health").status_code == 401
