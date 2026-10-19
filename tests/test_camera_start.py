"""FNC-S-001 캠 시작 (POST /analyzer/v1/cameras/{id}/start)"""

import threading
from pathlib import Path

import pytest

from white_eyes_flask import create_app
from white_eyes_flask.config import Config

TOKEN = "test-token"
HEADERS = {"X-Internal-Token": TOKEN}
START_URL = "/analyzer/v1/cameras/7/start"
DUMMY_BODY = {"sourceType": "DUMMY", "sourceUri": "dummy/videos/a.mp4"}


@pytest.fixture
def media_root(tmp_path: Path) -> Path:
    video = tmp_path / "dummy" / "videos" / "a.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"")
    return tmp_path


@pytest.fixture
def app(media_root: Path):
    config = Config(
        internal_token=TOKEN,
        yolo_weights=None,
        host="127.0.0.1",
        port=5000,
        media_root=str(media_root),
    )
    app = create_app(config)
    yield app
    app.extensions["camera_service"].shutdown()


@pytest.fixture
def client(app):
    return app.test_client()


def test_dummy_start_without_cache_returns_preparing(client):
    res = client.post(START_URL, json=DUMMY_BODY, headers=HEADERS)
    assert res.status_code == 200
    assert res.get_json() == {"cameraId": 7, "status": "PREPARING"}


def test_start_twice_keeps_single_job(client, app):
    first = client.post(START_URL, json=DUMMY_BODY, headers=HEADERS)
    second = client.post(START_URL, json=DUMMY_BODY, headers=HEADERS)
    assert second.status_code == 200
    assert second.get_json() == first.get_json()
    assert app.extensions["camera_service"].job_count() == 1


def test_concurrent_starts_keep_single_job(app):
    results: list[int] = []

    def start() -> None:
        res = app.test_client().post(START_URL, json=DUMMY_BODY, headers=HEADERS)
        results.append(res.status_code)

    threads = [threading.Thread(target=start) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == [200] * 8
    assert app.extensions["camera_service"].job_count() == 1


def test_different_cameras_get_separate_jobs(client, app):
    client.post("/analyzer/v1/cameras/7/start", json=DUMMY_BODY, headers=HEADERS)
    client.post("/analyzer/v1/cameras/8/start", json=DUMMY_BODY, headers=HEADERS)
    assert app.extensions["camera_service"].job_count() == 2


def test_dummy_file_missing_returns_400_source_not_found(client):
    body = {"sourceType": "DUMMY", "sourceUri": "dummy/videos/none.mp4"}
    res = client.post(START_URL, json=body, headers=HEADERS)
    assert res.status_code == 400
    assert res.get_json()["code"] == "SOURCE_NOT_FOUND"


@pytest.mark.parametrize(
    "source_uri",
    [
        "dummy/videos/../../secret.mp4",
        "/etc/passwd",
        "other/videos/a.mp4",
        "dummy/videos/a.avi",
    ],
)
def test_dummy_invalid_path_returns_400_source_not_found(client, source_uri):
    body = {"sourceType": "DUMMY", "sourceUri": source_uri}
    res = client.post(START_URL, json=body, headers=HEADERS)
    assert res.status_code == 400
    assert res.get_json()["code"] == "SOURCE_NOT_FOUND"


def test_failed_start_does_not_create_job(client, app):
    body = {"sourceType": "DUMMY", "sourceUri": "dummy/videos/none.mp4"}
    client.post(START_URL, json=body, headers=HEADERS)
    assert app.extensions["camera_service"].job_count() == 0


@pytest.mark.parametrize(
    "body",
    [
        None,
        {},
        {"sourceType": "FILE", "sourceUri": "dummy/videos/a.mp4"},
        {"sourceType": "DUMMY"},
        {"sourceType": "DUMMY", "sourceUri": 123},
    ],
)
def test_invalid_request_body_returns_400(client, body):
    res = client.post(START_URL, json=body, headers=HEADERS)
    assert res.status_code == 400
    assert res.get_json()["code"] == "INVALID_REQUEST"


@pytest.mark.parametrize("source_type", [[], {}, 1, None])
def test_non_string_source_type_returns_400(client, source_type):
    body = {"sourceType": source_type, "sourceUri": "dummy/videos/a.mp4"}
    res = client.post(START_URL, json=body, headers=HEADERS)
    assert res.status_code == 400
    assert res.get_json()["code"] == "INVALID_REQUEST"


def test_invalid_request_has_field_errors(client):
    body = {"sourceType": "FILE", "sourceUri": 123}
    res = client.post(START_URL, json=body, headers=HEADERS)
    fields = {item["field"] for item in res.get_json()["fieldErrors"]}
    assert fields == {"sourceType", "sourceUri"}


def test_source_not_found_has_no_field_errors(client):
    body = {"sourceType": "DUMMY", "sourceUri": "dummy/videos/none.mp4"}
    res = client.post(START_URL, json=body, headers=HEADERS)
    assert "fieldErrors" not in res.get_json()


def test_null_byte_in_path_returns_400_source_not_found(client):
    body = {"sourceType": "DUMMY", "sourceUri": "dummy/videos/a\u0000.mp4"}
    res = client.post(START_URL, json=body, headers=HEADERS)
    assert res.status_code == 400
    assert res.get_json()["code"] == "SOURCE_NOT_FOUND"


def test_symlink_outside_media_root_returns_400_source_not_found(
    client, media_root: Path, tmp_path_factory
):
    outside = tmp_path_factory.mktemp("outside") / "secret.mp4"
    outside.write_bytes(b"")
    link = media_root / "dummy" / "videos" / "link.mp4"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("심볼릭 링크를 만들 권한이 없음")
    body = {"sourceType": "DUMMY", "sourceUri": "dummy/videos/link.mp4"}
    res = client.post(START_URL, json=body, headers=HEADERS)
    assert res.status_code == 400
    assert res.get_json()["code"] == "SOURCE_NOT_FOUND"


def test_running_camera_returns_current_status_even_for_live(client, app):
    client.post(START_URL, json=DUMMY_BODY, headers=HEADERS)
    body = {"sourceType": "LIVE", "sourceUri": "rtsp://example/stream"}
    res = client.post(START_URL, json=body, headers=HEADERS)
    assert res.status_code == 200
    assert res.get_json() == {"cameraId": 7, "status": "PREPARING"}
    assert app.extensions["camera_service"].job_count() == 1


def test_running_camera_ignores_source_changes(client, app, media_root: Path):
    client.post(START_URL, json=DUMMY_BODY, headers=HEADERS)
    (media_root / "dummy" / "videos" / "a.mp4").unlink()
    res = client.post(START_URL, json=DUMMY_BODY, headers=HEADERS)
    assert res.status_code == 200
    assert app.extensions["camera_service"].job_count() == 1


def test_live_is_not_implemented_in_stage1(client, app):
    body = {"sourceType": "LIVE", "sourceUri": "rtsp://example/stream"}
    res = client.post(START_URL, json=body, headers=HEADERS)
    assert res.status_code == 501
    assert app.extensions["camera_service"].job_count() == 0


def test_start_without_token_is_401(client):
    assert client.post(START_URL, json=DUMMY_BODY).status_code == 401
