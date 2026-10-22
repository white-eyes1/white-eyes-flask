"""FNC-S-011 실시간 화면(MJPEG)·카드 미리보기(스냅샷)"""

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from white_eyes_flask import create_app
from white_eyes_flask.camera.preparer import NullPreparer
from white_eyes_flask.config import Config
from white_eyes_flask.stream.stream_service import (
    SNAPSHOT_REUSE_SEC,
    SNAPSHOT_WANTED_SEC,
    CameraStream,
    LiveFrame,
)

TOKEN = "test-token"
HEADERS = {"X-Internal-Token": TOKEN}
START_BODY = {"sourceType": "DUMMY", "sourceUri": "dummy/videos/a.mp4"}


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def make_frame(person_count: int = 2, fill: int = 0, width: int = 1280) -> LiveFrame:
    image = np.full((720 * width // 1280, width, 3), fill, dtype=np.uint8)
    boxes = np.array([[100, 100, 200, 300, 0.9], [400, 200, 460, 330, 0.8]])
    return LiveFrame(image, boxes, person_count)


def decode(jpeg: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)


def wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


# ---------- 스냅샷 ----------


def test_snapshot_is_none_before_first_frame():
    assert CameraStream().snapshot_jpeg() is None


def test_snapshot_is_320px_wide_jpeg_with_green_boxes():
    stream = CameraStream()
    stream.publish(make_frame())
    image = decode(stream.snapshot_jpeg())
    assert image.shape[1] == 320
    green = (image[:, :, 1] > 150) & (image[:, :, 0] < 100) & (image[:, :, 2] < 100)
    assert green.any()


def test_snapshot_within_one_second_returns_same_image():
    clock = FakeClock()
    stream = CameraStream(clock)
    stream.publish(make_frame(fill=0))
    first = stream.snapshot_jpeg()
    stream.publish(make_frame(fill=200))  # 그 사이 화면이 바뀌어도
    clock.now += SNAPSHOT_REUSE_SEC - 0.01
    assert stream.snapshot_jpeg() == first


def test_snapshot_after_one_second_is_made_again():
    clock = FakeClock()
    stream = CameraStream(clock)
    stream.publish(make_frame(fill=0))
    first = stream.snapshot_jpeg()
    stream.publish(make_frame(fill=200))
    clock.now += SNAPSHOT_REUSE_SEC
    assert stream.snapshot_jpeg() != first


def test_concurrent_snapshots_encode_once(monkeypatch):
    from white_eyes_flask.stream import stream_service

    calls = []
    real = stream_service.render_jpeg

    def counting(*args, **kwargs):
        calls.append(1)
        time.sleep(0.05)
        return real(*args, **kwargs)

    monkeypatch.setattr(stream_service, "render_jpeg", counting)
    stream = CameraStream()
    stream.publish(make_frame())
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(stream.snapshot_jpeg()))
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1
    assert len(set(results)) == 1


# ---------- 프레임을 만들어야 하는 때 ----------


def test_does_not_want_frames_without_viewer_or_recent_snapshot():
    assert CameraStream().wants_frames() is False


def test_wants_frames_for_ten_seconds_after_snapshot_request():
    clock = FakeClock()
    stream = CameraStream(clock)
    stream.snapshot_jpeg()
    clock.now += SNAPSHOT_WANTED_SEC
    assert stream.wants_frames() is True
    clock.now += 0.01
    assert stream.wants_frames() is False


def test_wants_frames_while_mjpeg_viewer_exists():
    stream = CameraStream()
    stream.publish(make_frame())
    live = stream.live_jpegs()
    next(live)
    assert stream.wants_frames() is True
    live.close()
    assert stream.wants_frames() is False


# ---------- MJPEG ----------


def test_viewer_count_goes_up_while_open_and_down_when_closed():
    stream = CameraStream()
    stream.publish(make_frame())
    first, second = stream.live_jpegs(), stream.live_jpegs()
    next(first)
    next(second)
    assert stream.viewer_count == 2
    first.close()
    assert stream.viewer_count == 1
    second.close()
    assert stream.viewer_count == 0


def test_generator_never_started_does_not_count_as_viewer():
    stream = CameraStream()
    stream.live_jpegs()
    assert stream.viewer_count == 0


def test_live_frame_is_960px_wide_and_new_frames_follow():
    stream = CameraStream()
    stream.publish(make_frame(fill=0))
    live = stream.live_jpegs()
    first = next(live)
    assert decode(first).shape[1] == 960
    stream.publish(make_frame(fill=200))
    second = next(live)
    assert second != first
    live.close()


def test_live_waits_for_new_frame_and_does_not_resend_same_one():
    stream = CameraStream()
    stream.publish(make_frame())
    live = stream.live_jpegs()
    next(live)
    got = []
    t = threading.Thread(target=lambda: got.append(next(live, None)))
    t.start()
    time.sleep(0.2)
    assert got == []
    stream.publish(make_frame(fill=50))
    t.join(2)
    assert len(got) == 1 and got[0] is not None
    live.close()


def test_live_ends_when_stream_closes():
    stream = CameraStream()
    stream.publish(make_frame())
    live = stream.live_jpegs()
    next(live)
    done = threading.Event()

    def drain():
        list(live)
        done.set()

    threading.Thread(target=drain, daemon=True).start()
    stream.close()
    assert done.wait(2)
    assert stream.viewer_count == 0


def test_live_encodes_once_for_many_viewers(monkeypatch):
    from white_eyes_flask.stream import stream_service

    calls = []
    real = stream_service.render_jpeg
    monkeypatch.setattr(
        stream_service,
        "render_jpeg",
        lambda *a, **k: calls.append(1) or real(*a, **k),
    )
    stream = CameraStream()
    stream.publish(make_frame())
    viewers = [stream.live_jpegs() for _ in range(3)]
    jpegs = [next(v) for v in viewers]
    assert len(calls) == 1
    assert len(set(jpegs)) == 1
    for v in viewers:
        v.close()


def test_live_respects_min_interval():
    stream = CameraStream()
    stream.publish(make_frame())
    live = stream.live_jpegs(min_interval=0.3)
    next(live)
    stream.publish(make_frame(fill=9))
    started = time.monotonic()
    next(live)
    assert time.monotonic() - started >= 0.25
    live.close()


# ---------- HTTP ----------


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
        analyze_fps=50,
    )
    app = create_app(config, preparer=NullPreparer())
    yield app
    app.extensions["camera_service"].shutdown()


@pytest.fixture
def client(app):
    return app.test_client()


def start_cam(client, camera_id: int = 7) -> None:
    res = client.post(
        f"/analyzer/v1/cameras/{camera_id}/start", json=START_BODY, headers=HEADERS
    )
    assert res.status_code == 200


@pytest.mark.parametrize("tail", ["live", "snapshot"])
def test_requires_internal_token(client, tail):
    start_cam(client)
    assert client.get(f"/analyzer/v1/cameras/7/{tail}").status_code == 401
    wrong = {"X-Internal-Token": "nope"}
    assert (
        client.get(f"/analyzer/v1/cameras/7/{tail}", headers=wrong).status_code == 401
    )


@pytest.mark.parametrize("tail", ["live", "snapshot"])
def test_no_job_is_404(client, tail):
    res = client.get(f"/analyzer/v1/cameras/7/{tail}", headers=HEADERS)
    assert res.status_code == 404
    assert res.get_json()["code"] == "CAMERA_NOT_FOUND"


@pytest.mark.parametrize("tail", ["live", "snapshot"])
def test_after_stop_is_404(client, tail):
    start_cam(client)
    client.post("/analyzer/v1/cameras/7/stop", headers=HEADERS)
    assert (
        client.get(f"/analyzer/v1/cameras/7/{tail}", headers=HEADERS).status_code == 404
    )


def test_restart_after_stop_serves_again(client, app):
    start_cam(client)
    client.post("/analyzer/v1/cameras/7/stop", headers=HEADERS)
    start_cam(client)
    stream = app.extensions["stream_service"].get(7)
    stream.publish(make_frame())
    res = client.get("/analyzer/v1/cameras/7/snapshot", headers=HEADERS)
    assert res.status_code == 200


def test_snapshot_response_headers_and_body(client, app):
    start_cam(client)
    app.extensions["stream_service"].get(7).publish(make_frame())
    res = client.get("/analyzer/v1/cameras/7/snapshot?t=123", headers=HEADERS)
    assert res.status_code == 200
    assert res.mimetype == "image/jpeg"
    assert res.headers["Cache-Control"] == "no-store"
    assert decode(res.data).shape[1] == 320


def test_snapshot_twice_within_one_second_is_same_image(client, app):
    start_cam(client)
    stream = app.extensions["stream_service"].get(7)
    stream.publish(make_frame(fill=0))
    first = client.get("/analyzer/v1/cameras/7/snapshot", headers=HEADERS).data
    stream.publish(make_frame(fill=200))
    second = client.get("/analyzer/v1/cameras/7/snapshot", headers=HEADERS).data
    assert first == second


def test_snapshot_before_first_frame_is_503(client):
    start_cam(client)
    res = client.get("/analyzer/v1/cameras/7/snapshot", headers=HEADERS)
    assert res.status_code == 503
    assert res.get_json()["code"] == "FRAME_NOT_READY"


def test_live_is_multipart_and_closing_lowers_viewer_count(client, app):
    start_cam(client)
    stream = app.extensions["stream_service"].get(7)
    stream.publish(make_frame())
    res = client.get("/analyzer/v1/cameras/7/live", headers=HEADERS, buffered=False)
    assert res.status_code == 200
    assert res.headers["Content-Type"] == "multipart/x-mixed-replace; boundary=frame"
    assert res.headers["X-Accel-Buffering"] == "no"
    chunks = res.response
    head = next(chunks)
    jpeg = next(chunks)
    assert head.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n")
    assert f"Content-Length: {len(jpeg)}".encode() in head
    assert jpeg[:2] == b"\xff\xd8"
    assert stream.viewer_count == 1
    res.close()  # 창을 닫음
    assert wait_until(lambda: stream.viewer_count == 0)


def test_stop_ends_open_live_connection(client, app):
    start_cam(client)
    stream = app.extensions["stream_service"].get(7)
    stream.publish(make_frame())
    res = client.get("/analyzer/v1/cameras/7/live", headers=HEADERS, buffered=False)
    chunks = res.response
    next(chunks)
    next(chunks)
    client.post("/analyzer/v1/cameras/7/stop", headers=HEADERS)
    assert list(chunks)[-1:] in ([b"\r\n"], [])
    assert stream.viewer_count == 0
