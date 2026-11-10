"""FNC-S-011 실시간 화면·카드 미리보기 API (/analyzer/v1/cameras/{id}/live, /snapshot)"""

from typing import Iterator

from flask import Blueprint, Response, current_app

from white_eyes_flask.common.errors import error_response
from white_eyes_flask.stream.stream_service import StreamService

stream_bp = Blueprint("stream", __name__, url_prefix="/analyzer/v1/cameras")

BOUNDARY = "frame"


@stream_bp.get("/<int:camera_id>/live")
def live(camera_id: int) -> Response | tuple[Response, int]:
    service: StreamService = current_app.extensions["stream_service"]
    stream = service.get(camera_id)
    if stream is None:
        return _no_job()
    fps = max(1, current_app.config["APP_CONFIG"].analyze_fps)
    response = Response(
        _multipart(stream.live_jpegs(min_interval=1 / fps)),
        content_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
    )
    response.headers["Cache-Control"] = "no-store"
    # nginx가 응답을 모아 두지 않고 바로 흘려보내게 한다(proxy_buffering off와 같은 뜻).
    response.headers["X-Accel-Buffering"] = "no"
    return response


@stream_bp.get("/<int:camera_id>/snapshot")
def snapshot(camera_id: int) -> Response | tuple[Response, int]:
    service: StreamService = current_app.extensions["stream_service"]
    stream = service.get(camera_id)
    if stream is None:
        return _no_job()
    jpeg = stream.snapshot_jpeg()
    if jpeg is None:
        # 작업은 있으나 첫 프레임이 아직 없다(명세에 없는 경우). 곧 다시 요청하면 받는다.
        response, _ = error_response(
            "FRAME_NOT_READY", "아직 보여줄 화면이 없습니다.", 503
        )
        response.headers["Retry-After"] = "1"
        return response, 503
    response = Response(jpeg, mimetype="image/jpeg")
    response.headers["Cache-Control"] = "no-store"
    return response


def _no_job() -> tuple[Response, int]:
    return error_response("CAMERA_NOT_FOUND", "가동 중인 캠 작업이 없습니다.", 404)


def _multipart(jpegs: Iterator[bytes]) -> Iterator[bytes]:
    for jpeg in jpegs:
        yield (
            f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
            f"Content-Length: {len(jpeg)}\r\n\r\n"
        ).encode("ascii")
        yield jpeg
        yield b"\r\n"
