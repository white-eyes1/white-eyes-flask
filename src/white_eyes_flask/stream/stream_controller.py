"""FNC-S-011 실시간 화면·카드 미리보기 API (/analyzer/v1/cameras/{id}/live, /snapshot)"""

from typing import Iterator

from flask import Blueprint, Response, current_app

from white_eyes_flask.common.errors import error_response
from white_eyes_flask.stream.stream_service import StreamService

stream_bp = Blueprint("stream", __name__, url_prefix="/analyzer/v1/cameras")

# multipart 응답에서 JPEG 조각을 가르는 경계 문자열
BOUNDARY = "frame"


@stream_bp.get("/<int:camera_id>/live")
def live(camera_id: int) -> Response | tuple[Response, int]:
    """실시간 화면을 MJPEG(JPEG를 계속 이어 보내는 multipart 스트림)으로 보낸다. 캠 작업이 없으면 404."""
    service: StreamService = current_app.extensions["stream_service"]
    stream = service.get(camera_id)
    if stream is None:
        return _no_job()
    # 분석 fps 이하로만 보내 불필요한 변환·전송을 줄인다(0이어도 0으로 나누지 않게 최소 1)
    fps = max(1, current_app.config["APP_CONFIG"].analyze_fps)
    # 제너레이터를 응답 본문으로 넘기면 Flask가 조각이 나올 때마다 바로 내려보낸다.
    # mixed-replace는 브라우저가 이전 이미지를 새 이미지로 계속 갈아 끼우게 하는 형식이다
    response = Response(
        _multipart(stream.live_jpegs(min_interval=1 / fps)),
        content_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
    )
    # 실시간 영상이 캐시에 남지 않게 한다
    response.headers["Cache-Control"] = "no-store"
    # nginx가 응답을 모아 두지 않고 바로 흘려보내게 한다(proxy_buffering off와 같은 뜻).
    response.headers["X-Accel-Buffering"] = "no"
    return response


@stream_bp.get("/<int:camera_id>/snapshot")
def snapshot(camera_id: int) -> Response | tuple[Response, int]:
    """카드 목록에 쓰는 정지 이미지(JPEG 한 장)를 돌려준다. 캠 작업이 없으면 404, 첫 프레임 전이면 503."""
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
        # 1초 뒤에 다시 요청하라는 힌트
        response.headers["Retry-After"] = "1"
        return response, 503
    response = Response(jpeg, mimetype="image/jpeg")
    response.headers["Cache-Control"] = "no-store"
    return response


def _no_job() -> tuple[Response, int]:
    """가동 중인 캠 작업이 없을 때의 공통 404 응답."""
    return error_response("CAMERA_NOT_FOUND", "가동 중인 캠 작업이 없습니다.", 404)


def _multipart(jpegs: Iterator[bytes]) -> Iterator[bytes]:
    """JPEG 바이트를 multipart/x-mixed-replace 조각(경계선 + 헤더 + 본문 + 줄바꿈)으로 감싸 이어 보낸다."""
    for jpeg in jpegs:
        # 조각 머리: 경계선과 이 조각의 형식·길이
        yield (
            f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
            f"Content-Length: {len(jpeg)}\r\n\r\n"
        ).encode("ascii")
        yield jpeg
        yield b"\r\n"
