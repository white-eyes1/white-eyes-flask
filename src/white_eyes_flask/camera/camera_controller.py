from flask import Blueprint, Response, current_app, request

from white_eyes_flask.camera.camera_service import (
    CameraService,
    SourceNotFoundError,
    SourceTypeNotSupportedError,
)
from white_eyes_flask.common.errors import error_response

camera_bp = Blueprint("camera", __name__, url_prefix="/analyzer/v1/cameras")

SOURCE_TYPES = {"LIVE", "DUMMY"}


@camera_bp.post("/<int:camera_id>/start")
def start_camera(camera_id: int) -> dict[str, object] | tuple[Response, int]:
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return _invalid_request([("body", "JSON 객체여야 합니다.")])

    source_type = body.get("sourceType")
    source_uri = body.get("sourceUri")
    problems: list[tuple[str, str]] = []
    if not isinstance(source_type, str) or source_type not in SOURCE_TYPES:
        problems.append(("sourceType", "LIVE 또는 DUMMY여야 합니다."))
    if not isinstance(source_uri, str):
        problems.append(("sourceUri", "문자열이어야 합니다."))
    if problems:
        return _invalid_request(problems)

    service: CameraService = current_app.extensions["camera_service"]
    try:
        status = service.start(camera_id, source_type, source_uri)
    except SourceNotFoundError:
        return error_response("SOURCE_NOT_FOUND", "영상 파일을 찾을 수 없습니다.", 400)
    except SourceTypeNotSupportedError:
        return error_response(
            "SOURCE_TYPE_NOT_SUPPORTED", "아직 지원하지 않는 영상 종류입니다.", 501
        )
    return {"cameraId": camera_id, "status": status}


def _invalid_request(problems: list[tuple[str, str]]) -> tuple[Response, int]:
    field_errors = [{"field": field, "message": text} for field, text in problems]
    return error_response(
        "INVALID_REQUEST", "요청 형식이 올바르지 않습니다.", 400, field_errors
    )
