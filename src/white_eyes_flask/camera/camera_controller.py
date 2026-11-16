"""FNC-S-001·002 캠 시작·정지 API (/analyzer/v1/cameras)"""

from flask import Blueprint, Response, current_app, request

from white_eyes_flask.camera.camera_service import (
    CameraService,
    CameraStoppingError,
    SourceNotFoundError,
)
from white_eyes_flask.common.errors import error_response

# 이 모듈의 라우트를 모두 /analyzer/v1/cameras 아래에 두는 블루프린트.
camera_bp = Blueprint("camera", __name__, url_prefix="/analyzer/v1/cameras")

# 요청의 sourceType으로 허용하는 값. LIVE는 RTSP 실시간 캠, DUMMY는 공유 폴더의 더미 영상 파일이다.
SOURCE_TYPES = {"LIVE", "DUMMY"}


@camera_bp.post("/<int:camera_id>/start")
def start_camera(camera_id: int) -> dict[str, object] | tuple[Response, int]:
    """FNC-S-001 캠 시작.

    본문(sourceType, sourceUri)의 형식을 검사한 뒤 CameraService.start에 맡기고
    {cameraId, status}를 돌려준다. 이미 돌고 있는 캠이면 현재 상태를 그대로 돌려준다.
    오류: 본문 형식 400 INVALID_INPUT, 영상 없음 400 SOURCE_NOT_FOUND, 정지 지연 409 CAMERA_STOPPING.
    """
    # silent=True: JSON이 아니거나 깨져도 예외 대신 None이 되어 아래에서 형식 오류로 처리한다.
    body = request.get_json(silent=True)
    # JSON 객체가 아니면(배열·문자열·본문 없음) 필드를 꺼낼 수 없다.
    if not isinstance(body, dict):
        return _invalid_input([("body", "JSON 객체여야 합니다.")])

    # 필드별로 값을 꺼낸 뒤 문제를 모아서 한 번에 알려 준다(첫 오류에서 바로 끝내지 않는다).
    source_type = body.get("sourceType")
    source_uri = body.get("sourceUri")
    # (필드 이름, 사유) 목록. 비어 있으면 입력이 올바른 것이다.
    problems: list[tuple[str, str]] = []
    # 문자열이면서 허용 값(LIVE·DUMMY) 중 하나여야 한다.
    if not isinstance(source_type, str) or source_type not in SOURCE_TYPES:
        problems.append(("sourceType", "LIVE 또는 DUMMY여야 합니다."))
    # 여기서는 문자열인지만 본다. rtsp 주소·파일 경로 규칙은 CameraService가 검사한다.
    if not isinstance(source_uri, str):
        problems.append(("sourceUri", "문자열이어야 합니다."))
    # 모든 필드 오류를 한 응답에 담아 돌려준다.
    if problems:
        return _invalid_input(problems)

    # 앱에 등록된 CameraService 하나를 모든 요청이 공유한다.
    service: CameraService = current_app.extensions["camera_service"]
    try:
        # 이미 돌고 있으면 새로 만들지 않고 현재 상태를 받는다(요청 내용은 무시). 그래서 재요청해도 안전하다.
        status = service.start(camera_id, source_type, source_uri)
    # DUMMY 파일이 없거나 경로 규칙에 어긋남, 또는 LIVE 주소가 rtsp://로 시작하지 않음.
    except SourceNotFoundError:
        return error_response("SOURCE_NOT_FOUND", "영상 파일을 찾을 수 없습니다.", 400)
    # 같은 캠의 옛 작업 스레드가 아직 정지 중이다. 잠시 뒤 다시 시작하면 되므로 409로 알린다.
    except CameraStoppingError:
        return error_response(
            "CAMERA_STOPPING",
            "정지가 아직 끝나지 않았습니다. 잠시 뒤 다시 시작해 주세요.",
            409,
        )
    # 시작이 받아들여졌다. status는 시작 시점의 작업 상태(CameraStatus)다.
    return {"cameraId": camera_id, "status": status}


@camera_bp.post("/<int:camera_id>/stop")
def stop_camera(camera_id: int) -> Response:
    """FNC-S-002 캠 정지.

    작업이 없어도 오류 없이 200을 돌려준다(멱등).
    """
    service: CameraService = current_app.extensions["camera_service"]
    # 정지가 10초 안에 끝나지 않아도 예외 없이 돌아온다(그 경우의 처리는 CameraService.stop 참고).
    service.stop(camera_id)
    # 본문 없는 200.
    return Response(status=200)


def _invalid_input(problems: list[tuple[str, str]]) -> tuple[Response, int]:
    """INVALID_INPUT 오류 응답을 만든다. problems의 (필드, 사유)를 fieldErrors 항목으로 바꿔 담는다."""
    field_errors = [{"field": field, "message": text} for field, text in problems]
    return error_response(
        "INVALID_INPUT", "요청 형식이 올바르지 않습니다.", 400, field_errors
    )
