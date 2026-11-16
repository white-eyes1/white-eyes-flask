"""모든 API가 같은 모양으로 오류를 돌려주게 하는 공통 응답 도우미."""

from flask import Response, jsonify


def error_response(
    code: str,
    message: str,
    status: int,
    field_errors: list[dict[str, str]] | None = None,
) -> tuple[Response, int]:
    """공통 오류 응답 (conventions.md '오류 응답').

    fieldErrors는 입력 검증 오류일 때만 넣는다.
    """
    # 모든 오류 응답에 공통인 기본 본문(코드 + 사람이 읽는 메시지)
    body: dict[str, object] = {"code": code, "message": message}
    # 빈 목록이면 키 자체를 빼서, 검증 오류가 아닌 응답에는 fieldErrors가 보이지 않게 한다
    if field_errors:
        body["fieldErrors"] = field_errors
    # (JSON 응답, HTTP 상태 코드) 튜플은 Flask가 그대로 응답으로 바꿔 준다
    return jsonify(body), status
