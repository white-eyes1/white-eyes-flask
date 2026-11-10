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
    body: dict[str, object] = {"code": code, "message": message}
    if field_errors:
        body["fieldErrors"] = field_errors
    return jsonify(body), status
