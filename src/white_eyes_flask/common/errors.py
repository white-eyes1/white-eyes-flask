from flask import Response, jsonify


def error_response(code: str, message: str, status: int) -> tuple[Response, int]:
    """공통 오류 응답 {code, message} (conventions.md '오류 응답')."""
    return jsonify(code=code, message=message), status
