"""관리자 앱 내부 API 호출(package-structure.md 5).

결과 전송(FNC-S-006), 캠 상태 보고(FNC-S-001 후속 동작), 녹화 등록(FNC-S-008·010),
재시작 복구의 가동 대상 조회(FNC-S-012)가 이 모듈로만 관리자 앱을 부른다.
주소·X-Internal-Token 헤더·시간 제한·리다이렉트 막기를 여기서 한 번만 처리하고,
응답을 어떻게 다룰지(재시도·거절·로그)는 부르는 쪽이 정한다.
"""

import http.client
import json
import urllib.request
from typing import Any

TOKEN_HEADER = "X-Internal-Token"
# 본문 없는 요청(GET)과 JSON null 본문을 가른다
_NO_BODY = object()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        # 따라가지 않는다. 토큰이 다른 주소로 가지 않게 한다(3xx는 HTTPError로 올라간다).
        return None


class AdminClient:
    """관리자 앱 내부 API(/internal/v1/**)를 부른다."""

    def __init__(self, base_url: str, token: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._opener = urllib.request.build_opener(_NoRedirect)

    def open(
        self, method: str, path: str, *, timeout: float, body: Any = _NO_BODY
    ) -> http.client.HTTPResponse:
        """요청을 보내고 응답을 돌려준다. with 문으로 닫는다.

        2xx가 아니면 urllib.error.HTTPError, 연결 실패·타임아웃·깨진 응답은 OSError·ValueError·
        http.client.HTTPException을 그대로 올린다. 토큰은 예외 메시지에 들어가지 않는다.

        :param path: 기준 주소 뒤의 경로(예: /internal/v1/results)
        :param body: JSON으로 보낼 값. 주지 않으면 본문 없이 보낸다
        """
        headers = {TOKEN_HEADER: self._token}
        data = None
        if body is not _NO_BODY:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self._base_url}{path}", data=data, method=method, headers=headers
        )
        return self._opener.open(request, timeout=timeout)
