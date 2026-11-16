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

# 관리자 앱이 내부 API 호출자를 확인하는 인증 헤더 이름. 값은 설정의 내부 토큰이다.
TOKEN_HEADER = "X-Internal-Token"
# 본문 없는 요청(GET)과 JSON null 본문을 가른다
_NO_BODY = object()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """리다이렉트(3xx)를 따라가지 않는 핸들러.

    기본 동작은 Location 주소로 같은 요청(헤더 포함)을 다시 보내는 것이라, 토큰 헤더가
    엉뚱한 주소로 새어 나갈 수 있어 막는다.
    """

    def redirect_request(self, *args, **kwargs):
        """새 요청을 만들지 않고 None을 돌려 리다이렉트를 거절한다."""
        # 따라가지 않는다. 토큰이 다른 주소로 가지 않게 한다(3xx는 HTTPError로 올라간다).
        return None


class AdminClient:
    """관리자 앱 내부 API(/internal/v1/**)를 부른다."""

    def __init__(self, base_url: str, token: str) -> None:
        """관리자 앱 호출 준비.

        :param base_url: 관리자 앱 내부 API의 기준 주소(예: http://host:8080/admin)
        :param token: X-Internal-Token 헤더에 실을 공유 토큰
        """
        # 끝의 /를 떼어 두면 path(항상 /로 시작)와 이어 붙여도 //가 생기지 않는다.
        self._base_url = base_url.rstrip("/")
        self._token = token
        # 기본 opener 대신 리다이렉트를 막은 opener를 만들어 모든 요청에 쓴다.
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
        # 모든 요청에 토큰 헤더를 붙인다.
        headers = {TOKEN_HEADER: self._token}
        # GET처럼 본문이 없으면 data=None으로 둔다(urllib은 data가 있으면 본문 요청으로 본다).
        data = None
        # body를 주지 않은 경우(_NO_BODY)와 JSON null(None)을 주려는 경우를 구분한다.
        if body is not _NO_BODY:
            # 한글이 들어가도 되도록 UTF-8 바이트로 직렬화한다.
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        # 메서드·헤더·본문을 담은 요청 객체를 만든다(아직 보내지 않는다).
        request = urllib.request.Request(
            f"{self._base_url}{path}", data=data, method=method, headers=headers
        )
        # 실제 전송. 반환한 응답은 호출한 쪽이 with 문으로 닫는다. 실패는 예외로 그대로 올라간다.
        return self._opener.open(request, timeout=timeout)
