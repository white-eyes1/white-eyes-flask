"""FNC-S-006 1초 결과 묶음 전송 (POST /internal/v1/results)"""

import http.client
import json
import logging
import urllib.error
import urllib.request
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)

TOKEN_HEADER = "X-Internal-Token"
# 같은 묶음을 다시 보내도 소용없는 응답은 형식 오류(400)뿐이다. 그 밖의 4xx(토큰 설정 실수인
# 401·403 포함)는 원인을 고친 뒤 보관 중인 결과를 살릴 수 있도록 다시 보낸다(PM 결정).
_REJECTED_STATUSES = {400}
_LOG_EVERY = 30


class SendResult(StrEnum):
    OK = "OK"
    RETRY = "RETRY"  # 연결 실패·타임아웃·5xx와 400을 뺀 4xx(401·403·408·429 등): 다시 보낼 만함
    REJECTED = "REJECTED"  # 400(형식 오류): 같은 묶음을 다시 보내도 소용없음


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        # 따라가지 않는다. 토큰이 다른 주소로 가지 않게 한다.
        return None


class ResultClient:
    """분석 서버가 관리자 앱에 1초 결과 묶음을 보낸다(POST /internal/v1/results)."""

    def __init__(self, base_url: str, token: str, timeout: float = 3.0) -> None:
        self._url = f"{base_url.rstrip('/')}/internal/v1/results"
        self._token = token
        self._timeout = timeout
        self._opener = urllib.request.build_opener(_NoRedirect)
        self._failures = 0  # 연속으로 실패한 횟수(로그가 초마다 쌓이지 않게)

    def send(self, items: list[dict[str, Any]]) -> SendResult:
        request = urllib.request.Request(
            self._url,
            data=json.dumps(items).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json", TOKEN_HEADER: self._token},
        )
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                status = response.status
        except urllib.error.HTTPError as error:
            error.close()  # 응답 본문·연결 자원을 바로 놓는다
            return self._failed(error.code, len(items))
        except (OSError, ValueError, http.client.HTTPException) as error:
            # URLError·타임아웃, 응답 형식이 깨진 경우(BadStatusLine 등) 포함
            # 토큰은 로그에 남기지 않는다
            self._warn("결과 전송 실패(%s건): %s", len(items), error)
            return SendResult.RETRY
        if 200 <= status < 300:
            if self._failures:
                logger.info("결과 전송 복구(연속 %s번 실패 뒤)", self._failures)
            self._failures = 0
            return SendResult.OK
        self._warn("결과 전송 예상 밖 응답 status=%s(%s건)", status, len(items))
        return SendResult.RETRY

    def _failed(self, status: int, count: int) -> SendResult:
        if status in _REJECTED_STATUSES:
            self._failures = 0  # 서버는 응답했다
            logger.error("결과 전송 거절 status=%s(%s건)", status, count)
            return SendResult.REJECTED
        if status in (401, 403):
            self._warn(
                "결과 전송 인증 실패 status=%s(%s건), 토큰 설정을 확인하세요",
                status,
                count,
            )
            return SendResult.RETRY
        self._warn("결과 전송 실패 status=%s(%s건)", status, count)
        return SendResult.RETRY

    def _warn(self, message: str, *args: Any) -> None:
        """연속 실패는 처음과 30번째마다만 남긴다(서버가 오래 꺼져 있어도 로그가 넘치지 않게)."""
        self._failures += 1
        if self._failures == 1 or self._failures % _LOG_EVERY == 0:
            logger.warning(message + " (연속 %s번째)", *args, self._failures)
