"""FNC-S-006 1초 결과 묶음 전송 (POST /internal/v1/results)"""

import http.client
import logging
import urllib.error
from enum import StrEnum
from typing import Any

from white_eyes_flask.camera.admin_client import AdminClient

logger = logging.getLogger(__name__)

# 같은 묶음을 다시 보내도 소용없는 응답은 형식 오류(400)뿐이다. 그 밖의 4xx(토큰 설정 실수인
# 401·403 포함)는 원인을 고친 뒤 보관 중인 결과를 살릴 수 있도록 다시 보낸다(PM 결정).
_REJECTED_STATUSES = {400}
# 같은 실패가 이어질 때 처음과 이 횟수마다만 로그를 남긴다.
_LOG_EVERY = 30


class SendResult(StrEnum):
    """묶음 전송 결과. ResultSender가 이 값으로 묶음을 버릴지 다시 보낼지 정한다."""

    OK = "OK"
    RETRY = "RETRY"  # 연결 실패·타임아웃·5xx와 400을 뺀 4xx(401·403·408·429 등): 다시 보낼 만함
    REJECTED = "REJECTED"  # 400(형식 오류): 같은 묶음을 다시 보내도 소용없음


class ResultClient:
    """분석 서버가 관리자 앱에 1초 결과 묶음을 보낸다(POST /internal/v1/results)."""

    def __init__(self, base_url: str, token: str, timeout: float = 3.0) -> None:
        """전송기를 만든다.

        :param base_url: 관리자 앱 내부 API의 기준 주소
        :param token: 내부 API 인증 토큰
        :param timeout: 요청 제한 시간(초)
        """
        self._admin = AdminClient(base_url, token)
        self._timeout = timeout
        self._failures = 0  # 연속으로 실패한 횟수(로그가 초마다 쌓이지 않게)

    def send(self, items: list[dict[str, Any]]) -> SendResult:
        """1초 결과 묶음(JSON 배열)을 한 번 보낸다.

        성공하면 OK, 400이면 REJECTED, 그 밖의 실패는 모두 RETRY다. 예외를 올리지 않는다.
        """
        try:
            with self._admin.open(
                "POST", "/internal/v1/results", timeout=self._timeout, body=items
            ) as response:
                status = response.status
        except urllib.error.HTTPError as error:
            error.close()  # 응답 본문·연결 자원을 바로 놓는다
            # 서버가 응답은 했지만 2xx가 아니다. 코드별 처리는 _failed에서 한다.
            return self._failed(error.code, len(items))
        except (OSError, ValueError, http.client.HTTPException) as error:
            # URLError·타임아웃, 응답 형식이 깨진 경우(BadStatusLine 등) 포함
            # 토큰은 로그에 남기지 않는다
            self._warn("결과 전송 실패(%s건): %s", len(items), error)
            return SendResult.RETRY
        # 2xx면 성공이다. 그동안 실패가 쌓였다면 복구 로그를 남기고 연속 실패 기록을 지운다.
        if 200 <= status < 300:
            if self._failures:
                logger.info("결과 전송 복구(연속 %s번 실패 뒤)", self._failures)
            self._failures = 0
            return SendResult.OK
        # 2xx도 아니고 예외도 아닌 예상 밖 응답은 재시도로 본다.
        self._warn("결과 전송 예상 밖 응답 status=%s(%s건)", status, len(items))
        return SendResult.RETRY

    def _failed(self, status: int, count: int) -> SendResult:
        """HTTP 오류 코드를 SendResult로 바꾸고 로그를 남긴다. 400만 거절이고 나머지는 모두 재시도다."""
        if status in _REJECTED_STATUSES:
            self._failures = 0  # 서버는 응답했다
            logger.error("결과 전송 거절 status=%s(%s건)", status, count)
            return SendResult.REJECTED
        # 토큰 설정 실수일 수 있다. 원인을 고친 뒤 보관 중인 결과를 살릴 수 있도록 버리지 않고 재시도한다.
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
