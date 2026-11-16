"""캠 상태 보고 (PATCH /internal/v1/cameras/{id}/status)

FNC-S-001 후속 동작에서 정한 보고다. 준비·재생·라이브·재시작 복구(FNC-S-007·009·010·012)가
같이 쓰고, 관리자 앱이 받는다(FNC-A-025).
"""

import http.client
import logging
import urllib.error
from enum import StrEnum

from white_eyes_flask.camera.admin_client import AdminClient

logger = logging.getLogger(__name__)

# 같은 보고를 다시 보내도 소용없는 응답: 형식 오류(400), 캠 없음(404)
_REJECTED_STATUSES = {400, 404}
# 토큰 설정 실수로 보는 응답 코드. 로그에 안내 문구를 붙이는 데만 쓴다.
_AUTH_FAILURES = {401, 403}


class ReportResult(StrEnum):
    """상태 보고 결과. 호출한 쪽이 이 값으로 다시 보낼지 정한다."""

    OK = "OK"
    RETRY = "RETRY"  # 연결 실패·타임아웃·5xx와 400·404를 뺀 4xx: 다시 보낼 만함
    REJECTED = "REJECTED"  # 400·404: 다시 보내도 소용없음


class StatusReporter:
    """분석 서버가 관리자 앱에 캠 상태를 알린다(PATCH /internal/v1/cameras/{id}/status).

    보고가 실패해도 분석을 멈추지 않는다. report()는 실패를 로그만 남기고 False를 돌려준다.
    다시 보낼지 가려야 하면 send()로 결과(ReportResult)를 받아 호출한 쪽이 로그를 남긴다.
    """

    def __init__(self, base_url: str, token: str, timeout: float = 2.0) -> None:
        """보고기를 만든다.

        :param base_url: 관리자 앱 내부 API의 기준 주소
        :param token: 내부 API 인증 토큰
        :param timeout: 요청 제한 시간(초). 분석을 오래 붙잡지 않게 짧다
        """
        self._admin = AdminClient(base_url, token)
        self._timeout = timeout

    def report(
        self,
        camera_id: int,
        status: str,
        progress: int | None = None,
        error_message: str | None = None,
    ) -> bool:
        """상태를 보고하고 성공했는지만 돌려준다. 실패하면 경고 로그를 남긴다.

        :param status: PREPARING, RUNNING, ERROR 중 하나
        :param progress: 준비 진행률(0~100). 해당이 없으면 None
        :param error_message: ERROR일 때의 사유. 해당이 없으면 None
        """
        result, detail = self.send(camera_id, status, progress, error_message)
        if result == ReportResult.OK:
            return True
        logger.warning(
            "상태 보고 실패 cameraId=%s status=%s: %s", camera_id, status, detail
        )
        return False

    def send(
        self,
        camera_id: int,
        status: str,
        progress: int | None = None,
        error_message: str | None = None,
    ) -> tuple[ReportResult, str]:
        """보고하고 (결과, 실패 이유)를 돌려준다. 로그는 남기지 않는다. 토큰은 이유에 넣지 않는다."""
        # 진행률과 사유는 해당이 없으면 null로 보낸다.
        body = {"status": status, "progress": progress, "errorMessage": error_message}
        try:
            # 2xx면 정상이다. 응답 본문은 읽지 않고 with 문이 응답을 닫는다.
            with self._admin.open(
                "PATCH",
                f"/internal/v1/cameras/{camera_id}/status",
                timeout=self._timeout,
                body=body,
            ):
                return ReportResult.OK, ""
        # 서버가 응답은 했지만 2xx가 아니다. 코드로 거절과 재시도를 가른다.
        except urllib.error.HTTPError as error:
            error.close()
            detail = str(error)
            if error.code in _AUTH_FAILURES:
                detail += " (인증 실패: 토큰 설정을 확인하세요)"
            # 400·404는 같은 보고를 다시 보내도 소용없다(형식 오류, 캠이 없음).
            if error.code in _REJECTED_STATUSES:
                return ReportResult.REJECTED, detail
            return ReportResult.RETRY, detail
        except (OSError, ValueError, http.client.HTTPException) as error:
            # URLError·타임아웃, 응답 형식이 깨진 경우 포함
            return ReportResult.RETRY, str(error)
