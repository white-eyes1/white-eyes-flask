import http.client
import json
import logging
import urllib.error
import urllib.request
from enum import StrEnum

logger = logging.getLogger(__name__)

TOKEN_HEADER = "X-Internal-Token"
# 같은 보고를 다시 보내도 소용없는 응답: 형식 오류(400), 캠 없음(404)
_REJECTED_STATUSES = {400, 404}
_AUTH_FAILURES = {401, 403}


class ReportResult(StrEnum):
    OK = "OK"
    RETRY = "RETRY"  # 연결 실패·타임아웃·5xx와 400·404를 뺀 4xx: 다시 보낼 만함
    REJECTED = "REJECTED"  # 400·404: 다시 보내도 소용없음


class StatusReporter:
    """분석 서버가 관리자 앱에 캠 상태를 알린다(PATCH /internal/v1/cameras/{id}/status).

    보고가 실패해도 분석을 멈추지 않는다. report()는 실패를 로그만 남기고 False를 돌려준다.
    다시 보낼지 가려야 하면 send()로 결과(ReportResult)를 받아 호출한 쪽이 로그를 남긴다.
    """

    def __init__(self, base_url: str, token: str, timeout: float = 2.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout

    def report(
        self,
        camera_id: int,
        status: str,
        progress: int | None = None,
        error_message: str | None = None,
    ) -> bool:
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
        body = json.dumps(
            {"status": status, "progress": progress, "errorMessage": error_message}
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}/internal/v1/cameras/{camera_id}/status",
            data=body,
            method="PATCH",
            headers={"Content-Type": "application/json", TOKEN_HEADER: self._token},
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout):
                return ReportResult.OK, ""
        except urllib.error.HTTPError as error:
            error.close()
            detail = str(error)
            if error.code in _AUTH_FAILURES:
                detail += " (인증 실패: 토큰 설정을 확인하세요)"
            if error.code in _REJECTED_STATUSES:
                return ReportResult.REJECTED, detail
            return ReportResult.RETRY, detail
        except (OSError, ValueError, http.client.HTTPException) as error:
            # URLError·타임아웃, 응답 형식이 깨진 경우 포함
            return ReportResult.RETRY, str(error)
