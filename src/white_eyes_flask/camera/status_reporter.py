import json
import logging
import urllib.request

logger = logging.getLogger(__name__)

TOKEN_HEADER = "X-Internal-Token"


class StatusReporter:
    """분석 서버가 관리자 앱에 캠 상태를 알린다(PATCH /internal/v1/cameras/{id}/status).

    보고가 실패해도 분석을 멈추지 않는다. 실패는 로그만 남기고 False를 돌려준다.
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
                return True
        except (OSError, ValueError) as error:  # HTTPError·URLError 포함
            # 토큰은 로그에 남기지 않는다
            logger.warning(
                "상태 보고 실패 cameraId=%s status=%s: %s", camera_id, status, error
            )
            return False
