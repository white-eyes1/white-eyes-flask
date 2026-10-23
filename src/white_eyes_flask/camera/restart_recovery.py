"""FNC-S-012 분석 서버 재시작 복구 (GET /internal/v1/cameras/desired)"""

import http.client
import json
import logging
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from white_eyes_flask.camera.camera_service import (
    CameraStoppingError,
    SourceNotFoundError,
)
from white_eyes_flask.camera.status_reporter import ReportResult

logger = logging.getLogger(__name__)

TOKEN_HEADER = "X-Internal-Token"
RETRY_INTERVAL_SEC = 10
SHUTDOWN_JOIN_SEC = 2.0
_LOG_EVERY = 30
ERROR_SOURCE_NOT_FOUND = "영상 파일을 찾을 수 없음"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        # 따라가지 않는다. 토큰이 다른 주소로 가지 않게 한다.
        return None


class DesiredClient:
    """가동 대상 캠 목록을 관리자 앱에서 받는다. 실패하면 None을 돌려주고 로그만 남긴다."""

    def __init__(self, base_url: str, token: str, timeout: float = 5.0) -> None:
        self._url = f"{base_url.rstrip('/')}/internal/v1/cameras/desired"
        self._token = token
        self._timeout = timeout
        self._opener = urllib.request.build_opener(_NoRedirect)
        self._failures = 0

    def fetch(self) -> list[Any] | None:
        request = urllib.request.Request(
            self._url, method="GET", headers={TOKEN_HEADER: self._token}
        )
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as error:
            error.close()
            if error.code in (401, 403):
                self._warn(
                    "가동 대상 조회 인증 실패 status=%s, 토큰 설정을 확인하세요",
                    error.code,
                )
            else:
                self._warn("가동 대상 조회 실패 status=%s", error.code)
            return None
        except (OSError, ValueError, http.client.HTTPException) as error:
            # 연결 실패·타임아웃·깨진 응답. 토큰은 로그에 남기지 않는다
            self._warn("가동 대상 조회 실패: %s", error)
            return None
        if not isinstance(payload, list):
            self._warn("가동 대상 조회 응답이 목록이 아님")
            return None
        if self._failures:
            logger.info("가동 대상 조회 복구(연속 %s번 실패 뒤)", self._failures)
        self._failures = 0
        return payload

    def _warn(self, message: str, *args: Any) -> None:
        """연속 실패는 처음과 30번째마다만 남긴다(관리자 앱이 오래 꺼져 있어도 로그가 넘치지 않게)."""
        self._failures += 1
        if self._failures == 1 or self._failures % _LOG_EVERY == 0:
            logger.warning(message + " (연속 %s번째)", *args, self._failures)


class Starter(Protocol):
    def start(self, camera_id: int, source_type: str, source_uri: str) -> str: ...

    def current_status(self, camera_id: int) -> str | None: ...


class RestartRecovery:
    """서버가 켜질 때 가동 대상 캠을 받아 FNC-S-001과 같은 방식(CameraService.start)으로 시작한다.

    관리자 앱에 닿지 못하면 retry_interval마다 계속 다시 시도한다. 서버 시작을 막지 않게
    백그라운드 스레드에서 돈다. 이미 가동 중인 캠은 CameraService.start가 현재 상태만
    돌려주므로 중복으로 시작되지 않는다. 한 캠이 실패해도 다음 캠을 계속 처리한다.

    다시 시도하는 일은 두 가지다. 정지가 끝나지 않아 못 시작한 캠, 상태 보고에 실패한 캠.
    보고는 다시 보낼 때 저장해 둔 값이 아니라 그때의 캠 상태를 보내고, 그 사이 캠 작업이
    없어졌으면(관리자가 정지) 버린다. 관리자 앱이 400·404로 거절한 보고도 버린다.
    """

    def __init__(
        self,
        client: Any,
        service: Starter,
        reporter: Any,
        *,
        retry_interval: float = RETRY_INTERVAL_SEC,
    ) -> None:
        self._client = client
        self._service = service
        self._reporter = reporter
        self._retry_interval = retry_interval
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self.run, name="restart-recovery", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(SHUTDOWN_JOIN_SEC)

    def run(self) -> None:
        """목록을 받을 때까지, 그리고 받은 뒤 남은 일이 없어질 때까지 반복한다."""
        pending: list[Any] | None = None  # None이면 목록을 아직 못 받음
        while not self._stop.is_set():
            if pending is None:
                pending = self._client.fetch()
            if pending is not None:
                pending = self.recover(pending)
                if not pending:
                    return
            self._stop.wait(self._retry_interval)

    def recover(self, desired: list[Any]) -> list[Any]:
        """목록의 캠을 시작하고, 나중에 다시 해야 할 일(정지 중이라 못 시작, 상태 보고 실패)을 돌려준다."""
        retry: list[Any] = []
        for item in desired:
            if self._stop.is_set():
                return []
            again = self._process(item)
            if again is not None:
                retry.append(again)
        logger.info("재시작 복구 대상=%s건 다시 시도=%s건", len(desired), len(retry))
        return retry

    def _process(self, item: Any) -> "_ReportAgain | dict | None":
        """다시 시도할 일을 돌려준다. 끝났거나 다시 해도 소용없으면 None."""
        if isinstance(item, _ReportAgain):
            return self._deliver(item)
        parsed = _parse(item)
        if parsed is None:
            logger.error("가동 대상 항목 형식이 올바르지 않아 건너뜀: %r", item)
            return None
        camera_id, source_type, source_uri = parsed
        try:
            self._service.start(camera_id, source_type, source_uri)
        except SourceNotFoundError:
            # DUMMY 파일이 없거나 LIVE 주소가 rtsp://가 아님
            logger.error(
                "복구할 영상 파일이 없거나 주소가 올바르지 않음 cameraId=%s sourceType=%s",
                camera_id,
                source_type,
            )
            return self._deliver(
                _ReportAgain(camera_id, ERROR_SOURCE_NOT_FOUND, follow_job=False)
            )
        except CameraStoppingError:
            # 정지가 끝나면 시작할 수 있다. 같은 항목을 다음 차례에 다시 시도한다.
            logger.warning("정지가 끝나지 않아 나중에 다시 시도 cameraId=%s", camera_id)
            return item
        except Exception:
            logger.exception("캠 복구 중 예외 cameraId=%s", camera_id)
            return None
        return self._deliver(_ReportAgain(camera_id, None, follow_job=True))

    def _deliver(self, report: "_ReportAgain") -> "_ReportAgain | None":
        """상태를 보고한다. 성공하거나 버리면 None, 다시 보낼 만하면 report를 돌려준다."""
        if report.follow_job:
            # 시작 뒤 지금까지 상태가 바뀌었거나 관리자가 정지했을 수 있어 그때의 상태를 보낸다.
            current = self._service.current_status(report.camera_id)
            if current is None:
                logger.info(
                    "캠 작업이 이미 없어 상태 보고를 버림 cameraId=%s", report.camera_id
                )
                return None
            status, message = str(current), None
        else:
            status, message = "ERROR", report.error_message
        try:
            result, detail = self._reporter.send(
                report.camera_id, status, error_message=message
            )
        except Exception as error:
            result, detail = ReportResult.RETRY, f"예외: {error}"
        if result == ReportResult.OK:
            return None
        if result == ReportResult.REJECTED:
            logger.error(
                "상태 보고가 거절돼 버림 cameraId=%s status=%s: %s "
                "(캠이 삭제됐거나 ADMIN_INTERNAL_BASE_URL의 포트·/admin 경로가 틀렸을 수 있음)",
                report.camera_id,
                status,
                detail,
            )
            return None
        report.failures += 1
        if report.failures == 1 or report.failures % _LOG_EVERY == 0:
            logger.warning(
                "상태 보고 실패 cameraId=%s status=%s: %s (연속 %s번째)",
                report.camera_id,
                status,
                detail,
                report.failures,
            )
        return report


@dataclass
class _ReportAgain:
    """관리자 앱에 알려야 하는 캠 상태.

    follow_job이면 보낼 때마다 캠 작업의 지금 상태(RUNNING·PREPARING)를 보내고, 작업이 없으면
    버린다. 아니면 시작하지 못한 캠이라 ERROR와 error_message를 그대로 보낸다.
    """

    camera_id: int
    error_message: str | None
    follow_job: bool
    failures: int = 0


def _parse(item: Any) -> tuple[int, str, str] | None:
    if not isinstance(item, dict):
        return None
    camera_id = item.get("cameraId")
    source_type = item.get("sourceType")
    source_uri = item.get("sourceUri")
    if (
        not isinstance(camera_id, int)
        or isinstance(camera_id, bool)
        or not isinstance(source_type, str)
        or not isinstance(source_uri, str)
    ):
        return None
    return camera_id, source_type, source_uri
