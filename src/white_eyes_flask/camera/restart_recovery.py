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
    SourceTypeNotSupportedError,
)

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


class RestartRecovery:
    """서버가 켜질 때 가동 대상 캠을 받아 FNC-S-001과 같은 방식(CameraService.start)으로 시작한다.

    관리자 앱에 닿지 못하면 retry_interval마다 계속 다시 시도한다. 서버 시작을 막지 않게
    백그라운드 스레드에서 돈다. 이미 가동 중인 캠은 CameraService.start가 현재 상태만
    돌려주므로 중복으로 시작되지 않는다. 한 캠이 실패해도 다음 캠을 계속 처리한다.
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
        started = 0
        for item in desired:
            if self._stop.is_set():
                return []
            outcome = self._process(item)
            if outcome is True:
                started += 1
            elif outcome is not False:
                retry.append(outcome)
        logger.info(
            "재시작 복구 대상=%s건 시작=%s건 다시 시도=%s건",
            len(desired),
            started,
            len(retry),
        )
        return retry

    def _process(self, item: Any) -> "bool | _ReportAgain | dict":
        """True=시작함, False=건너뜀(다시 해도 소용없음), 그 밖=다시 시도할 일."""
        if isinstance(item, _ReportAgain):
            return True if self._report(item) else item
        parsed = _parse(item)
        if parsed is None:
            logger.error("가동 대상 항목 형식이 올바르지 않아 건너뜀: %r", item)
            return False
        camera_id, source_type, source_uri = parsed
        try:
            status = self._service.start(camera_id, source_type, source_uri)
        except SourceTypeNotSupportedError:
            # LIVE는 라이브 분석(FNC-S-007) 구현 전까지 시작할 수 없다. 서버는 계속 뜬다.
            logger.warning(
                "복구할 수 없는 영상 종류라 건너뜀 cameraId=%s sourceType=%s",
                camera_id,
                source_type,
            )
            return False
        except SourceNotFoundError:
            logger.error("복구할 영상 파일이 없음 cameraId=%s", camera_id)
            report = _ReportAgain(camera_id, "ERROR", ERROR_SOURCE_NOT_FOUND)
            return False if self._report(report) else report
        except CameraStoppingError:
            # 정지가 끝나면 시작할 수 있다. 같은 항목을 다음 차례에 다시 시도한다.
            logger.warning("정지가 끝나지 않아 나중에 다시 시도 cameraId=%s", camera_id)
            return item
        except Exception:
            logger.exception("캠 복구 중 예외 cameraId=%s", camera_id)
            return False
        report = _ReportAgain(camera_id, str(status), None)
        return True if self._report(report) else report

    def _report(self, report: "_ReportAgain") -> bool:
        try:
            return bool(
                self._reporter.report(
                    report.camera_id, report.status, error_message=report.error_message
                )
            )
        except Exception:
            logger.warning("상태 보고 중 오류 cameraId=%s", report.camera_id)
            return False


@dataclass(frozen=True)
class _ReportAgain:
    """캠은 이미 시작했는데 상태 보고가 실패해 다시 보내야 하는 일."""

    camera_id: int
    status: str
    error_message: str | None


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
