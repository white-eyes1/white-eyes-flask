"""FNC-S-012 분석 서버 재시작 복구 (GET /internal/v1/cameras/desired)"""

import http.client
import json
import logging
import threading
import urllib.error
from dataclasses import dataclass
from typing import Any, Protocol

from white_eyes_flask.camera.admin_client import AdminClient
from white_eyes_flask.camera.camera_service import (
    CameraStoppingError,
    SourceNotFoundError,
)
from white_eyes_flask.camera.status_reporter import ReportResult

logger = logging.getLogger(__name__)

# 가동 대상 목록을 못 받았거나 정지 때문에 못 시작한 캠을 다시 시도하는 간격(초).
RETRY_INTERVAL_SEC = 10
# 복구 스레드가 끝나길 기다리는 최대 시간(초).
SHUTDOWN_JOIN_SEC = 2.0
# 같은 실패가 이어질 때 처음과 이 횟수마다만 경고 로그를 남긴다.
_LOG_EVERY = 30
# 영상 파일이 없어 시작하지 못한 캠을 ERROR로 보고할 때 붙이는 사유 문구.
ERROR_SOURCE_NOT_FOUND = "영상 파일을 찾을 수 없음"


class DesiredClient:
    """가동 대상 캠 목록을 관리자 앱에서 받는다. 실패하면 None을 돌려주고 로그만 남긴다."""

    def __init__(self, base_url: str, token: str, timeout: float = 5.0) -> None:
        """조회 준비.

        :param base_url: 관리자 앱 내부 API의 기준 주소
        :param token: 내부 API 인증 토큰
        :param timeout: 요청 제한 시간(초)
        """
        self._admin = AdminClient(base_url, token)
        self._timeout = timeout
        # 연속 실패 횟수(로그를 줄이는 데만 쓴다).
        self._failures = 0

    def fetch(self) -> list[Any] | None:
        """관리자 앱에서 가동 대상 캠 목록(JSON 배열)을 받아 온다.

        성공하면 목록을, 실패하면 이유를 로그에 남기고 None을 돌려준다(예외를 올리지 않는다).
        """
        try:
            with self._admin.open(
                "GET", "/internal/v1/cameras/desired", timeout=self._timeout
            ) as response:
                # 본문은 JSON 배열이어야 한다. 읽기·파싱 실패는 아래 except에서 처리한다.
                payload = json.loads(response.read())
        # 서버가 응답은 했지만 2xx가 아니다.
        except urllib.error.HTTPError as error:
            error.close()
            # 401·403이면 토큰 설정 실수일 가능성이 커서 안내를 붙인다.
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
        # 배열이 아니면(객체 등) 쓸 수 없으니 실패로 본다.
        if not isinstance(payload, list):
            self._warn("가동 대상 조회 응답이 목록이 아님")
            return None
        # 실패가 이어지다가 처음 성공했을 때 복구 로그를 한 번 남긴다.
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
    """복구가 CameraService에서 쓰는 두 메서드만 적어 둔 규칙(Protocol)."""

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
        """복구기를 만든다(스레드는 start()에서 시작).

        :param client: fetch()로 가동 대상 목록을 주는 객체(DesiredClient)
        :param service: 캠을 시작하고 현재 상태를 알려 주는 객체(CameraService)
        :param reporter: send()로 상태를 보고하는 객체(StatusReporter)
        :param retry_interval: 다시 시도하는 간격(초)
        """
        self._client = client
        self._service = service
        self._reporter = reporter
        self._retry_interval = retry_interval
        # 복구 루프를 멈추는 신호(서버가 꺼질 때 stop()이 세운다).
        self._stop = threading.Event()
        # daemon 스레드라 서버 종료를 막지 않는다. 서버 시작을 기다리게 하지 않으려고 백그라운드에서 돌린다.
        self._thread = threading.Thread(
            target=self.run, name="restart-recovery", daemon=True
        )

    def start(self) -> None:
        """복구 스레드를 시작한다."""
        self._thread.start()

    def stop(self) -> None:
        """복구 루프를 멈추고 스레드가 끝나길 잠시 기다린다."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(SHUTDOWN_JOIN_SEC)

    def run(self) -> None:
        """목록을 받을 때까지, 그리고 받은 뒤 남은 일이 없어질 때까지 반복한다."""
        pending: list[Any] | None = None  # None이면 목록을 아직 못 받음
        while not self._stop.is_set():
            # 목록을 아직 못 받았으면 받아 본다. 한 번 받은 뒤에는 목록을 다시 받지 않고 남은 일만 이어서 한다.
            if pending is None:
                pending = self._client.fetch()
            if pending is not None:
                # recover는 아직 끝나지 않은 일만 돌려준다.
                pending = self.recover(pending)
                # 남은 일이 없으면 복구가 끝난 것이다.
                if not pending:
                    return
            # 다시 시도하기 전에 간격을 둔다. 정지 신호가 오면 바로 깨어난다.
            self._stop.wait(self._retry_interval)

    def recover(self, desired: list[Any]) -> list[Any]:
        """목록의 캠을 시작하고, 나중에 다시 해야 할 일(정지 중이라 못 시작, 상태 보고 실패)을 돌려준다."""
        retry: list[Any] = []
        # 한 캠이 실패해도 다음 캠을 계속 처리한다.
        for item in desired:
            # 서버가 꺼지는 중이면 남은 일을 하지 않는다.
            if self._stop.is_set():
                return []
            # 나중에 다시 해야 할 일이 있으면 돌려받아 모은다.
            again = self._process(item)
            if again is not None:
                retry.append(again)
        logger.info("재시작 복구 대상=%s건 다시 시도=%s건", len(desired), len(retry))
        return retry

    def _process(self, item: Any) -> "_ReportAgain | dict | None":
        """다시 시도할 일을 돌려준다. 끝났거나 다시 해도 소용없으면 None."""
        # 지난 차례에 상태 보고에 실패해 남은 항목이다. 시작은 이미 했으니 보고만 다시 한다.
        if isinstance(item, _ReportAgain):
            return self._deliver(item)
        # 새 항목이다. 형식을 확인해 (캠 번호, 종류, 주소)로 푼다.
        parsed = _parse(item)
        if parsed is None:
            logger.error("가동 대상 항목 형식이 올바르지 않아 건너뜀: %r", item)
            return None
        camera_id, source_type, source_uri = parsed
        try:
            # FNC-S-001과 같은 시작 경로를 쓴다. 이미 돌고 있으면 현재 상태만 돌려받고 중복으로 시작하지 않는다.
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
        # 예상 밖 예외도 이 캠만 건너뛰고 다음 캠을 계속 처리한다. 다시 시도하지는 않는다.
        except Exception:
            logger.exception("캠 복구 중 예외 cameraId=%s", camera_id)
            return None
        # 시작이 받아들여졌으니 그 캠의 지금 상태(PREPARING·RUNNING)를 관리자 앱에 알린다.
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
            # 시작하지 못한 캠은 ERROR와 저장해 둔 사유를 그대로 보낸다.
            status, message = "ERROR", report.error_message
        try:
            # StatusReporter.send는 로그를 남기지 않고 (결과, 이유)만 돌려준다. 로그는 아래에서 이 쪽이 남긴다.
            result, detail = self._reporter.send(
                report.camera_id, status, error_message=message
            )
        # 보고기 안에서 난 예외도 일시 오류로 보고 다시 시도하게 한다.
        except Exception as error:
            result, detail = ReportResult.RETRY, f"예외: {error}"
        # 보고에 성공했다. 더 할 일이 없다.
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
        # 일시 오류다. 실패 횟수를 올리고 다음 차례에 다시 보내도록 돌려준다. 로그는 처음과 30번째마다만 남긴다.
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

    # 보고할 캠 번호.
    camera_id: int
    # follow_job이 False일 때 ERROR와 함께 보낼 사유.
    error_message: str | None
    # True면 보낼 때마다 캠 작업의 지금 상태를 보낸다.
    follow_job: bool
    # 지금까지 연속으로 보고에 실패한 횟수(로그를 줄이는 데 쓴다).
    failures: int = 0


def _parse(item: Any) -> tuple[int, str, str] | None:
    """가동 대상 항목(JSON 객체)을 (캠 번호, sourceType, sourceUri)로 푼다. 형식이 틀리면 None."""
    if not isinstance(item, dict):
        return None
    camera_id = item.get("cameraId")
    source_type = item.get("sourceType")
    source_uri = item.get("sourceUri")
    if (
        not isinstance(camera_id, int)
        # bool은 int의 하위 타입이라 true·false가 캠 번호로 통과하지 않게 따로 거른다.
        or isinstance(camera_id, bool)
        or not isinstance(source_type, str)
        or not isinstance(source_uri, str)
    ):
        return None
    return camera_id, source_type, source_uri
