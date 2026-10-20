import logging
import threading
from enum import StrEnum

from white_eyes_flask.camera.preparer import NullPreparer, PrepareOutcome, Preparer

logger = logging.getLogger(__name__)

STOP_WAIT_SEC = 10


class CameraStatus(StrEnum):
    PREPARING = "PREPARING"
    RUNNING = "RUNNING"
    ERROR = "ERROR"


class CameraJob:
    """캠마다 스레드 1개 + 정지 플래그 (FNC-S-001)."""

    def __init__(
        self,
        camera_id: int,
        source_uri: str,
        status: CameraStatus,
        preparer: Preparer | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.source_uri = source_uri
        self.status = status
        self._preparer = preparer or NullPreparer()
        self._stop_flag = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"cam-{camera_id}", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def stop(self, timeout: float = STOP_WAIT_SEC) -> bool:
        """정지 플래그를 세우고 종료를 timeout초까지 기다린다. 끝났으면 True."""
        self._stop_flag.set()
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _run(self) -> None:
        if self.status == CameraStatus.PREPARING:
            # 더미 분석 준비(FNC-S-009). 준비가 끝날 때까지 이 스레드가 맡는다.
            try:
                outcome = self._preparer.prepare(
                    self.camera_id, self.source_uri, self._stop_flag
                )
            except Exception:
                logger.exception("더미 분석 준비 중 예외 cameraId=%s", self.camera_id)
                outcome = PrepareOutcome.FAILED
            if outcome == PrepareOutcome.FAILED:
                # 오류는 이미 보고했다. 스레드를 끝내서 다시 시작할 수 있게 한다.
                self.status = CameraStatus.ERROR
                return
            if outcome == PrepareOutcome.STOPPED:
                return
            self.status = CameraStatus.RUNNING
        # 재생(FNC-S-010) 본문은 해당 이슈에서 채운다.
        # 지금은 정지 플래그가 설정될 때까지 대기한다.
        self._stop_flag.wait()
