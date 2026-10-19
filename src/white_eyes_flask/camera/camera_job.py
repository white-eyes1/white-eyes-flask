import logging
import threading
from enum import StrEnum

logger = logging.getLogger(__name__)

STOP_WAIT_SEC = 10


class CameraStatus(StrEnum):
    PREPARING = "PREPARING"
    RUNNING = "RUNNING"


class CameraJob:
    """캠마다 스레드 1개 + 정지 플래그 (FNC-S-001)."""

    def __init__(self, camera_id: int, source_uri: str, status: CameraStatus) -> None:
        self.camera_id = camera_id
        self.source_uri = source_uri
        self.status = status
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
        # 준비(FNC-S-009)·재생(FNC-S-010) 본문은 해당 이슈에서 채운다.
        # 지금은 정지 플래그가 설정될 때까지 대기한다.
        self._stop_flag.wait()
