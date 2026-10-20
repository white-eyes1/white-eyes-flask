import logging
import threading
from pathlib import Path

from white_eyes_flask.camera.camera_job import CameraJob, CameraStatus

logger = logging.getLogger(__name__)

DUMMY_PREFIX = "dummy/videos/"
DUMMY_SUFFIX = ".mp4"


class SourceNotFoundError(Exception):
    """DUMMY 영상 파일이 없거나 경로 규칙에 맞지 않음."""


class SourceTypeNotSupportedError(Exception):
    """1단계에서 구현하지 않은 sourceType(LIVE)."""


class CameraService:
    def __init__(self, media_root: str) -> None:
        self._media_root = Path(media_root)
        self._lock = threading.Lock()
        self._jobs: dict[int, CameraJob] = {}

    def start(self, camera_id: int, source_type: str, source_uri: str) -> CameraStatus:
        """캠 작업을 시작하고 상태를 돌려준다. 이미 돌고 있으면 그대로 돌려준다."""
        with self._lock:
            # 명세 처리 순서 1번: 이미 돌고 있으면 요청 내용과 상관없이 현재 상태
            running = self._jobs.get(camera_id)
            if running is not None and running.is_alive():
                return running.status

            if source_type == "LIVE":
                # LIVE 분기(FNC-S-007)는 2단계
                raise SourceTypeNotSupportedError(source_type)

            self._check_dummy_source(source_uri)
            status = (
                CameraStatus.RUNNING
                if self._has_cache(source_uri)
                else CameraStatus.PREPARING
            )
            job = CameraJob(camera_id, source_uri, status)
            job.start()
            self._jobs[camera_id] = job

        logger.info("캠 가동 cameraId=%s status=%s", camera_id, status)
        return status

    def job_count(self) -> int:
        with self._lock:
            return sum(1 for job in self._jobs.values() if job.is_alive())

    def shutdown(self) -> None:
        with self._lock:
            jobs = list(self._jobs.values())
            self._jobs.clear()
        for job in jobs:
            job.stop()

    def _check_dummy_source(self, source_uri: str) -> None:
        if (
            not source_uri.startswith(DUMMY_PREFIX)
            or not source_uri.endswith(DUMMY_SUFFIX)
            or ".." in source_uri
        ):
            raise SourceNotFoundError(source_uri)
        try:
            path = (self._media_root / source_uri).resolve()
            inside = path.is_relative_to(self._media_root.resolve())
            exists = inside and path.is_file()
        except (OSError, ValueError):
            raise SourceNotFoundError(source_uri) from None
        if not exists:
            # 심볼릭 링크로 공유 폴더 밖을 가리키는 경우도 여기서 막힌다
            raise SourceNotFoundError(source_uri)

    def _has_cache(self, source_uri: str) -> bool:
        # 더미 분석 캐시 확인은 FNC-S-009에서 구현한다. 그 전까지는 항상 없음.
        return False
