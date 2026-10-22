"""FNC-S-001·002 캠 작업 시작·정지 (CameraService)"""

import logging
import threading
from pathlib import Path

from white_eyes_flask.camera.camera_job import STOP_WAIT_SEC, CameraJob, CameraStatus
from white_eyes_flask.camera.player import NullPlayer, Player
from white_eyes_flask.camera.preparer import NullPreparer, Preparer

logger = logging.getLogger(__name__)

DUMMY_PREFIX = "dummy/videos/"
DUMMY_SUFFIX = ".mp4"


class SourceNotFoundError(Exception):
    """DUMMY 영상 파일이 없거나 경로 규칙에 맞지 않음."""


class SourceTypeNotSupportedError(Exception):
    """1단계에서 구현하지 않은 sourceType(LIVE)."""


class CameraStoppingError(Exception):
    """정지가 10초 안에 끝나지 않아 옛 작업 스레드가 아직 살아 있음."""


class CameraService:
    def __init__(
        self,
        media_root: str,
        preparer: Preparer | None = None,
        player: Player | None = None,
    ) -> None:
        self._media_root = Path(media_root)
        self._preparer = preparer or NullPreparer()
        self._player = player or NullPlayer()
        # _lock은 아래 세 딕셔너리를 읽고 쓰는 짧은 순간에만 쓴다.
        # 같은 캠의 시작·정지는 캠별 잠금(_camera_lock)으로 줄 세운다.
        self._lock = threading.Lock()
        self._jobs: dict[int, CameraJob] = {}
        # 정지가 10초 안에 끝나지 않은 작업. 스레드가 끝날 때까지 같은 캠은 새로 시작하지 않는다.
        self._stopping: dict[int, CameraJob] = {}
        self._camera_locks: dict[int, threading.Lock] = {}

    def start(self, camera_id: int, source_type: str, source_uri: str) -> CameraStatus:
        """캠 작업을 시작하고 상태를 돌려준다. 이미 돌고 있으면 그대로 돌려준다."""
        with self._camera_lock(camera_id):
            # 명세 처리 순서 1번: 이미 돌고 있으면 요청 내용과 상관없이 현재 상태
            running = self._get_job(camera_id)
            if running is not None and running.is_alive():
                return running.status

            # 같은 캠에 옛 스레드가 남아 있으면 새 스레드를 만들지 않는다(스레드 겹침 방지).
            if self._has_lingering_job(camera_id):
                raise CameraStoppingError(camera_id)

            if source_type == "LIVE":
                # LIVE 분기(FNC-S-007)는 2단계
                raise SourceTypeNotSupportedError(source_type)

            self._check_dummy_source(source_uri)
            status = (
                CameraStatus.RUNNING
                if self._has_cache(source_uri)
                else CameraStatus.PREPARING
            )
            job = CameraJob(camera_id, source_uri, status, self._preparer, self._player)
            job.start()
            with self._lock:
                self._jobs[camera_id] = job

        logger.info("캠 가동 cameraId=%s status=%s", camera_id, status)
        return status

    def stop(self, camera_id: int) -> None:
        """캠 작업을 멈춘다. 작업이 없어도 오류 없이 끝난다(멱등)."""
        # 캠별 잠금을 쥔 채 종료를 기다린다. 같은 캠의 새 시작이 옛 스레드가
        # 끝나기 전에 들어와 스레드가 둘이 되는 것을 막는다. 다른 캠은 막히지 않는다.
        with self._camera_lock(camera_id):
            with self._lock:
                # 옛 작업이 아직 남아 있으면 정지를 다시 요청해 그 종료를 한 번 더 기다린다.
                job = self._jobs.pop(camera_id, None) or self._stopping.get(camera_id)
            if job is None:
                return

            # 녹화 파일 마무리(FNC-S-008 4~6번)는 해당 이슈에서 채운다.
            # 준비 취소(FNC-S-009)는 정지 플래그로 준비 루프가 스스로 멈추고 임시 파일을 지운다.
            # 전송 대기 중인 결과는 건드리지 않는다(전송 작업이 보냄).
            stopped = job.stop(STOP_WAIT_SEC)
            with self._lock:
                if stopped:
                    self._stopping.pop(camera_id, None)
                else:
                    self._stopping[camera_id] = job

        if stopped:
            logger.info("캠 정지 cameraId=%s", camera_id)
        else:
            # 10초 안에 안 끝나도 응답은 200. 이 스레드는 정지 플래그가 선 채 스스로 끝나고,
            # 그때까지 같은 캠의 새 시작은 CameraStoppingError로 막는다.
            logger.warning(
                "캠 정지 지연 cameraId=%s, %s초 안에 끝나지 않음",
                camera_id,
                STOP_WAIT_SEC,
            )

    def _camera_lock(self, camera_id: int) -> threading.Lock:
        with self._lock:
            return self._camera_locks.setdefault(camera_id, threading.Lock())

    def _get_job(self, camera_id: int) -> CameraJob | None:
        with self._lock:
            return self._jobs.get(camera_id)

    def _has_lingering_job(self, camera_id: int) -> bool:
        """정지가 끝나지 않은 옛 스레드가 아직 살아 있나. 끝났으면 기록을 지운다."""
        with self._lock:
            job = self._stopping.get(camera_id)
            if job is None:
                return False
            if job.is_alive():
                return True
            del self._stopping[camera_id]
            return False

    def current_status(self, camera_id: int) -> CameraStatus | None:
        """작업 스레드가 살아 있는 캠의 지금 상태. 작업이 없거나 끝났으면 None."""
        job = self._get_job(camera_id)
        if job is None or not job.is_alive():
            return None
        return job.status

    def job_count(self) -> int:
        with self._lock:
            return sum(1 for job in self._jobs.values() if job.is_alive())

    def shutdown(self) -> None:
        """모든 캠을 stop()으로 멈춘다. 진행 중인 정지와 남은 옛 스레드도 캠별 잠금으로 기다린다."""
        with self._lock:
            camera_ids = list(self._camera_locks)
        for camera_id in camera_ids:
            self.stop(camera_id)

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
        # 캐시·좌표 파일이 모두 있으면 준비 없이 바로 RUNNING(FNC-S-009)
        return self._preparer.has_cache(source_uri)
