"""FNC-S-001·002 캠 작업 시작·정지 (CameraService)"""

import logging
import threading
from pathlib import Path

from white_eyes_flask.camera.camera_job import STOP_WAIT_SEC, CameraJob, CameraStatus
from white_eyes_flask.camera.player import NullPlayer, Player
from white_eyes_flask.camera.preparer import NullPreparer, Preparer
from white_eyes_flask.live.live_factory import LiveWorkFactory
from white_eyes_flask.live.live_player import LivePlayer
from white_eyes_flask.live.live_source import is_rtsp_uri, mask_uri
from white_eyes_flask.stream.stream_service import StreamService

logger = logging.getLogger(__name__)

# DUMMY의 sourceUri는 공유 폴더(media_root) 기준 dummy/videos/ 아래의 .mp4 파일만 허용한다.
DUMMY_PREFIX = "dummy/videos/"
DUMMY_SUFFIX = ".mp4"


class SourceNotFoundError(Exception):
    """DUMMY 영상 파일이 없거나 경로 규칙에 맞지 않음. LIVE는 rtsp:// 주소가 아님."""


class CameraStoppingError(Exception):
    """정지가 10초 안에 끝나지 않아 옛 작업 스레드가 아직 살아 있음."""


class CameraService:
    """캠 번호별 작업(CameraJob)의 시작·정지·상태 조회를 맡는 서비스.

    컨트롤러·재시작 복구·상태 확인이 함께 쓰므로 여러 스레드가 동시에 부른다. 잠금은 두 종류다.
    _lock은 아래 딕셔너리를 읽고 쓰는 짧은 순간만 지키고, 캠별 잠금은 같은 캠의 start·stop을
    한 줄로 세운다(정지를 기다리는 동안에도 다른 캠은 막히지 않는다).
    """

    def __init__(
        self,
        media_root: str,
        preparer: Preparer | None = None,
        player: Player | None = None,
        streams: StreamService | None = None,
        live: LiveWorkFactory | None = None,
    ) -> None:
        """서비스를 준비한다.

        :param media_root: 공유 폴더 루트(더미 영상의 기준 경로)
        :param preparer: 더미 분석 준비 구현. 없으면 Null(항상 PREPARING에서 대기)
        :param player: 더미 재생 구현. 없으면 정지될 때까지 기다리기만 하는 Null
        :param streams: 실시간 화면용 스트림 관리. 없으면 새로 만든다
        :param live: LIVE 작업의 재생기·녹화기 공장. 없으면 LIVE는 읽지 않는다
        """
        self._media_root = Path(media_root)
        # 없으면 LIVE 캠은 읽지 않고 정지될 때까지 기다리기만 한다(시험용 NullPlayer와 같다)
        self._live = live
        # 캠 작업이 있는 동안만 실시간 화면용 스트림이 열려 있다(FNC-S-011).
        self._streams = streams or StreamService()
        self._preparer = preparer or NullPreparer()
        self._player = player or NullPlayer()
        # _lock은 아래 세 딕셔너리를 읽고 쓰는 짧은 순간에만 쓴다.
        # 같은 캠의 시작·정지는 캠별 잠금(_camera_lock)으로 줄 세운다.
        self._lock = threading.Lock()
        # 지금 관리 중인 작업(캠 번호 -> 작업). 정지하면 빠진다.
        self._jobs: dict[int, CameraJob] = {}
        # 정지가 10초 안에 끝나지 않은 작업. 스레드가 끝날 때까지 같은 캠은 새로 시작하지 않는다.
        self._stopping: dict[int, CameraJob] = {}
        # 캠별 잠금. 한 번 만들면 지우지 않는다(shutdown이 이 목록으로 모든 캠을 돈다).
        self._camera_locks: dict[int, threading.Lock] = {}

    def start(self, camera_id: int, source_type: str, source_uri: str) -> CameraStatus:
        """캠 작업을 시작하고 상태를 돌려준다. 이미 돌고 있으면 그대로 돌려준다."""
        # 같은 캠의 start·stop이 동시에 들어오면 한 줄로 세운다.
        with self._camera_lock(camera_id):
            # 명세 처리 순서 1번: 이미 돌고 있으면 요청 내용과 상관없이 현재 상태
            running = self._get_job(camera_id)
            if running is not None and running.is_alive():
                return running.status

            # 같은 캠에 옛 스레드가 남아 있으면 새 스레드를 만들지 않는다(스레드 겹침 방지).
            # 정지는 10초만 기다리고 응답하므로 옛 스레드가 남아 있을 수 있다.
            if self._has_lingering_job(camera_id):
                raise CameraStoppingError(camera_id)

            # LIVE는 주소 형식만 검사하고 RUNNING으로 바로 시작한다. DUMMY는 영상 파일을 확인한다.
            if source_type == "LIVE":
                job = self._new_live_job(camera_id, source_uri)
            else:
                # 파일이 없거나 규칙에 어긋나면 SourceNotFoundError(컨트롤러가 400으로 바꾼다).
                self._check_dummy_source(source_uri)
                job = CameraJob(
                    camera_id,
                    source_uri,
                    # 캐시(분석 결과)가 이미 있으면 분석을 건너뛰고 RUNNING, 없으면 PREPARING에서 분석부터 한다.
                    (
                        CameraStatus.RUNNING
                        if self._has_cache(source_uri)
                        else CameraStatus.PREPARING
                    ),
                    self._preparer,
                    self._player,
                )
            # 응답에 돌려줄 시작 시점의 상태를 스레드를 시작하기 전에 읽어 둔다.
            status = job.status
            # 실시간 화면용 스트림을 먼저 열어, 작업이 첫 프레임을 낼 때 이미 받을 곳이 있게 한다.
            self._streams.open(camera_id)
            # 이제 작업 스레드가 돌기 시작한다.
            job.start()
            with self._lock:
                # 짧은 _lock 안에서만 등록한다. 바깥의 캠별 잠금 덕에 같은 캠의 다른 start·stop은 이 사이에 끼어들지 못한다.
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
            # 돌던 작업도, 정지가 밀린 작업도 없다. 이미 멈춘 상태라 할 일이 없다(멱등).
            if job is None:
                return

            # 작업이 없어진 것으로 본다: 이후 /live·/snapshot은 404, 보던 MJPEG 연결은 끝난다.
            self._streams.close(camera_id)
            # 녹화 파일 마무리(FNC-S-008 4~6번)는 작업 스레드가 끝날 때 CameraJob이 한다.
            # 공유 폴더로 옮기고 등록하는 재시도는 게시기 스레드가 맡아 이 정지를 막지 않는다.
            # 준비 취소(FNC-S-009)는 정지 플래그로 준비 루프가 스스로 멈추고 임시 파일을 지운다.
            # 전송 대기 중인 결과는 건드리지 않는다(전송 작업이 보냄).
            # 정지 플래그를 세우고 최대 10초 기다린다. 이 동안 이 캠의 잠금을 쥐고 있다.
            stopped = job.stop(STOP_WAIT_SEC)
            with self._lock:
                # 끝났으면 지연 기록을 지우고, 아니면 기록해 두어 start가 새 스레드를 만들지 못하게 한다.
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
        """캠 번호의 전용 잠금을 돌려준다(없으면 만든다). 딕셔너리 접근 자체는 _lock으로 지킨다."""
        with self._lock:
            return self._camera_locks.setdefault(camera_id, threading.Lock())

    def _get_job(self, camera_id: int) -> CameraJob | None:
        """관리 중인 작업을 돌려준다. 없으면 None."""
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
            # 정지가 밀렸던 스레드가 그사이 끝났다. 기록을 지워 새 시작을 허용한다.
            del self._stopping[camera_id]
            return False

    def current_status(self, camera_id: int) -> CameraStatus | None:
        """작업 스레드가 살아 있는 캠의 지금 상태. 작업이 없거나 끝났으면 None."""
        job = self._get_job(camera_id)
        if job is None or not job.is_alive():
            return None
        return job.status

    def job_count(self) -> int:
        """스레드가 살아 있는 작업의 수."""
        with self._lock:
            return sum(1 for job in self._jobs.values() if job.is_alive())

    def camera_states(self) -> list[tuple[int, CameraStatus, float]]:
        """상태 확인(FNC-S-013)용 (캠 번호, 상태, 시작 시각 epoch초). 캠 번호 순.

        정지하기 전까지는 스레드가 오류로 끝난 캠(ERROR)도 포함한다.
        """
        with self._lock:
            # 딕셔너리는 잠금 안에서 복사·정렬만 하고, 상태를 읽는 일은 잠금 밖에서 한다.
            jobs = sorted(self._jobs.values(), key=lambda job: job.camera_id)
        return [(job.camera_id, job.status, job.started_at) for job in jobs]

    def shutdown(self) -> None:
        """모든 캠을 stop()으로 멈춘다. 진행 중인 정지와 남은 옛 스레드도 캠별 잠금으로 기다린다."""
        with self._lock:
            # 시작 요청이 한 번이라도 온 모든 캠. 복사해 두고 돌아야 stop이 딕셔너리를 바꿔도 안전하다.
            camera_ids = list(self._camera_locks)
        for camera_id in camera_ids:
            # 캠별 잠금을 거치므로 진행 중인 정지·시작이 끝난 뒤에 멈춘다.
            self.stop(camera_id)

    def _new_live_job(self, camera_id: int, source_uri: str) -> CameraJob:
        """LIVE 캠 작업(FNC-S-007). 명세대로 RUNNING으로 시작하고, 이후 상태는 라이브 재생기가 바꾼다."""
        # LIVE는 rtsp:// 주소만 받는다. 예외에는 mask_uri로 가린 주소를 담는다.
        if not is_rtsp_uri(source_uri):
            raise SourceNotFoundError(mask_uri(source_uri))
        # 공장이 없는 환경(시험 등)에서는 읽지도 녹화하지도 않는 작업이 된다.
        if self._live is None:
            player, recorder = NullPlayer(), None
        else:
            # 공장이 이 캠에 맞는 재생기와 녹화기를 한 쌍으로 만들어 준다.
            player, recorder = self._live.create(camera_id)
        # LIVE는 명세대로 PREPARING 없이 RUNNING으로 시작한다.
        job = CameraJob(
            camera_id,
            source_uri,
            CameraStatus.RUNNING,
            self._preparer,
            player,
            recorder,
        )
        if isinstance(player, LivePlayer):
            # 연결이 끊겨 ERROR가 돼도 작업(스레드)은 계속 돈다. 그 상태를 작업에도 반영한다.
            player.on_status = lambda status: setattr(job, "status", status)
        return job

    def _check_dummy_source(self, source_uri: str) -> None:
        """DUMMY 영상 경로가 허용 규칙을 지키고 실제 파일이 있는지 검사한다. 아니면 SourceNotFoundError."""
        # 1차(문자열 검사): 접두사·확장자 규칙과 상위 경로(..) 시도를 막는다.
        if (
            not source_uri.startswith(DUMMY_PREFIX)
            or not source_uri.endswith(DUMMY_SUFFIX)
            or ".." in source_uri
        ):
            raise SourceNotFoundError(source_uri)
        try:
            # 2차(실제 경로 검사): 링크를 풀어 낸 실제 경로가 공유 폴더 안의 파일이어야 한다.
            path = (self._media_root / source_uri).resolve()
            inside = path.is_relative_to(self._media_root.resolve())
            exists = inside and path.is_file()
        except (OSError, ValueError):
            # 경로가 잘못돼 풀리지 않는 경우도 없는 파일로 본다. 원인 예외는 숨긴다.
            raise SourceNotFoundError(source_uri) from None
        if not exists:
            # 심볼릭 링크로 공유 폴더 밖을 가리키는 경우도 여기서 막힌다
            raise SourceNotFoundError(source_uri)

    def _has_cache(self, source_uri: str) -> bool:
        # 캐시·좌표 파일이 모두 있으면 준비 없이 바로 RUNNING(FNC-S-009)
        """더미 분석 결과(캐시)가 이미 있나."""
        return self._preparer.has_cache(source_uri)
