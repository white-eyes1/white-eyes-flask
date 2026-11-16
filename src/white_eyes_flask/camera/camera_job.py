"""캠 하나의 작업을 스레드 한 개로 돌리는 CameraJob과 그 상태값(CameraStatus).

작업 스레드는 (필요하면) 준비 단계 -> 재생 단계 순으로 실행하고, 어떻게 끝나든 마지막에
녹화기를 닫는다(FNC-S-001·007·008·009·010).
"""

import logging
import threading
import time
from enum import StrEnum

from white_eyes_flask.camera.player import NullPlayer, Player, PlayOutcome
from white_eyes_flask.camera.preparer import NullPreparer, PrepareOutcome, Preparer
from white_eyes_flask.recording.recorder import Recorder

logger = logging.getLogger(__name__)

# stop()이 작업 스레드가 끝나길 기다리는 최대 시간(초).
STOP_WAIT_SEC = 10


class CameraStatus(StrEnum):
    """캠 작업의 상태. 관리자 앱에 보고하는 값과 같은 문자열이다."""

    # 더미 영상을 분석해 캐시를 만드는 중
    PREPARING = "PREPARING"
    # 결과를 만들어 내보내는 중(라이브는 시작 즉시)
    RUNNING = "RUNNING"
    # 준비나 재생이 실패했거나 라이브 연결이 끊김. 정지하기 전까지 기록에 남는다
    ERROR = "ERROR"


class CameraJob:
    """캠마다 스레드 1개 + 정지 플래그 (FNC-S-001)."""

    # 상태는 락으로 지키고, 정지는 threading.Event 플래그로 알린다. 스레드는 daemon이라
    # 서버 종료를 막지 않는다. 단계(준비·재생)의 실제 일은 주입받은 preparer·player가 한다.

    def __init__(
        self,
        camera_id: int,
        source_uri: str,
        status: CameraStatus,
        preparer: Preparer | None = None,
        player: Player | None = None,
        recorder: Recorder | None = None,
    ) -> None:
        """작업을 만들기만 하고 스레드는 start()에서 시작한다.

        :param status: 시작 시점의 상태(캐시가 있거나 LIVE면 RUNNING, 아니면 PREPARING)
        :param preparer: 준비 단계 구현. 없으면 아무것도 안 하는 NullPreparer
        :param player: 재생 단계 구현. 없으면 정지될 때까지 기다리기만 하는 NullPlayer
        :param recorder: 라이브 녹화기. LIVE 작업에만 주고 그 밖에는 None
        """
        self.camera_id = camera_id
        self.source_uri = source_uri
        # 상태 확인(FNC-S-013)이 이전 작업의 결과 시각을 걸러내는 데 쓴다
        self.started_at = time.time()
        # 작업 스레드가 쓰고 CameraService.start()가 호출 스레드에서 읽는다(code-conventions 4-3).
        self._status_lock = threading.Lock()
        self._status = status
        # 주입이 없으면 Null 구현으로 채워, 실행 단계에서 None 검사를 하지 않아도 되게 한다.
        self._preparer = preparer or NullPreparer()
        self._player = player or NullPlayer()
        self._recorder = recorder
        # 정지 요청 신호. 준비·재생 루프가 이 플래그를 보고 스스로 끝난다.
        self._stop_flag = threading.Event()
        # daemon=True: 서버가 종료될 때 이 스레드 때문에 프로세스가 끝나지 않는 일을 막는다. 이름은 로그에 cam-{id}로 나온다.
        self._thread = threading.Thread(
            target=self._run, name=f"cam-{camera_id}", daemon=True
        )

    @property
    def status(self) -> CameraStatus:
        """현재 상태. 작업 스레드가 쓰는 값을 다른 스레드가 안전하게 읽는다."""
        with self._status_lock:
            return self._status

    @status.setter
    def status(self, value: CameraStatus) -> None:
        """상태를 바꾼다(작업 스레드와 라이브 재생기의 상태 콜백이 쓴다)."""
        with self._status_lock:
            self._status = value

    def start(self) -> None:
        """작업 스레드를 시작한다. 한 번만 부를 수 있다(스레드는 재시작되지 않는다)."""
        self._thread.start()

    def is_alive(self) -> bool:
        """작업 스레드가 아직 도는 중인가."""
        return self._thread.is_alive()

    def stop(self, timeout: float = STOP_WAIT_SEC) -> bool:
        """정지 플래그를 세우고 종료를 timeout초까지 기다린다. 끝났으면 True."""
        # 먼저 플래그를 세워 준비·재생 루프가 스스로 빠져나오게 한다.
        self._stop_flag.set()
        # 스레드를 강제로 끝낼 수는 없으므로 최대 timeout초만 기다린다.
        self._thread.join(timeout)
        # 아직 살아 있으면 False. 호출한 CameraService가 '정지 지연'으로 따로 기록한다.
        return not self._thread.is_alive()

    def _run(self) -> None:
        """작업 스레드 본체. 단계 실행이 어떻게 끝나든(예외 포함) finally에서 녹화기를 닫는다."""
        try:
            self._run_stages()
        finally:
            self._close_recorder()

    def _close_recorder(self) -> None:
        """작업이 어떻게 끝나든 녹화 파일을 닫아 등록한다(FNC-S-008 4~6번)."""
        # DUMMY 작업처럼 녹화기가 없으면 할 일이 없다.
        if self._recorder is None:
            return
        try:
            # 마지막 파일(1분 미만 포함)을 완성해 게시기에 넘기는 마무리. 몇 초 걸릴 수 있다.
            self._recorder.close()
        # 마무리 실패가 스레드 종료를 막지 않게 로그만 남긴다.
        except Exception:
            logger.exception("녹화 마무리 중 예외 cameraId=%s", self.camera_id)

    def _run_stages(self) -> None:
        """두 단계를 차례로 실행한다.

        1) PREPARING으로 시작했으면 준비(분석)를 하고, 2) 재생을 정지될 때까지 한다.
        각 단계가 돌려주는 Outcome으로 다음 동작을 정한다.
        """
        # 캐시가 있거나 LIVE인 작업은 RUNNING으로 시작하므로 이 단계를 건너뛴다.
        if self.status == CameraStatus.PREPARING:
            # 더미 분석 준비(FNC-S-009). 준비가 끝날 때까지 이 스레드가 맡는다.
            try:
                outcome = self._preparer.prepare(
                    self.camera_id, self.source_uri, self._stop_flag
                )
            # 준비 구현이 던진 예외는 준비 실패와 같게 취급한다.
            except Exception:
                logger.exception("더미 분석 준비 중 예외 cameraId=%s", self.camera_id)
                outcome = PrepareOutcome.FAILED
            if outcome == PrepareOutcome.FAILED:
                # 오류는 이미 보고했다. 스레드를 끝내서 다시 시작할 수 있게 한다.
                self.status = CameraStatus.ERROR
                return
            # 준비 중 정지 요청이 왔다. 오류가 아니므로 상태를 바꾸지 않고 끝낸다.
            if outcome == PrepareOutcome.STOPPED:
                return
            # 준비가 끝났으니 상태를 RUNNING으로 바꾸고 재생 단계로 넘어간다.
            self.status = CameraStatus.RUNNING
        # 캐시된 결과를 실제 시간에 맞춰 내보낸다(FNC-S-010). 정지될 때까지 이어진다.
        try:
            # 정지 플래그가 설 때까지 이 호출이 돌아오지 않는다.
            outcome = self._player.play(
                self.camera_id, self.source_uri, self._stop_flag
            )
        # 재생 구현이 던진 예외도 재생 실패로 본다.
        except Exception:
            logger.exception("더미 재생 중 예외 cameraId=%s", self.camera_id)
            outcome = PlayOutcome.FAILED
        # FAILED만 ERROR로 표시한다. STOPPED는 정상 종료라 상태를 그대로 둔다.
        if outcome == PlayOutcome.FAILED:
            self.status = CameraStatus.ERROR
