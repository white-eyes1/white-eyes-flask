"""FNC-S-007 라이브 분석·재연결 (LivePlayer)

캠 작업 스레드가 RTSP를 읽고(모든 프레임을 녹화기에 넘김), 분석은 따로 도는 스레드가
맡는다. 두 스레드 사이에는 최신 프레임 한 장만 남기는 자리가 있어서, 분석이 프레임 속도를
못 따라가면 오래된 프레임은 버려지고 늘 최신 프레임을 분석한다.
"""

import logging
import math
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from white_eyes_flask.analysis.detector import Detector
from white_eyes_flask.analysis.detector_provider import DetectorUnavailableError
from white_eyes_flask.analysis.result import TrackedBox
from white_eyes_flask.camera.camera_job import CameraStatus
from white_eyes_flask.camera.player import PlayOutcome
from white_eyes_flask.camera.result_queue import ResultSink
from white_eyes_flask.camera.status_reporter import ReportResult
from white_eyes_flask.live.live_analyzer import LiveAnalyzer
from white_eyes_flask.live.live_source import mask_uri, open_capture
from white_eyes_flask.recording.gap_recorder import GapSplittingRecorder
from white_eyes_flask.recording.recorder import (
    LiveRecorder,
    RecorderFailed,
    RecorderUnavailable,
)
from white_eyes_flask.stream.stream_service import LiveFrame, StreamService

logger = logging.getLogger(__name__)

# 명세(FNC-S-007 4·5번, conventions.md LIVE_RECONNECT_FAIL_SEC)
LIVE_RECONNECT_FAIL_SEC = 30
# 연결이 끊겼을 때 다시 연결을 시도하는 간격(정상 상태일 때)
RECONNECT_INTERVAL_SEC = 2
# 이미 ERROR로 보고한 뒤에는 서버·네트워크 부담을 줄이려고 더 느리게 시도한다
ERROR_RECONNECT_INTERVAL_SEC = 30
# 못 보낸 상태 보고를 다시 보내는 간격(보고가 막혀도 읽기를 오래 붙잡지 않게 읽기와 같은 스레드에서 가끔만)
REPORT_RETRY_SEC = 5
# 종료할 때 분석 스레드가 끝나기를 기다리는 최대 시간(초)
ANALYSIS_JOIN_SEC = 2.0
# 분석 스레드가 새 프레임을 기다리는 한 번의 최대 시간. 프레임이 없어도 이 주기로 깨어나 지난 초 결과를 내보낸다
ANALYSIS_POLL_SEC = 0.5
_LOG_EVERY = 30  # 같은 실패가 이어질 때 로그는 처음과 이 횟수마다만

# 관리자 서버에 ERROR 상태로 보고할 때 쓰는 사유 문구
ERROR_STREAM = "스트림 연결 실패"
ERROR_MODEL = "분석 모델을 불러올 수 없음"
ERROR_RECORDER_START = "녹화를 시작할 수 없음"
ERROR_RECORDER = "녹화 중 오류가 발생함"
ERROR_UNEXPECTED = "분석 중 오류가 발생함"


class _LatestFrame:
    """분석 스레드에 넘길 최신 프레임 한 장. 새 프레임이 오면 아직 안 가져간 프레임은 버린다."""

    def __init__(self) -> None:
        # 읽기 스레드(put)와 분석 스레드(take)가 값을 주고받고, 기다림·깨움에 쓰는 조건 변수
        self._cond = threading.Condition()
        # 아직 분석 스레드가 가져가지 않은 (프레임, 읽은 시각). 없으면 None
        self._frame: tuple[np.ndarray, float] | None = None
        # 스트림이 끊겼다는 알림이 아직 전달되지 않았는지
        self._reset = False
        # 더 넣을 프레임이 없다(작업 종료)는 표시
        self._closed = False
        # 분석이 늦어 가져가기 전에 덮어써져 버려진 프레임 수(종료 때 로그용)
        self.dropped = 0

    def put(self, frame: np.ndarray, at: float) -> None:
        """읽기 스레드가 최신 프레임을 넣는다. 아직 안 가져간 프레임이 있으면 덮어쓰고 버린 수를 센다."""
        with self._cond:
            if self._frame is not None:
                self.dropped += 1
            self._frame = (frame, at)
            # 기다리는 분석 스레드를 깨운다
            self._cond.notify()

    def put_reset(self) -> None:
        """스트림이 끊겼다. 끊긴 연결의 프레임은 버리고 분석 스레드에 알린다."""
        with self._cond:
            self._frame = None
            self._reset = True
            self._cond.notify()

    def close(self) -> None:
        """작업이 끝났음을 분석 스레드에 알려 루프를 빠져나가게 한다."""
        with self._cond:
            self._closed = True
            self._cond.notify()

    def take(
        self, timeout: float
    ) -> tuple[bool, tuple[np.ndarray, float] | None, bool]:
        """(끊김 알림, 프레임, 닫힘). 아무것도 없으면 timeout초 기다린다."""
        with self._cond:
            # 프레임·끊김·종료 중 하나라도 생기면 깨어난다. 아무 일도 없으면 timeout 뒤에 빈손으로 돌아간다
            self._cond.wait_for(
                lambda: self._frame is not None or self._reset or self._closed,
                timeout=timeout,
            )
            # 꺼내면서 비워서 같은 프레임·알림이 두 번 전달되지 않게 한다
            reset, self._reset = self._reset, False
            frame, self._frame = self._frame, None
            return reset, frame, self._closed


class LivePlayer:
    """LIVE 캠 하나의 읽기·재연결·분석 연결. 캠 작업마다 하나 만들어 CameraJob에 넘긴다.

    상태: 시작할 때 RUNNING. 연결이 LIVE_RECONNECT_FAIL_SEC(기본 30초)를 넘게 끊겨 있으면 ERROR를
    보고하고 on_status로 작업 상태도 ERROR로 바꾼다. 그 뒤에도 30초 간격으로 계속 다시
    연결하고, 프레임을 다시 읽으면 RUNNING으로 돌아와 보고한다.
    녹화기를 못 쓰거나 녹화 중 죽으면(RecorderUnavailable·RecorderFailed) ERROR를 보고하고 끝낸다.
    """

    def __init__(
        self,
        detector_provider: Callable[[], Detector],
        sink: ResultSink,
        reporter: Any,
        streams: StreamService,
        recorder: LiveRecorder | GapSplittingRecorder | None,
        *,
        fps: int,
        conf_thres: float,
        edge_margin: float,
        capture_factory: Callable[[str], Any] = open_capture,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
        reconnect_fail_sec: float | None = None,
    ) -> None:
        # 설정값(LIVE_RECONNECT_FAIL_SEC). 주지 않으면 명세 기본값
        self._reconnect_fail_sec = (
            LIVE_RECONNECT_FAIL_SEC
            if reconnect_fail_sec is None
            else reconnect_fail_sec
        )
        self._detector_provider = detector_provider
        self._sink = sink
        self._reporter = reporter
        self._streams = streams
        self._recorder = recorder
        self._fps = fps
        self._conf_thres = conf_thres
        self._edge_margin = edge_margin
        self._capture_factory = capture_factory
        # clock은 경과 시간 측정용(되돌아가지 않는 monotonic), wall은 프레임에 찍는 실제 시각(epoch 초).
        # 시험에서 가짜 시계를 넣을 수 있게 주입받는다
        self._clock = clock
        self._wall = wall
        # CameraService가 작업 상태를 바꾸는 함수를 넣는다(작업이 살아 있는 채 ERROR가 되므로)
        self.on_status: Callable[[CameraStatus], None] = lambda status: None
        # 아직 관리자 서버에 전달하지 못한 상태 보고((상태, 사유)). 가장 최근 것 하나만 둔다
        self._pending_report: tuple[str, str | None] | None = None
        # 이 시각(clock 기준) 이전에는 상태 보고를 다시 보내지 않는다
        self._next_report_try = 0.0
        # play()가 끝나면 켜지는 표시
        self._finished = (
            threading.Event()
        )  # play()가 끝났다. 남은 분석 스레드는 더 넘기지 않는다

    def play(
        self, camera_id: int, source_uri: str, stop: threading.Event
    ) -> PlayOutcome:
        """캠 작업 스레드에서 실행된다. 분석 스레드를 띄우고 RTSP 읽기 루프를 돌리다가 stop이 서거나 치명 오류가 나면 끝난다.

        반환값 PlayOutcome이 STOPPED면 정상 정지, FAILED면 ERROR를 보고하고 끝난 것이다.
        """
        # 탐지 모델은 첫 사용 때 불러오므로 여기서 실패할 수 있다. 모델이 없으면 분석이 불가능하니 바로 ERROR로 끝낸다
        try:
            detector = self._detector_provider()
        except DetectorUnavailableError:
            logger.exception("탐지 모델 사용 불가 cameraId=%s", camera_id)
            return self._fail(camera_id, ERROR_MODEL)
        except Exception:
            logger.exception("탐지 모델 준비 중 예외 cameraId=%s", camera_id)
            return self._fail(camera_id, ERROR_UNEXPECTED)

        analyzer = LiveAnalyzer(
            camera_id,
            detector,
            self._sink,
            fps=self._fps,
            conf_thres=self._conf_thres,
            edge_margin=self._edge_margin,
        )
        # 읽기 스레드(이 스레드)와 분석 스레드를 잇는 최신 프레임 보관함.
        # 분석(탐지)이 느려도 읽기가 막히지 않게 하려고 둘을 분리한다
        latest = _LatestFrame()
        worker = threading.Thread(
            target=self._analysis_loop,
            args=(camera_id, analyzer, latest),
            name=f"cam-{camera_id}-analysis",
            # 서버가 종료될 때 분석 스레드가 프로세스 종료를 막지 않게 한다
            daemon=True,
        )
        worker.start()
        try:
            return self._read_loop(camera_id, source_uri, stop, latest)
        # 아래 예외들은 읽기를 더 이어갈 수 없는 치명 오류라 상태를 ERROR로 보고하고 작업을 끝낸다
        except RecorderUnavailable:
            logger.exception("녹화를 시작할 수 없음 cameraId=%s", camera_id)
            return self._fail(camera_id, ERROR_RECORDER_START)
        except RecorderFailed:
            logger.exception("녹화가 중단됨 cameraId=%s", camera_id)
            return self._fail(camera_id, ERROR_RECORDER)
        except Exception:
            logger.exception("라이브 읽기 중 예외 cameraId=%s", camera_id)
            return self._fail(camera_id, ERROR_UNEXPECTED)
        finally:
            # 분석 스레드에 종료를 알리고 잠시 기다린다(남은 초 결과를 내보낼 시간을 준다)
            latest.close()
            worker.join(ANALYSIS_JOIN_SEC)
            # 제때 안 끝난 분석 스레드가 뒤늦게 녹화기·화면에 넘기지 못하도록 표시한다
            self._finished.set()
            if worker.is_alive():
                logger.warning(
                    "라이브 분석 스레드가 제때 끝나지 않음 cameraId=%s", camera_id
                )

    # ---- 읽기·재연결 (캠 작업 스레드) ----

    def _read_loop(
        self,
        camera_id: int,
        source_uri: str,
        stop: threading.Event,
        latest: _LatestFrame,
    ) -> PlayOutcome:
        """RTSP에 연결해 프레임을 읽는 루프. 모든 프레임을 녹화기에 넘기고, 초당 fps장만 분석용으로 넘긴다.

        끊기면 간격을 두고 다시 연결하며, 끊긴 시간이 reconnect_fail_sec를 넘으면 ERROR를 보고한다.
        """
        # 로그에는 계정·비밀번호를 가린 주소만 남긴다
        shown = mask_uri(source_uri)
        capture = None
        failed = False  # ERROR로 보고한 상태
        last_good = self._clock()  # 마지막으로 프레임을 읽은 때(처음엔 시작한 때)
        # 연속 실패 횟수(로그를 줄이고 재시도 간격을 정하는 데 쓴다)
        attempts = 0
        # 마지막으로 분석용으로 넘긴 슬롯(실제 시각 x fps)
        last_slot = -1
        try:
            while not stop.is_set():
                # 이전에 못 보낸 상태 보고가 있으면 한 번 더 보낸다
                self._retry_report(camera_id)
                # 연결이 없으면 새로 연결한다
                if capture is None:
                    capture = self._connect(source_uri)
                    if capture is not None:
                        if stop.is_set():
                            break  # 연결하는 동안 정지가 들어왔다. 읽지 않고 나간다
                        logger.info(
                            "스트림 연결됨 cameraId=%s uri=%s", camera_id, shown
                        )
                # 연결된 상태면 프레임 하나를 읽는다
                if capture is not None:
                    try:
                        ok, frame = capture.read()
                    except Exception:
                        # cv2.error 등: 읽기 실패와 같은 경로로 연결을 닫고 다시 연결한다
                        if attempts == 0 or attempts % _LOG_EVERY == 0:
                            logger.exception(
                                "프레임을 읽다 예외 cameraId=%s", camera_id
                            )
                        ok, frame = False, None
                    # 정상 프레임(비어 있지 않음): 읽은 시각을 찍고 실패 카운터를 초기화한다
                    if ok and frame is not None and getattr(frame, "size", 1) > 0:
                        at = self._wall()
                        last_good = self._clock()
                        attempts = 0
                        # ERROR였다가 프레임이 다시 들어오면 RUNNING으로 복구를 보고한다
                        if failed:
                            failed = False
                            self._set_status(camera_id, "RUNNING", None)
                        # 녹화는 모든 프레임을 넘긴다(분석과 달리 솎지 않는다)
                        if self._recorder is not None:
                            self._recorder.write_frame(frame, at)
                        # 분석은 초당 fps장이면 충분하므로, 같은 슬롯의 나머지 프레임은 분석 보관함에 넣지 않는다
                        slot = math.floor(at * self._fps)
                        if slot != last_slot:
                            last_slot = slot
                            latest.put(frame, at)
                        continue
                    # 읽기 실패·빈 프레임: 연결을 닫고 분석 쪽에 끊김을 알린 뒤 아래에서 재연결한다
                    self._release(capture)
                    capture = None
                    latest.put_reset()  # 끊긴 연결의 추적은 여기서 끝낸다
                # 연결하지 못했거나 읽기에 실패했다: 간격을 두고 다시 연결한다
                attempts += 1
                if attempts == 1 or attempts % _LOG_EVERY == 0:
                    logger.warning(
                        "스트림 연결 실패 cameraId=%s uri=%s (연속 %s번째)",
                        camera_id,
                        shown,
                        attempts,
                    )
                # 마지막으로 프레임을 읽은 뒤 설정 시간이 지나면 ERROR로 한 번 보고한다
                if not failed and self._clock() - last_good >= self._reconnect_fail_sec:
                    failed = True
                    self._set_status(camera_id, "ERROR", ERROR_STREAM)
                # 재시도 사이에 대기한다. time.sleep 대신 stop.wait를 써서 정지 요청이 오면 즉시 깨어난다
                stop.wait(
                    ERROR_RECONNECT_INTERVAL_SEC if failed else RECONNECT_INTERVAL_SEC
                )
        finally:
            # 어떤 이유로 루프를 나가든 열린 연결은 닫는다
            if capture is not None:
                self._release(capture)
        return PlayOutcome.STOPPED

    def _connect(self, source_uri: str) -> Any:
        """열린 VideoCapture 또는 None."""
        try:
            capture = self._capture_factory(source_uri)
        except Exception:
            logger.exception("스트림을 열다 예외")
            return None
        if capture is not None and capture.isOpened():
            return capture
        # 열리지 않은 객체도 자원을 쥐고 있을 수 있어 닫아 둔다
        if capture is not None:
            self._release(capture)
        return None

    @staticmethod
    def _release(capture: Any) -> None:
        """연결을 닫는다. 닫다가 난 예외는 기록만 하고 삼킨다(정리 중 오류로 작업이 죽지 않게)."""
        try:
            capture.release()
        except Exception:
            logger.exception("스트림을 닫다 예외")

    # ---- 분석 (분석 스레드) ----

    def _analysis_loop(
        self, camera_id: int, analyzer: LiveAnalyzer, latest: _LatestFrame
    ) -> None:
        """분석 스레드의 본체. 보관함에서 최신 프레임을 꺼내 분석하고, 프레임이 없는 동안에도 지난 초 결과를 내보낸다."""
        errors = 0
        try:
            while True:
                reset, item, closed = latest.take(ANALYSIS_POLL_SEC)
                try:
                    # 끊김 알림이 오면 이전 연결의 추적을 끝낸다
                    if reset:
                        analyzer.end_session()
                    if item is not None:
                        self._analyze(camera_id, analyzer, *item)
                    # 프레임이 안 들어와도 시간이 지나면 지난 초의 결과를 보낸다
                    analyzer.flush_idle(self._wall())
                except Exception:
                    # 한 번의 예외로 분석 스레드가 죽어 이후 분석이 멈추지 않게 한다
                    errors += 1
                    if errors == 1 or errors % _LOG_EVERY == 0:
                        logger.exception(
                            "라이브 분석 중 예외 cameraId=%s (누적 %s번)",
                            camera_id,
                            errors,
                        )
                # 종료 신호가 왔고 남은 프레임도 없으면 루프를 끝낸다(남은 프레임은 다 분석한 뒤 끝낸다)
                if item is None and closed:
                    break
        finally:
            # 마지막 초 결과를 내보내고 정리한다. 여기서 난 예외가 스레드를 비정상 종료시키지 않게 막는다
            try:
                analyzer.close()
            except Exception:
                logger.exception("라이브 분석 마무리 중 예외 cameraId=%s", camera_id)
            if latest.dropped:
                logger.info(
                    "분석이 따라가지 못해 오래된 프레임을 버림 cameraId=%s (%s장)",
                    camera_id,
                    latest.dropped,
                )

    def _analyze(
        self, camera_id: int, analyzer: LiveAnalyzer, frame: np.ndarray, at: float
    ) -> None:
        """프레임 하나를 분석하고, 결과 박스를 녹화기(좌표 기록)와 실시간 화면에 넘긴다."""
        result = analyzer.analyze(frame, at)
        if result is None or self._finished.is_set():
            return  # 읽기 쪽이 끝난 뒤(분석이 늦게 끝난 경우)에는 녹화기·화면에 넘기지 않는다
        if self._recorder is not None:
            self._recorder.add_boxes(at, result.boxes)
        # 보는 사람이 없으면 프레임을 만들어 보낼 필요가 없어 wants_frames로 먼저 확인한다
        stream = self._streams.get(camera_id)
        if stream is not None and stream.wants_frames():
            stream.publish(
                LiveFrame(frame, _pixel_boxes(result.boxes, frame), result.count)
            )

    # ---- 상태 보고 ----

    def _fail(self, camera_id: int, message: str) -> PlayOutcome:
        """ERROR를 보고하고 작업 실패(FAILED)로 끝낼 때 쓴다."""
        self._set_status(camera_id, "ERROR", message)
        return PlayOutcome.FAILED

    def _set_status(self, camera_id: int, status: str, message: str | None) -> None:
        """작업 상태를 바꾸고 관리자 서버에 보고한다. 보고가 실패해도 _retry_report가 나중에 다시 보낸다."""
        # 먼저 로컬 작업 상태를 바꾼다(서버 보고 성공 여부와 무관하게 상태는 맞춰 둔다)
        self.on_status(CameraStatus(status))
        # 이전에 못 보낸 보고는 최신 상태로 덮어쓰고, 대기 시간 없이 바로 시도한다
        self._pending_report = (status, message)
        self._next_report_try = 0.0
        self._retry_report(camera_id)

    def _retry_report(self, camera_id: int) -> None:
        """못 보낸 상태 보고를 보낸다. 지금 상태 하나만 남기고, 거절(400·404)은 버린다."""
        if self._pending_report is None or self._clock() < self._next_report_try:
            return
        status, message = self._pending_report
        try:
            result, detail = self._reporter.send(
                camera_id, status, error_message=message
            )
        except Exception as error:
            # 보고기 예외로 읽기 루프가 죽지 않게, 재시도 대상으로 취급한다
            result, detail = ReportResult.RETRY, f"예외: {error}"
        # 전달 성공: 대기 중인 보고를 비운다
        if result == ReportResult.OK:
            self._pending_report = None
            return
        # 서버가 요청 자체를 거절(400·404)하면 다시 보내도 같은 결과이므로 버린다
        if result == ReportResult.REJECTED:
            logger.error(
                "상태 보고가 거절돼 버림 cameraId=%s status=%s: %s",
                camera_id,
                status,
                detail,
            )
            self._pending_report = None
            return
        # 일시 오류: 보고를 남겨 두고 REPORT_RETRY_SEC 뒤에 다시 보낸다
        logger.warning(
            "상태 보고 실패, 다시 보냄 cameraId=%s status=%s: %s",
            camera_id,
            status,
            detail,
        )
        self._next_report_try = self._clock() + REPORT_RETRY_SEC


def _pixel_boxes(boxes: list[TrackedBox], frame: np.ndarray) -> np.ndarray:
    """0~1 비율 박스를 프레임 픽셀 좌표 [x1, y1, x2, y2] 배열로 바꾼다."""
    height, width = frame.shape[:2]
    # 박스가 없어도 (0, 4) 모양을 유지하도록 reshape한다. (x, y, w, h) -> (x1, y1, x2, y2) 변환
    return np.array(
        [
            [b.x * width, b.y * height, (b.x + b.w) * width, (b.y + b.h) * height]
            for b in boxes
        ],
        dtype=np.float32,
    ).reshape(-1, 4)
