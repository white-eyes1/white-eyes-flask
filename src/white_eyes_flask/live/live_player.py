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
from white_eyes_flask.recording.recorder import (
    LiveRecorder,
    RecorderFailed,
    RecorderUnavailable,
)
from white_eyes_flask.stream.stream_service import LiveFrame, StreamService

logger = logging.getLogger(__name__)

# 명세(FNC-S-007 4·5번, conventions.md LIVE_RECONNECT_FAIL_SEC)
LIVE_RECONNECT_FAIL_SEC = 30
RECONNECT_INTERVAL_SEC = 2
ERROR_RECONNECT_INTERVAL_SEC = 30
# 못 보낸 상태 보고를 다시 보내는 간격(보고가 막혀도 읽기를 오래 붙잡지 않게 읽기와 같은 스레드에서 가끔만)
REPORT_RETRY_SEC = 5
ANALYSIS_JOIN_SEC = 2.0
ANALYSIS_POLL_SEC = 0.5
_LOG_EVERY = 30  # 같은 실패가 이어질 때 로그는 처음과 이 횟수마다만

ERROR_STREAM = "스트림 연결 실패"
ERROR_MODEL = "분석 모델을 불러올 수 없음"
ERROR_RECORDER_START = "녹화를 시작할 수 없음"
ERROR_RECORDER = "녹화 중 오류가 발생함"
ERROR_UNEXPECTED = "분석 중 오류가 발생함"


class _LatestFrame:
    """분석 스레드에 넘길 최신 프레임 한 장. 새 프레임이 오면 아직 안 가져간 프레임은 버린다."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: tuple[np.ndarray, float] | None = None
        self._reset = False
        self._closed = False
        self.dropped = 0

    def put(self, frame: np.ndarray, at: float) -> None:
        with self._cond:
            if self._frame is not None:
                self.dropped += 1
            self._frame = (frame, at)
            self._cond.notify()

    def put_reset(self) -> None:
        """스트림이 끊겼다. 끊긴 연결의 프레임은 버리고 분석 스레드에 알린다."""
        with self._cond:
            self._frame = None
            self._reset = True
            self._cond.notify()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify()

    def take(
        self, timeout: float
    ) -> tuple[bool, tuple[np.ndarray, float] | None, bool]:
        """(끊김 알림, 프레임, 닫힘). 아무것도 없으면 timeout초 기다린다."""
        with self._cond:
            self._cond.wait_for(
                lambda: self._frame is not None or self._reset or self._closed,
                timeout=timeout,
            )
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
        recorder: LiveRecorder | None,
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
        self._clock = clock
        self._wall = wall
        # CameraService가 작업 상태를 바꾸는 함수를 넣는다(작업이 살아 있는 채 ERROR가 되므로)
        self.on_status: Callable[[CameraStatus], None] = lambda status: None
        self._pending_report: tuple[str, str | None] | None = None
        self._next_report_try = 0.0
        self._finished = (
            threading.Event()
        )  # play()가 끝났다. 남은 분석 스레드는 더 넘기지 않는다

    def play(
        self, camera_id: int, source_uri: str, stop: threading.Event
    ) -> PlayOutcome:
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
        latest = _LatestFrame()
        worker = threading.Thread(
            target=self._analysis_loop,
            args=(camera_id, analyzer, latest),
            name=f"cam-{camera_id}-analysis",
            daemon=True,
        )
        worker.start()
        try:
            return self._read_loop(camera_id, source_uri, stop, latest)
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
            latest.close()
            worker.join(ANALYSIS_JOIN_SEC)
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
        shown = mask_uri(source_uri)
        capture = None
        failed = False  # ERROR로 보고한 상태
        last_good = self._clock()  # 마지막으로 프레임을 읽은 때(처음엔 시작한 때)
        attempts = 0
        last_slot = -1
        try:
            while not stop.is_set():
                self._retry_report(camera_id)
                if capture is None:
                    capture = self._connect(source_uri)
                    if capture is not None:
                        if stop.is_set():
                            break  # 연결하는 동안 정지가 들어왔다. 읽지 않고 나간다
                        logger.info(
                            "스트림 연결됨 cameraId=%s uri=%s", camera_id, shown
                        )
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
                    if ok and frame is not None and getattr(frame, "size", 1) > 0:
                        at = self._wall()
                        last_good = self._clock()
                        attempts = 0
                        if failed:
                            failed = False
                            self._set_status(camera_id, "RUNNING", None)
                        if self._recorder is not None:
                            self._recorder.write_frame(frame, at)
                        slot = math.floor(at * self._fps)
                        if slot != last_slot:
                            last_slot = slot
                            latest.put(frame, at)
                        continue
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
                if not failed and self._clock() - last_good >= self._reconnect_fail_sec:
                    failed = True
                    self._set_status(camera_id, "ERROR", ERROR_STREAM)
                stop.wait(
                    ERROR_RECONNECT_INTERVAL_SEC if failed else RECONNECT_INTERVAL_SEC
                )
        finally:
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
        if capture is not None:
            self._release(capture)
        return None

    @staticmethod
    def _release(capture: Any) -> None:
        try:
            capture.release()
        except Exception:
            logger.exception("스트림을 닫다 예외")

    # ---- 분석 (분석 스레드) ----

    def _analysis_loop(
        self, camera_id: int, analyzer: LiveAnalyzer, latest: _LatestFrame
    ) -> None:
        errors = 0
        try:
            while True:
                reset, item, closed = latest.take(ANALYSIS_POLL_SEC)
                try:
                    if reset:
                        analyzer.end_session()
                    if item is not None:
                        self._analyze(camera_id, analyzer, *item)
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
                if item is None and closed:
                    break
        finally:
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
        result = analyzer.analyze(frame, at)
        if result is None or self._finished.is_set():
            return  # 읽기 쪽이 끝난 뒤(분석이 늦게 끝난 경우)에는 녹화기·화면에 넘기지 않는다
        if self._recorder is not None:
            self._recorder.add_boxes(at, result.boxes)
        stream = self._streams.get(camera_id)
        if stream is not None and stream.wants_frames():
            stream.publish(
                LiveFrame(frame, _pixel_boxes(result.boxes, frame), result.count)
            )

    # ---- 상태 보고 ----

    def _fail(self, camera_id: int, message: str) -> PlayOutcome:
        self._set_status(camera_id, "ERROR", message)
        return PlayOutcome.FAILED

    def _set_status(self, camera_id: int, status: str, message: str | None) -> None:
        self.on_status(CameraStatus(status))
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
            result, detail = ReportResult.RETRY, f"예외: {error}"
        if result == ReportResult.OK:
            self._pending_report = None
            return
        if result == ReportResult.REJECTED:
            logger.error(
                "상태 보고가 거절돼 버림 cameraId=%s status=%s: %s",
                camera_id,
                status,
                detail,
            )
            self._pending_report = None
            return
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
    return np.array(
        [
            [b.x * width, b.y * height, (b.x + b.w) * width, (b.y + b.h) * height]
            for b in boxes
        ],
        dtype=np.float32,
    ).reshape(-1, 4)
