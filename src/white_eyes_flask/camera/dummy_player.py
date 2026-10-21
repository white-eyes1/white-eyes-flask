import contextlib
import copy
import logging
import math
import queue
import threading
import time
from typing import Any, Callable

from white_eyes_flask.analysis.dummy_cache import (
    CacheInvalidError,
    CachedResult,
    DummyCache,
)
from white_eyes_flask.camera.kst import format_at
from white_eyes_flask.camera.player import PlayOutcome
from white_eyes_flask.camera.recording_registrar import RegisterResult
from white_eyes_flask.camera.result_queue import ResultSink

logger = logging.getLogger(__name__)

ERROR_CACHE = "캐시 파일을 읽을 수 없음"
MAX_CATCH_UP_SEC = 5  # 이보다 오래 멈췄으면 놓친 초를 채우지 않고 건너뛴다
WAKE_MARGIN_SEC = 0.002  # 초 경계보다 살짝 뒤에 깨어나 같은 초를 두 번 보지 않게 한다
SHUTDOWN_JOIN_SEC = 2.0
QUEUE_POLL_SEC = 0.2  # 작업 스레드가 정지를 확인하는 간격
MAX_PENDING_REGISTRATIONS = 100


class RegistrationWorker:
    """녹화 등록(POST)을 재생 루프와 따로 순서대로 처리한다.

    관리자 앱이 느리거나 꺼져 있어도 1초 결과가 밀리지 않는다. 일시 오류는 같은
    startedAt으로 retry_interval마다 다시 보내고, retry_window를 넘기면 포기하고 다음
    반복으로 넘어간다. 거절(400·401·404)은 다시 보내지 않는다. inline=True이면 호출한
    스레드에서 바로 한 번만 보낸다(시험용).
    """

    def __init__(
        self,
        registrar: Any,
        camera_id: int,
        video_path: str,
        box_path: str,
        stop: threading.Event,
        *,
        retry_interval: float = 1.0,
        retry_window: float = 300.0,
        max_pending: int = MAX_PENDING_REGISTRATIONS,
        inline: bool = False,
    ) -> None:
        self._registrar = registrar
        self._camera_id = camera_id
        self._video_path = video_path
        self._box_path = box_path
        self._stop = stop
        self._retry_interval = retry_interval
        self._retry_window = retry_window
        self._inline = inline
        # (startedMs, 큐에 넣은 시각). 관리자 앱이 오래 꺼져 있어도 무한히 쌓이지 않는다.
        self._queue: queue.Queue[tuple[int, float]] = queue.Queue(maxsize=max_pending)
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"cam-{camera_id}-recording", daemon=True
        )

    def start(self) -> None:
        if not self._inline:
            self._thread.start()

    def submit(self, started_ms: int) -> None:
        if self._inline:
            self._send_once(started_ms)
            return
        item = (started_ms, time.monotonic())
        while True:
            try:
                self._queue.put_nowait(item)
                return
            except queue.Full:
                # 가장 오래된 등록을 버린다. 관리자 앱은 다음 반복이 등록될 때 이전 반복을 닫는다.
                with contextlib.suppress(queue.Empty):
                    dropped, _ = self._queue.get_nowait()
                    logger.warning(
                        "녹화 등록이 밀려 가장 오래된 것을 버림 cameraId=%s startedMs=%s",
                        self._camera_id,
                        dropped,
                    )

    def shutdown(self) -> None:
        """남은 등록은 버리고 작업 스레드를 끝낸다."""
        self._closed.set()
        if not self._inline:
            self._thread.join(SHUTDOWN_JOIN_SEC)

    def _halted(self) -> bool:
        return self._stop.is_set() or self._closed.is_set()

    def _run(self) -> None:
        while not self._halted():
            try:
                started_ms, queued_at = self._queue.get(timeout=QUEUE_POLL_SEC)
            except queue.Empty:
                continue
            self._register(started_ms, queued_at)

    def _send_once(self, started_ms: int) -> RegisterResult:
        return self._registrar.register(
            self._camera_id, started_ms, self._video_path, self._box_path
        )

    def _register(self, started_ms: int, queued_at: float) -> None:
        # 기한은 큐에 넣은 때부터 잰다(앞 등록을 기다린 시간도 포함)
        deadline = queued_at + self._retry_window
        while not self._halted():
            if time.monotonic() >= deadline:
                logger.error(
                    "녹화 등록 포기 cameraId=%s startedMs=%s, %s초 안에 성공하지 못함",
                    self._camera_id,
                    started_ms,
                    self._retry_window,
                )
                return
            result = self._send_once(started_ms)
            if result == RegisterResult.OK:
                return
            if result == RegisterResult.REJECTED:
                # 같은 요청을 다시 보내도 소용없다. 원인은 registrar가 로그에 남겼다.
                logger.error(
                    "녹화 등록 거절 cameraId=%s startedMs=%s",
                    self._camera_id,
                    started_ms,
                )
                return
            self._closed.wait(self._retry_interval)


class DummyPlayer:
    """더미 재생·반복 등록(FNC-S-010).

    캐시된 초별 결과를 실제 시간에 맞춰 1초씩 전송 큐에 넣고, 영상 길이 L초가 지나
    처음부터 다시 시작할 때마다 녹화를 새로 등록한다.

    반복 시작 시각 S는 초 경계(밀리초 .000)에 맞춘다. 그러면 결과의 at - S가 정확히
    영상 위치 k와 같아져 다시보기 시점 계산이 맞는다. 첫 결과(k=0)도 S와 같은 초
    경계에서 보낸다.
    """

    def __init__(
        self,
        cache: DummyCache,
        registrar: Any,
        sink: ResultSink,
        reporter: Any,
        *,
        clock: Callable[[], float] = time.time,
        waiter: Callable[[threading.Event, float], bool] | None = None,
        retry_interval: float = 1.0,
        retry_window: float = 300.0,
        inline_registration: bool = False,
    ) -> None:
        self._cache = cache
        self._registrar = registrar
        self._sink = sink
        self._reporter = reporter
        self._clock = clock
        self._wait = waiter or (lambda stop, timeout: stop.wait(timeout))
        self._retry_interval = retry_interval
        self._retry_window = retry_window
        self._inline = inline_registration

    def play(
        self, camera_id: int, source_uri: str, stop: threading.Event
    ) -> PlayOutcome:
        if stop.is_set():
            return PlayOutcome.STOPPED
        try:
            cached = self._cache.load(source_uri)
        except CacheInvalidError:
            logger.exception("더미 캐시를 읽을 수 없음 cameraId=%s", camera_id)
            # 깨진 캐시는 지워서 다음 시작 때 다시 준비하게 한다. 지우지 못해도 ERROR는 보고한다.
            try:
                self._cache.invalidate(source_uri)
            except OSError:
                logger.exception("깨진 캐시를 지우지 못함 cameraId=%s", camera_id)
            self._report_error(camera_id)
            return PlayOutcome.FAILED
        worker = RegistrationWorker(
            self._registrar,
            camera_id,
            source_uri,
            self._cache.boxes_relative_path(source_uri),
            stop,
            retry_interval=self._retry_interval,
            retry_window=self._retry_window,
            inline=self._inline,
        )
        worker.start()
        try:
            self._loop(camera_id, cached, worker, stop)
        finally:
            worker.shutdown()
        return PlayOutcome.STOPPED

    def _loop(
        self,
        camera_id: int,
        cached: CachedResult,
        worker: RegistrationWorker,
        stop: threading.Event,
    ) -> None:
        state = _Repetition()
        last_at: int | None = None
        while not stop.is_set():
            now_sec = math.floor(self._clock())
            if last_at is None:
                last_at = now_sec  # 시작한 초는 지나갔다. 다음 경계부터 보낸다.
            elif now_sec > last_at:
                missed = now_sec - last_at
                if missed > MAX_CATCH_UP_SEC:
                    logger.warning(
                        "재생이 %s초 멈춰 그 사이 결과를 건너뜀 cameraId=%s",
                        missed - 1,
                        camera_id,
                    )
                    first = now_sec
                else:
                    first = last_at + 1  # 놓친 초를 순서대로 채운다
                for at in range(first, now_sec + 1):
                    if stop.is_set():
                        return
                    self._emit(camera_id, cached, worker, state, at)
                last_at = now_sec
            # 같은 초에 깨거나 시계가 되돌아가면 아무것도 하지 않고 다시 기다린다
            delay = max(0.0, now_sec + 1 - self._clock()) + WAKE_MARGIN_SEC
            if self._wait(stop, delay):
                break

    def _emit(
        self,
        camera_id: int,
        cached: CachedResult,
        worker: RegistrationWorker,
        state: "_Repetition",
        at: int,
    ) -> None:
        # k >= L이면 다음 반복(새 S로 녹화 등록). at이 초 경계라 S도 초 경계다.
        if state.start is None or at - state.start >= cached.duration_sec:
            state.start = at
            worker.submit(at * 1000)
        k = at - state.start
        second = copy.deepcopy(cached.seconds[k])
        self._sink.put(
            {
                "cameraId": camera_id,
                "at": format_at(at),
                "personCount": second["personCount"],
                "in": second["in"],
                "out": second["out"],
                "dwellSec": second["dwellSec"],
            }
        )

    def _report_error(self, camera_id: int) -> None:
        try:
            self._reporter.report(camera_id, "ERROR", error_message=ERROR_CACHE)
        except Exception:
            logger.warning("상태 보고 중 오류 cameraId=%s", camera_id)


class _Repetition:
    """지금 반복의 시작 시각 S(초 단위 epoch). 반복이 시작되기 전에는 None."""

    def __init__(self) -> None:
        self.start: int | None = None
