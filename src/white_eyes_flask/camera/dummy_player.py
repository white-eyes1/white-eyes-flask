"""더미 캠 재생(FNC-S-010).

준비 단계가 만들어 둔 캐시(초별 결과)를 실제 시간에 맞춰 1초마다 전송 큐에 넣고, 영상 길이가
끝나면 처음부터 반복하며 그때마다 녹화를 새로 등록한다.
구성: RegistrationWorker(녹화 등록 전담 스레드), DummyPlayer(재생 루프), _Repetition(지금
반복의 시작 시각을 재생 루프와 미리보기가 함께 보도록 공유).
"""

import contextlib
import copy
import logging
import math
import queue
import threading
import time
from typing import Any, Callable

from white_eyes_flask.analysis.dummy_cache import (
    CachedResult,
    CacheInvalidError,
    DummyCache,
)
from white_eyes_flask.camera.dummy_preview import DummyPreview
from white_eyes_flask.camera.kst import format_at
from white_eyes_flask.camera.player import PlayOutcome
from white_eyes_flask.camera.recording_registrar import RegisterResult
from white_eyes_flask.camera.result_queue import ResultSink

logger = logging.getLogger(__name__)

# 캐시가 깨졌을 때 ERROR 상태 보고에 붙이는 사유 문구.
ERROR_CACHE = "캐시 파일을 읽을 수 없음"
MAX_CATCH_UP_SEC = 5  # 이보다 오래 멈췄으면 놓친 초를 채우지 않고 건너뛴다
WAKE_MARGIN_SEC = 0.002  # 초 경계보다 살짝 뒤에 깨어나 같은 초를 두 번 보지 않게 한다
# 등록 스레드가 끝나길 기다리는 최대 시간(초).
SHUTDOWN_JOIN_SEC = 2.0
QUEUE_POLL_SEC = 0.2  # 작업 스레드가 정지를 확인하는 간격
# 등록 대기열 상한. 관리자 앱이 오래 꺼져 있어도 대기열이 끝없이 커지지 않게 한다.
MAX_PENDING_REGISTRATIONS = 100


class RegistrationWorker:
    """녹화 등록(POST)을 재생 루프와 따로 순서대로 처리한다.

    관리자 앱이 느리거나 꺼져 있어도 1초 결과가 밀리지 않는다. 일시 오류는 같은
    startedAt으로 retry_interval마다 다시 보내고, retry_window를 넘기면 포기하고 다음
    반복으로 넘어간다. 거절(400)은 다시 보내지 않는다. inline=True이면 호출한
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
        """등록 작업자를 만든다(스레드는 start()에서 시작).

        :param registrar: register()를 가진 녹화 등록기(RecordingRegistrar)
        :param video_path: 등록할 영상의 공유 폴더 기준 경로(더미는 sourceUri 그대로)
        :param box_path: 좌표 파일의 공유 폴더 기준 경로
        :param stop: 캠 정지 플래그. 서면 대기·재시도를 멈춘다
        :param max_pending: 대기열 상한
        :param inline: True면 스레드 없이 호출한 스레드에서 한 번만 보낸다(시험용)
        """
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
        # shutdown() 전용 신호. 캠 정지 플래그와 따로라 이 작업자만 끌 수 있다.
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"cam-{camera_id}-recording", daemon=True
        )

    def start(self) -> None:
        """inline이 아니면 등록 스레드를 시작한다."""
        if not self._inline:
            self._thread.start()

    def submit(self, started_ms: int) -> None:
        """반복이 시작된 시각(밀리초)을 등록 대기열에 넣는다.

        재생 루프가 느려지지 않게 절대 기다리지 않는다. 가득 차면 가장 오래된 것을 버린다.
        """
        if self._inline:
            self._send_once(started_ms)
            return
        # 큐에 넣은 시각을 함께 둔다. 재시도 기한을 큐에 들어온 때부터 재기 위해서다.
        item = (started_ms, time.monotonic())
        # 넣을 자리가 날 때까지: 가득 차 있으면 하나 버리고 다시 넣어 본다.
        while True:
            try:
                self._queue.put_nowait(item)
                return
            except queue.Full:
                # 가장 오래된 등록을 버린다. 관리자 앱은 다음 반복이 등록될 때 이전 반복을 닫는다.
                # 버리려는 순간 소비자가 이미 비웠을 수 있다. 그때는 그냥 다시 넣어 본다.
                with contextlib.suppress(queue.Empty):
                    dropped, _ = self._queue.get_nowait()
                    logger.warning(
                        "녹화 등록이 밀려 가장 오래된 것을 버림 cameraId=%s startedMs=%s",
                        self._camera_id,
                        dropped,
                    )

    def shutdown(self) -> None:
        """남은 등록은 버리고 작업 스레드를 끝낸다."""
        # 작업 스레드가 다음 확인에서 빠져나오게 신호를 준다.
        self._closed.set()
        if not self._inline:
            # 등록 요청이 진행 중이어도 SHUTDOWN_JOIN_SEC를 넘기면 기다리지 않는다(daemon 스레드).
            self._thread.join(SHUTDOWN_JOIN_SEC)

    def _halted(self) -> bool:
        """캠이 정지됐거나 이 작업자가 닫혔나. 참이면 등록을 그만둔다."""
        return self._stop.is_set() or self._closed.is_set()

    def _run(self) -> None:
        """작업 스레드 본체. 대기열에서 하나씩 꺼내 등록한다.

        꺼내기는 QUEUE_POLL_SEC마다 깨어나, 대기열이 비어 있어도 정지 여부를 확인한다.
        """
        while not self._halted():
            try:
                started_ms, queued_at = self._queue.get(timeout=QUEUE_POLL_SEC)
            except queue.Empty:
                continue
            self._register(started_ms, queued_at)

    def _send_once(self, started_ms: int) -> RegisterResult:
        """등록 요청을 한 번 보낸다. 결과(OK·RETRY·REJECTED)를 그대로 돌려준다."""
        return self._registrar.register(
            self._camera_id, started_ms, self._video_path, self._box_path
        )

    def _register(self, started_ms: int, queued_at: float) -> None:
        """한 반복의 등록을 성공·거절·기한 초과·정지 중 하나가 될 때까지 시도한다."""
        # 기한은 큐에 넣은 때부터 잰다(앞 등록을 기다린 시간도 포함)
        deadline = queued_at + self._retry_window
        while not self._halted():
            # 기한(retry_window)을 넘기면 포기하고 다음 반복으로 넘어간다.
            if time.monotonic() >= deadline:
                logger.error(
                    "녹화 등록 포기 cameraId=%s startedMs=%s, %s초 안에 성공하지 못함",
                    self._camera_id,
                    started_ms,
                    self._retry_window,
                )
                return
            # 일시 오류이면 같은 startedAt으로 다시 보낸다(새 반복이 아니라 같은 반복의 재시도다).
            result = self._send_once(started_ms)
            # 성공. 이 반복의 등록은 끝.
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
            # RETRY(일시 오류)이면 간격을 두고 다시 시도한다. sleep 대신 이벤트를 기다려 shutdown 때 바로 깨어난다.
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
        streams: Any = None,
        preview_fps: int = 5,
    ) -> None:
        """더미 재생기를 만든다.

        :param cache: 분석 결과 캐시(초별 결과·영상 길이를 읽는다)
        :param registrar: 녹화 등록기
        :param sink: 1초 결과를 넣는 전송 큐
        :param reporter: 캠 상태 보고기(ERROR 보고에 쓴다)
        :param clock: 현재 시각(epoch 초). 시험에서 바꿔 끼운다
        :param waiter: (정지 플래그, 초)를 받아 그만큼 기다리다 정지되면 True. 시험에서 바꿔 끼운다
        :param inline_registration: True면 녹화 등록을 스레드 없이 바로 보낸다(시험용)
        :param streams: 실시간 화면 스트림 관리. 없으면 미리보기 프레임을 만들지 않는다
        :param preview_fps: 미리보기 프레임을 만드는 초당 횟수
        """
        self._cache = cache
        self._registrar = registrar
        self._sink = sink
        self._reporter = reporter
        self._clock = clock
        # 기본 대기는 Event.wait. 정지 신호가 오면 True를 돌려주며 바로 깨어난다.
        self._wait = waiter or (lambda stop, timeout: stop.wait(timeout))
        self._retry_interval = retry_interval
        self._retry_window = retry_window
        self._inline = inline_registration
        # 실시간 화면·미리보기용 프레임을 넣을 StreamService(FNC-S-010 5번). 없으면 프레임을 만들지 않는다
        self._streams = streams
        self._preview_fps = preview_fps

    def play(
        self, camera_id: int, source_uri: str, stop: threading.Event
    ) -> PlayOutcome:
        """캐시된 결과를 정지될 때까지 재생한다.

        등록 작업자와 미리보기 스레드를 띄우고 _loop를 돈다. 끝나면(정지) 둘을 반드시 정리한다.
        캐시가 깨졌으면 캐시를 지우고 ERROR를 보고한 뒤 FAILED를 돌려준다.
        """
        # 시작하기도 전에 정지가 들어왔다.
        if stop.is_set():
            return PlayOutcome.STOPPED
        try:
            # 영상 길이와 초별 결과(personCount·in·out·dwellSec)를 읽는다.
            cached = self._cache.load(source_uri)
        # 캐시 파일이 깨져 재생할 수 없다.
        except CacheInvalidError:
            logger.exception("더미 캐시를 읽을 수 없음 cameraId=%s", camera_id)
            # 깨진 캐시는 지워서 다음 시작 때 다시 준비하게 한다. 지우지 못해도 ERROR는 보고한다.
            try:
                self._cache.invalidate(source_uri)
            except OSError:
                logger.exception("깨진 캐시를 지우지 못함 cameraId=%s", camera_id)
            self._report_error(camera_id)
            return PlayOutcome.FAILED
        # 녹화 등록은 따로 도는 작업자에게 맡긴다. 관리자 앱이 느려도 1초 결과가 밀리지 않게 하려는 것이다.
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
        # 재생 루프와 미리보기가 같은 반복 시작 시각을 보도록 공유하는 객체.
        state = _Repetition(cached.duration_sec)
        # 미리보기 시작이 예외로 실패해도 finally가 None을 검사하며 정리하도록 먼저 None으로 둔다.
        preview: DummyPreview | None = None
        try:
            preview = self._start_preview(camera_id, source_uri, cached, state, stop)
            # 정지 플래그가 서야 돌아온다.
            self._loop(camera_id, cached, worker, stop, state)
        finally:
            # 재생 루프가 예외로 끝나도 미리보기 스레드와 등록 작업을 반드시 끝낸다
            if preview is not None:
                preview.shutdown()
            worker.shutdown()
        # 루프는 정지로만 끝난다. 정상 종료는 STOPPED다.
        return PlayOutcome.STOPPED

    def _start_preview(
        self,
        camera_id: int,
        source_uri: str,
        cached: CachedResult,
        state: "_Repetition",
        stop: threading.Event,
    ) -> DummyPreview | None:
        """캠의 스트림이 있으면 미리보기 프레임을 만드는 스레드를 시작한다."""
        # 스트림 관리자가 없거나 이 캠의 스트림이 없으면 미리보기를 만들지 않는다.
        stream = self._streams.get(camera_id) if self._streams is not None else None
        if stream is None:
            return None
        preview = DummyPreview(
            stream,
            self._cache.video_path(source_uri),
            self._cache.boxes_path(source_uri),
            cached.seconds,
            # 지금 보여 줄 영상 위치(초)를 반복 시작 시각 S에서 계산하는 함수. 미리보기 스레드가 부른다.
            lambda: state.position(self._clock()),
            fps=self._preview_fps,
            stop=stop,
        )
        preview.start(f"cam-{camera_id}-preview")
        return preview

    def _loop(
        self,
        camera_id: int,
        cached: CachedResult,
        worker: RegistrationWorker,
        stop: threading.Event,
        state: "_Repetition | None" = None,
    ) -> None:
        """1초마다 결과를 만들어 큐에 넣는 재생 루프. 정지 신호가 오면 끝난다.

        시계가 초 경계를 넘을 때마다 그 초의 결과 1건을 만든다. 일시적으로 멈췄다 돌아오면
        놓친 초를 순서대로 채우되, 너무 오래(MAX_CATCH_UP_SEC 초과) 멈췄으면 건너뛴다.
        """
        state = state or _Repetition(cached.duration_sec)
        # 마지막으로 결과를 만든 초(epoch). None이면 아직 만든 적이 없다.
        last_at: int | None = None
        while not stop.is_set():
            # 지금이 속한 초. 결과는 초 경계마다 1건씩 만든다.
            now_sec = math.floor(self._clock())
            if last_at is None:
                last_at = now_sec  # 시작한 초는 지나갔다. 다음 경계부터 보낸다.
            # 새 초가 시작됐다. 그 사이 못 만든 초가 있으면(missed > 1) 순서대로 채운다.
            elif now_sec > last_at:
                # 마지막으로 만든 초와 지금 초의 차(보통 1).
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
                # first부터 지금 초까지 각 초의 결과를 만든다.
                for at in range(first, now_sec + 1):
                    if stop.is_set():
                        return
                    self._emit(camera_id, cached, worker, state, at)
                # 여기까지 만들었다고 기록한다.
                last_at = now_sec
            # 같은 초에 깨거나 시계가 되돌아가면 아무것도 하지 않고 다시 기다린다
            # 다음 초 경계까지 남은 시간만큼 기다린다. 약간의 여유를 더해 경계 직후에 깨어난다.
            delay = max(0.0, now_sec + 1 - self._clock()) + WAKE_MARGIN_SEC
            # 기다리는 동안 정지 신호가 왔으면(True) 루프를 끝낸다.
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
        """초 at의 결과 1건을 전송 큐에 넣는다.

        새 반복이 시작되는 초(처음 또는 영상 길이 경과)이면 녹화 등록도 요청한다.
        """
        # k >= L이면 다음 반복(새 S로 녹화 등록). at이 초 경계라 S도 초 경계다.
        if state.start is None or at - state.start >= cached.duration_sec:
            state.start = at
            # 등록 시각은 밀리초. at이 초 경계라 .000으로 끝난다.
            worker.submit(at * 1000)
        # k는 이번 반복에서 흐른 초이며 영상 위치와 같다. 캐시의 k번째 초 결과가 이 초의 결과다.
        k = at - state.start
        # 캐시 원본을 건드리지 않도록 복사한다(큐 쪽에서 값을 바꿔도 다음 반복에 영향이 없게).
        second = copy.deepcopy(cached.seconds[k])
        # 전송 큐에 넣는 1초 결과. at은 초 경계의 KST 시각 문자열이다.
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
        """캠 상태를 ERROR로 보고한다. 보고가 실패해도 재생 정리는 계속된다."""
        try:
            self._reporter.report(camera_id, "ERROR", error_message=ERROR_CACHE)
        # 보고 실패는 경고 로그만 남기고 넘어간다.
        except Exception:
            logger.warning("상태 보고 중 오류 cameraId=%s", camera_id)


class _Repetition:
    """지금 반복의 시작 시각 S(초 단위 epoch). 반복이 시작되기 전에는 None.

    재생 루프가 S를 정하고, 미리보기 스레드가 같은 S로 영상 위치를 계산한다.
    """

    def __init__(self, duration_sec: float) -> None:
        """반복 시작 시각을 만든다. 아직 시작 전이라 None이다.

        :param duration_sec: 영상 길이(초). 반복 주기이자 위치의 상한이다.
        """
        self._lock = threading.Lock()
        self._start: int | None = None
        self._duration_sec = duration_sec

    @property
    def start(self) -> int | None:
        """지금 반복의 시작 시각 S(초 단위 epoch). 시작 전이면 None."""
        with self._lock:
            return self._start

    @start.setter
    def start(self, value: int | None) -> None:
        """재생 루프가 새 반복을 시작할 때 S를 바꾼다."""
        with self._lock:
            self._start = value

    def position(self, now: float) -> float | None:
        """지금 영상 위치(초). 재생이 시작되지 않았거나 시계가 S보다 앞서면 None.

        다음 반복의 S가 정해지기 전(영상 길이를 넘은 직후 1초 안)에는 마지막 위치에 머문다.
        """
        start = self.start
        if start is None or now < start:
            return None
        # 영상 길이를 넘지 않게 상한을 둔다. 0.001초 모자라게 해 마지막 프레임 안쪽에 머문다.
        return min(now - start, max(0.0, self._duration_sec - 0.001))
