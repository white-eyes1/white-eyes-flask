import json
import logging
import threading
import time
from collections import deque
from datetime import datetime
from itertools import islice
from typing import Any, Callable, Protocol

from white_eyes_flask.camera.result_client import SendResult
from white_eyes_flask.camera.result_queue import ResultQueue

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 2000  # 요청 한 번에 보내는 최대 건수(API 명세)
DEFAULT_MAX_BUFFER = 50_000  # 보관 버퍼의 절대 상한(메모리 보호)
DEFAULT_MAX_BATCHES_PER_TICK = 5  # 한 번에 몰아 보내는 요청 수 상한
FUTURE_LIMIT_SEC = 24 * 60 * 60  # 이보다 먼 미래의 at은 잘못된 값으로 본다


class _Client(Protocol):
    def send(self, items: list[dict[str, Any]]) -> SendResult: ...


class ResultSender:
    """큐에 쌓인 1초 결과를 1초마다 관리자 앱에 보내고, 실패분은 일정 시간 보관해 다시 보낸다.

    실패(연결 실패·5xx·408·429)한 결과는 버퍼에 남겼다가 다음에 새 결과와 함께 큐에 들어온
    순서대로 보낸다. 버퍼에서 at이 (지금 − retry_buffer_sec)보다 오래된 결과는 버린다. 4xx(형식
    오류·토큰 불일치)로 거절된 묶음은 버리고 로그를 남긴다. 같은 결과를 두 번 받아도 관리자 앱이
    덮어쓴다는 전제다.

    버퍼는 전송 스레드 하나가 쓴다. 스레드를 시작하지 않고 `tick()`을 직접 부르는 것은 시험용이다.
    """

    def __init__(
        self,
        queue: ResultQueue,
        client: _Client,
        retry_buffer_sec: int,
        interval: float = 1.0,
        batch_size: int = MAX_BATCH_SIZE,
        max_buffer: int = DEFAULT_MAX_BUFFER,
        max_batches_per_tick: int = DEFAULT_MAX_BATCHES_PER_TICK,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if retry_buffer_sec <= 0 or interval <= 0:
            raise ValueError("보관 시간과 전송 간격은 0보다 커야 합니다")
        if not 0 < batch_size <= MAX_BATCH_SIZE:
            raise ValueError(f"묶음 크기는 1~{MAX_BATCH_SIZE}건이어야 합니다")
        if max_buffer <= 0 or max_batches_per_tick <= 0:
            raise ValueError("버퍼 상한과 틱당 요청 수는 0보다 커야 합니다")
        self._queue = queue
        self._client = client
        self._retry_buffer_sec = retry_buffer_sec
        self._interval = interval
        self._batch_size = batch_size
        self._max_buffer = max_buffer
        self._max_batches = max_batches_per_tick
        self._clock = clock
        self._buffer: deque[tuple[float, dict[str, Any]]] = deque()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lifecycle = threading.Lock()  # start·stop을 동시에 불러도 스레드는 하나

    @property
    def pending(self) -> int:
        """아직 못 보내고 보관 중인 결과 수."""
        return len(self._buffer)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        with self._lifecycle:
            if self.is_running:
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, name="result-sender", daemon=True
            )
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """멈추고, 남은 결과를 한 번 더 보내 보는 마지막 시도까지 기다린다(최대 timeout초)."""
        with self._lifecycle:
            thread = self._thread
            if thread is None:
                return
            self._stop.set()
            thread.join(timeout)
            if thread.is_alive():
                logger.warning("결과 전송 스레드가 %s초 안에 끝나지 않음", timeout)
            else:
                self._thread = None

    def tick(self, final: bool = False) -> None:
        """한 번의 전송 차례. 큐를 비워 버퍼에 잇고, 보관 시간이 지난 것을 버리고, 묶음으로 보낸다.

        final이면 종료 직전의 마지막 시도라 요청을 한 번만 보낸다.
        """
        self._absorb()
        limit = 1 if final else self._max_batches
        sent = 0
        while self._buffer and sent < limit:
            if self._stop.is_set() and not final:
                break
            self._expire()
            if not self._buffer:
                break
            batch = [entry[1] for entry in islice(self._buffer, self._batch_size)]
            result = self._client.send(batch)
            sent += 1
            if result is SendResult.RETRY:
                break  # 이번 차례는 여기까지. 다음 차례에 이어서 보낸다
            for _ in batch:
                self._buffer.popleft()
            if result is SendResult.REJECTED:
                logger.error("전송 거절로 %s건 버림", len(batch))

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            self._safe_tick(final=False)
            self._stop.wait(max(0.0, self._interval - (time.monotonic() - started)))
        self._safe_tick(final=True)

    def _safe_tick(self, final: bool) -> None:
        try:
            self.tick(final)
        except Exception:
            # 결과는 버퍼에 남아 있다. 스레드가 죽으면 전송이 멈추므로 로그만 남기고 이어 간다.
            logger.exception("결과 전송 중 오류")

    def _absorb(self) -> None:
        unusable = 0
        latest = self._clock() + FUTURE_LIMIT_SEC
        for result in self._queue.drain():
            epoch = _epoch_of(result)
            # 하루 넘게 미래인 시각은 오류다(만료되지 않아 버퍼에 계속 남는다)
            if epoch is None or epoch > latest or not _serializable(result):
                unusable += 1
                continue
            self._buffer.append((epoch, result))
        if unusable:
            logger.warning(
                "시각을 읽을 수 없거나 보낼 수 없는 형식이라 %s건 버림", unusable
            )
        # 만료된 것을 먼저 걸러야 상한 때문에 멀쩡한 결과가 밀려나지 않는다.
        self._expire()
        overflow = len(self._buffer) - self._max_buffer
        if overflow > 0:
            for _ in range(overflow):
                self._buffer.popleft()
            logger.warning("보관 버퍼가 가득 차 가장 오래된 %s건 버림", overflow)

    def _expire(self) -> None:
        cutoff = self._clock() - self._retry_buffer_sec
        if not any(epoch < cutoff for epoch, _ in self._buffer):
            return
        kept = deque(entry for entry in self._buffer if entry[0] >= cutoff)
        dropped = len(self._buffer) - len(kept)
        self._buffer = kept
        logger.warning(
            "보관 시간(%s초)이 지나 %s건 버림", self._retry_buffer_sec, dropped
        )


def _epoch_of(result: dict[str, Any]) -> float | None:
    """결과의 at(시간대가 붙은 ISO 시각)을 초 단위 epoch로. 읽을 수 없으면 None."""
    try:
        moment = datetime.fromisoformat(result["at"])
        if moment.tzinfo is None:
            return None
        return moment.timestamp()
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        return None


def _serializable(result: dict[str, Any]) -> bool:
    """JSON으로 바꿀 수 없는 결과 하나가 뒤의 정상 결과를 막지 않게 받을 때 거른다."""
    try:
        json.dumps(result)
    except (TypeError, ValueError):
        return False
    return True
