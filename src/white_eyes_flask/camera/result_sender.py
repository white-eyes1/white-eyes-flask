"""FNC-S-006 1초 결과 전송·재전송 (ResultSender)"""

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
    """묶음을 보내는 쪽의 규칙(Protocol). 실제 구현은 ResultClient다."""

    def send(self, items: list[dict[str, Any]]) -> SendResult: ...


class ResultSender:
    """큐에 쌓인 1초 결과를 1초마다 관리자 앱에 보내고, 실패분은 일정 시간 보관해 다시 보낸다.

    실패(연결 실패·5xx와 400을 뺀 4xx, 토큰 불일치 401·403 포함)한 결과는 버퍼에 남겼다가
    다음에 새 결과와 함께 큐에 들어온 순서대로 보낸다. 버퍼에서 at이 (지금 − retry_buffer_sec)
    보다 오래된 결과는 버린다. 400(형식 오류)으로 거절된 묶음만 버리고 로그를 남긴다. 같은
    결과를 두 번 받아도 관리자 앱이 덮어쓴다는 전제다.

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
        """전송기를 만든다(스레드는 start()에서 시작).

        :param queue: 1초 결과가 쌓이는 큐
        :param client: 묶음을 보내는 객체(send)
        :param retry_buffer_sec: 못 보낸 결과를 보관하는 시간(초). at이 이보다 오래된 것은 버린다
        :param interval: 전송 주기(초)
        :param batch_size: 요청 한 번에 보내는 건수(API 최대 2000)
        :param max_buffer: 보관 버퍼의 건수 상한
        :param max_batches_per_tick: 한 차례에 몰아 보내는 요청 수 상한
        :param clock: 현재 시각(epoch 초). 시험에서 바꿔 끼운다
        """
        # 잘못된 설정은 시작할 때 바로 알 수 있게 예외로 막는다.
        if retry_buffer_sec <= 0 or interval <= 0:
            raise ValueError("보관 시간과 전송 간격은 0보다 커야 합니다")
        # API가 한 요청에 받는 최대 건수를 넘기면 안 된다.
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
        # (at의 epoch 초, 결과) 목록. 맨 앞이 가장 오래된 것이며 전송 스레드 하나만 건드린다.
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
        """전송 스레드가 돌고 있나."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """전송 스레드를 시작한다. 이미 돌고 있으면 아무것도 하지 않는다."""
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
            # 종료 신호를 받은 스레드는 마지막 전송(final)까지 하고 끝난다. 그것을 timeout까지 기다린다.
            thread.join(timeout)
            if thread.is_alive():
                logger.warning("결과 전송 스레드가 %s초 안에 끝나지 않음", timeout)
            else:
                self._thread = None

    def tick(self, final: bool = False) -> None:
        """한 번의 전송 차례. 큐를 비워 버퍼에 잇고, 보관 시간이 지난 것을 버리고, 묶음으로 보낸다.

        final이면 종료 직전의 마지막 시도라 요청을 한 번만 보낸다.
        """
        # 먼저 큐의 새 결과를 버퍼 뒤로 옮긴다.
        self._absorb()
        # final(종료 직전 마지막 시도)은 요청을 한 번만 보낸다. 평소에는 한 차례에 최대 max_batches번까지 몰아 보낸다.
        limit = 1 if final else self._max_batches
        sent = 0
        while self._buffer and sent < limit:
            # 정지 신호가 왔으면 더 보내지 않고 나간다. 남은 것은 마지막 시도(final)가 처리한다.
            if self._stop.is_set() and not final:
                break
            # 이전 전송이 오래 걸렸을 수 있으니 보내기 직전에 만료분을 다시 걸러낸다.
            self._expire()
            if not self._buffer:
                break
            # 가장 오래된 것부터 batch_size건. 버퍼에서는 아직 빼지 않는다(실패하면 그대로 남아야 한다).
            batch = [entry[1] for entry in islice(self._buffer, self._batch_size)]
            result = self._client.send(batch)
            sent += 1
            if result is SendResult.RETRY:
                break  # 이번 차례는 여기까지. 다음 차례에 이어서 보낸다
            # OK이거나 REJECTED이면 이 묶음을 버퍼에서 뺀다(거절된 묶음은 다시 보내도 소용없다).
            for _ in batch:
                self._buffer.popleft()
            if result is SendResult.REJECTED:
                logger.error("전송 거절로 %s건 버림", len(batch))

    def _run(self) -> None:
        """전송 스레드 본체. interval마다 tick을 돌리고, 정지 신호가 오면 마지막으로 한 번 더 시도(final)한다."""
        while not self._stop.is_set():
            started = time.monotonic()
            self._safe_tick(final=False)
            # tick에 걸린 시간만큼 줄여 전송 주기를 일정하게 맞춘다. 정지 신호가 오면 바로 깨어난다.
            self._stop.wait(max(0.0, self._interval - (time.monotonic() - started)))
        # 루프를 빠져나온 뒤 마지막 한 번: 남은 결과를 보내 본다.
        self._safe_tick(final=True)

    def _safe_tick(self, final: bool) -> None:
        """tick에서 난 예외를 삼켜 전송 스레드가 죽지 않게 한다."""
        try:
            self.tick(final)
        except Exception:
            # 결과는 버퍼에 남아 있다. 스레드가 죽으면 전송이 멈추므로 로그만 남기고 이어 간다.
            logger.exception("결과 전송 중 오류")

    def _absorb(self) -> None:
        """큐의 결과를 모두 꺼내 버퍼 뒤에 잇는다.

        보낼 수 없는 결과는 거르고, 보관 시간이 지난 것과 버퍼 상한을 넘은 것을 버린다.
        """
        unusable = 0
        # 이보다 먼 미래의 at은 시계 오류로 본다.
        latest = self._clock() + FUTURE_LIMIT_SEC
        for result in self._queue.drain():
            epoch = _epoch_of(result)
            # 하루 넘게 미래인 시각은 오류다(만료되지 않아 버퍼에 계속 남는다)
            if epoch is None or epoch > latest or not _serializable(result):
                unusable += 1
                continue
            # (at의 epoch 초, 결과)로 보관한다. epoch는 만료 판정에 쓴다.
            self._buffer.append((epoch, result))
        if unusable:
            logger.warning(
                "시각을 읽을 수 없거나 보낼 수 없는 형식이라 %s건 버림", unusable
            )
        # 만료된 것을 먼저 걸러야 상한 때문에 멀쩡한 결과가 밀려나지 않는다.
        self._expire()
        # 버퍼 상한을 넘은 만큼 가장 오래된 것부터 버린다(메모리 보호).
        overflow = len(self._buffer) - self._max_buffer
        if overflow > 0:
            for _ in range(overflow):
                self._buffer.popleft()
            logger.warning("보관 버퍼가 가득 차 가장 오래된 %s건 버림", overflow)

    def _expire(self) -> None:
        """보관 시간(retry_buffer_sec)이 지난 결과를 버퍼에서 버린다."""
        cutoff = self._clock() - self._retry_buffer_sec
        # 만료된 것이 없으면 버퍼를 다시 만들지 않는다(매 차례의 비용을 줄인다).
        if not any(epoch < cutoff for epoch, _ in self._buffer):
            return
        # 기준 시각 이상인 것만 순서를 유지한 채 남긴다.
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
            # 시간대가 없는 시각은 어느 시각인지 알 수 없어 쓸 수 없다.
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
