import logging
import threading
import time
from collections import deque
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)

_WARN_EVERY = 100  # 가득 찬 상태가 이어져도 로그가 넘치지 않게 일부만 남긴다


class ResultSink(Protocol):
    def put(self, item: dict[str, Any]) -> None: ...


class ResultQueue:
    """1초 결과를 담아 두는 전송 큐. 보내는 쪽(FNC-S-006)이 drain()으로 가져간다.

    가득 차면 가장 오래된 결과를 버린다(실시간 통계라 새 결과가 더 중요하다).
    """

    def __init__(
        self, maxlen: int = 5000, clock: Callable[[], float] = time.time
    ) -> None:
        self._items: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._clock = clock
        # 캠별로 마지막 결과를 넣은 시각(epoch초). 상태 확인(FNC-S-013)이 읽는다
        self._last_put: dict[int, float] = {}
        self.dropped = 0

    def last_put_at(self, camera_id: int) -> float | None:
        with self._lock:
            return self._last_put.get(camera_id)

    def put(self, item: dict[str, Any]) -> None:
        with self._lock:
            camera_id = item.get("cameraId")
            if isinstance(camera_id, int):
                self._last_put[camera_id] = self._clock()
            if (
                self._items.maxlen is not None
                and len(self._items) == self._items.maxlen
            ):
                self.dropped += 1
                if self.dropped == 1 or self.dropped % _WARN_EVERY == 0:
                    logger.warning(
                        "전송 큐가 가득 차 가장 오래된 결과를 버림(누적 %s건)",
                        self.dropped,
                    )
            self._items.append(item)

    def drain(self, limit: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            count = len(self._items) if limit is None else min(limit, len(self._items))
            return [self._items.popleft() for _ in range(count)]

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
