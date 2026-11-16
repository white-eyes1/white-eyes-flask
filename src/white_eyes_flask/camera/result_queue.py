"""1초 결과를 전송 전까지 담아 두는 큐(ResultQueue)와 그 입력 규칙(ResultSink).

재생기·분석기가 put으로 넣고, 전송 스레드(ResultSender, FNC-S-006)가 drain으로 꺼내 간다.
"""

import logging
import threading
import time
from collections import deque
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)

_WARN_EVERY = 100  # 가득 찬 상태가 이어져도 로그가 넘치지 않게 일부만 남긴다


class ResultSink(Protocol):
    """결과를 넣는 쪽이 의존하는 규칙(Protocol). put 하나뿐이다."""

    def put(self, item: dict[str, Any]) -> None: ...


class ResultQueue:
    """1초 결과를 담아 두는 전송 큐. 보내는 쪽(FNC-S-006)이 drain()으로 가져간다.

    가득 차면 가장 오래된 결과를 버린다(실시간 통계라 새 결과가 더 중요하다).
    """

    def __init__(
        self, maxlen: int = 5000, clock: Callable[[], float] = time.time
    ) -> None:
        """큐를 만든다.

        :param maxlen: 보관 상한 건수
        :param clock: 현재 시각(epoch 초)을 돌려주는 함수. 시험에서 바꿔 끼운다
        """
        # maxlen을 준 deque는 가득 찬 상태에서 append하면 맨 앞(가장 오래된 것)이 자동으로 밀려난다.
        self._items: deque[dict[str, Any]] = deque(maxlen=maxlen)
        # 결과를 넣는 스레드들(캠 작업)과 꺼내는 스레드(전송)가 함께 쓰므로 잠금으로 지킨다.
        self._lock = threading.Lock()
        self._clock = clock
        # 캠별로 마지막 결과를 넣은 시각(epoch초). 상태 확인(FNC-S-013)이 읽는다
        self._last_put: dict[int, float] = {}
        # 가득 차서 버린 결과의 누적 건수.
        self.dropped = 0

    def last_put_at(self, camera_id: int) -> float | None:
        """그 캠이 마지막으로 결과를 넣은 시각(epoch 초). 넣은 적이 없으면 None."""
        with self._lock:
            return self._last_put.get(camera_id)

    def put(self, item: dict[str, Any]) -> None:
        """결과 1건을 큐에 넣는다. 가득 차 있으면 가장 오래된 것을 버리고 넣는다."""
        with self._lock:
            # 캠별 마지막 입력 시각을 갱신한다(상태 확인 FNC-S-013이 읽는다).
            camera_id = item.get("cameraId")
            if isinstance(camera_id, int):
                self._last_put[camera_id] = self._clock()
            # 가득 찬 deque에 append하면 맨 앞이 조용히 밀려나므로, 그 전에 버려지는 건수를 센다.
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
            # 맨 뒤에 붙인다(오래된 것이 앞쪽). 가득 찼다면 이때 맨 앞 1건이 밀려난다.
            self._items.append(item)

    def drain(self, limit: int | None = None) -> list[dict[str, Any]]:
        """쌓인 결과를 오래된 것부터 limit건(없으면 전부) 꺼내 큐에서 뺀다."""
        with self._lock:
            # limit이 남은 건수보다 크면 있는 만큼만 꺼낸다.
            count = len(self._items) if limit is None else min(limit, len(self._items))
            return [self._items.popleft() for _ in range(count)]

    def __len__(self) -> int:
        """큐에 쌓여 있는 결과 수."""
        with self._lock:
            return len(self._items)
