import threading
from enum import StrEnum
from typing import Protocol


class PlayOutcome(StrEnum):
    STOPPED = "STOPPED"  # 정지 요청으로 끝남
    FAILED = "FAILED"  # 오류(ERROR 보고함)


class Player(Protocol):
    def play(
        self, camera_id: int, source_uri: str, stop: threading.Event
    ) -> PlayOutcome: ...


class NullPlayer:
    """재생을 하지 않는다. 정지될 때까지 기다리기만 한다."""

    def play(
        self, camera_id: int, source_uri: str, stop: threading.Event
    ) -> PlayOutcome:
        stop.wait()
        return PlayOutcome.STOPPED
