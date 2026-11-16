"""캠 작업의 재생 단계 규칙(Player)과 아무것도 안 하는 기본 구현(NullPlayer).

구현체: 더미는 DummyPlayer, 라이브는 LivePlayer. CameraJob이 이 규칙으로만 부른다.
"""

import threading
from enum import StrEnum
from typing import Protocol


class PlayOutcome(StrEnum):
    """재생 단계가 어떻게 끝났는지. CameraJob이 이 값으로 상태를 정한다."""

    STOPPED = "STOPPED"  # 정지 요청으로 끝남
    FAILED = "FAILED"  # 오류(ERROR 보고함)


class Player(Protocol):
    """재생 구현이 지킬 규칙(Protocol).

    play는 정지 플래그가 설 때까지 돌아오지 않는 것이 보통이며, 정지로 끝나면 STOPPED,
    오류로 끝나면 FAILED를 돌려준다.
    """

    def play(
        self, camera_id: int, source_uri: str, stop: threading.Event
    ) -> PlayOutcome: ...


class NullPlayer:
    """재생을 하지 않는다. 정지될 때까지 기다리기만 한다."""

    def play(
        self, camera_id: int, source_uri: str, stop: threading.Event
    ) -> PlayOutcome:
        # 정지 요청이 올 때까지 아무것도 하지 않고 기다린다(시간 제한 없음).
        stop.wait()
        return PlayOutcome.STOPPED
