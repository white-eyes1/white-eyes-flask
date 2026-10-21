from dataclasses import dataclass


@dataclass(frozen=True)
class TrackedBox:
    """한 프레임의 사람 박스(VisDrone은 몸 전체). 좌표는 모두 영상 크기 대비 0~1 비율이다.

    x, y는 박스의 왼쪽 위 모서리, cx, cy는 중심점이다.
    track_id가 None이면 아직 추적 번호가 없는 박스(인원에는 넣고 이동 판정에서는 뺀다).
    """

    track_id: int | None
    x: float
    y: float
    w: float
    h: float
    cx: float
    cy: float
    conf: float


@dataclass(frozen=True)
class FrameResult:
    """프레임 하나의 분석 결과. count는 박스 수(추정 인원)다."""

    count: int
    boxes: list[TrackedBox]
