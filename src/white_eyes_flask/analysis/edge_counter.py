import threading
from dataclasses import dataclass
from typing import Mapping

DIRECTIONS = ("N", "E", "S", "W")
# 모서리에서 깊이가 같으면 남북을 동서보다 먼저 본다.
_TIE_ORDER = {"N": 0, "S": 1, "W": 2, "E": 3}


def band_of(cx: float, cy: float, margin: float) -> str | None:
    """중심점이 바깥 띠(북·남·서·동)에 있으면 그 방향, 안쪽이면 None.

    좌표는 0~1로 자른다. 띠 경계(margin, 1 - margin) 위는 안쪽이다.
    모서리처럼 두 띠에 걸치면 띠 안으로 더 깊이 들어간 쪽을 고른다.
    """
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    depths: dict[str, float] = {}
    if cy < margin:
        depths["N"] = margin - cy
    if cy > 1 - margin:
        depths["S"] = cy - (1 - margin)
    if cx < margin:
        depths["W"] = margin - cx
    if cx > 1 - margin:
        depths["E"] = cx - (1 - margin)
    if not depths:
        return None
    # 0.98 - 0.95와 0.05 - 0.02처럼 수학으로는 같은 깊이가 부동소수점으로는 다를 수
    # 있어서, 반올림한 값으로 비교해야 동률 규칙이 맞게 적용된다.
    return max(depths, key=lambda d: (round(depths[d], 9), -_TIE_ORDER[d]))


@dataclass
class _Person:
    entry: str | None  # 처음 본 곳이 바깥 띠면 그 방향, 안쪽이면 None
    band: str | None  # 지금 있는 띠(안쪽이면 None)
    leaving: str | None  # 안쪽에서 띠로 나간 방향(나가는 중)
    last_seen: int


class EdgeCounter:
    """캠 하나의 유입/유출 판정(FNC-S-004). 캠마다 따로 만든다.

    프레임마다 update()에 추적 번호별 중심점을 넣고, 1초 결과를 만들 때
    pop_counts()로 그동안 쌓인 방향별 유입·유출 수를 받아 간다.
    추적 번호가 없는 박스는 호출하는 쪽에서 빼고 넣는다.
    """

    def __init__(self, margin: float, lost_frames: int) -> None:
        """margin: 판정선 위치(EDGE_MARGIN_RATIO), lost_frames: 끊김으로 확정하는 프레임 수(ANALYZE_FPS)."""
        self._margin = margin
        self._lost_frames = lost_frames
        self._people: dict[int, _Person] = {}
        self._frame = -1  # 첫 프레임 번호는 0부터 받을 수 있다
        self._counts = self._zero()
        self._lock = threading.Lock()

    def update(
        self,
        points: Mapping[int, tuple[float, float]],
        frame_no: int | None = None,
    ) -> None:
        """한 프레임의 {추적 번호: (cx, cy)}를 넣는다.

        frame_no는 분석 슬롯 번호다. 오류로 건너뛴 슬롯이 있으면 그만큼 시간이
        지난 것으로 센다. 주지 않으면 호출 순서대로 1씩 올린다.
        """
        with self._lock:
            if frame_no is None:
                frame_no = self._frame + 1
            elif frame_no <= self._frame:
                raise ValueError(
                    f"프레임 번호는 늘어나야 합니다: {frame_no} <= {self._frame}"
                )
            self._frame = frame_no
            self._expire_missing(points)
            for track_id, (cx, cy) in points.items():
                self._observe(track_id, band_of(cx, cy, self._margin))

    def pop_counts(self) -> dict[str, dict[str, int]]:
        """지난 호출 이후 쌓인 {"in": {N,E,S,W}, "out": {N,E,S,W}}를 돌려주고 0으로 되돌린다."""
        with self._lock:
            counts, self._counts = self._counts, self._zero()
            return counts

    def _observe(self, track_id: int, band: str | None) -> None:
        person = self._people.get(track_id)
        if person is not None and self._missed(person) - 1 >= self._lost_frames:
            # 지금 프레임 전에 이미 lost_frames만큼 안 보였던 사람이 같은 번호로 다시 나타남:
            # 이전 사람은 확정하고 새 사람으로 본다.
            self._finish(track_id)
            person = None
        if person is None:
            self._people[track_id] = _Person(band, band, None, self._frame)
            return
        if person.band is not None and band is None:
            # 띠 -> 안쪽: 처음 본 띠와 같은 방향에서 들어왔을 때만 유입
            if person.entry == person.band:
                self._counts["in"][person.band] += 1
                person.entry = None
            person.leaving = None  # 나가던 중이었다면 돌아온 것
        elif person.band is None and band is not None:
            person.leaving = band  # 안쪽 -> 띠: 나가는 중
        elif person.band is not None and band is not None and person.leaving:
            person.leaving = band  # 나가는 중에 다른 띠로 옮기면 마지막 방향
        person.band = band
        person.last_seen = self._frame

    def _expire_missing(self, points: Mapping[int, tuple[float, float]]) -> None:
        lost = [
            track_id
            for track_id, person in self._people.items()
            if track_id not in points and self._missed(person) >= self._lost_frames
        ]
        for track_id in lost:
            self._finish(track_id)

    def _missed(self, person: _Person) -> int:
        """마지막으로 본 프레임부터 지금 프레임까지 지난 프레임 수.

        지금 프레임에 안 보이면 연속으로 안 보인 프레임 수와 같고, 보이면 그 앞에
        안 보인 프레임 수 + 1이다.
        """
        return self._frame - person.last_seen

    def _finish(self, track_id: int) -> None:
        person = self._people.pop(track_id)
        if person.leaving:
            self._counts["out"][person.leaving] += 1

    @staticmethod
    def _zero() -> dict[str, dict[str, int]]:
        return {kind: {d: 0 for d in DIRECTIONS} for kind in ("in", "out")}
