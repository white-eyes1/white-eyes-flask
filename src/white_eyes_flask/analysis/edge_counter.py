"""가장자리 띠를 기준으로 유입·유출과 체류 시간을 세는 모듈(FNC-S-004, FNC-S-005).

화면 바깥 가장자리에 폭 margin의 띠(북·남·서·동)를 두고, 사람이 띠에서 안쪽으로 들어오면 유입,
안쪽에서 띠로 나간 뒤 사라지면 유출로 센다.
"""

import threading
from dataclasses import dataclass
from typing import Any, Mapping

# 방향 이름: 북(위)·동(오른쪽)·남(아래)·서(왼쪽)
DIRECTIONS = ("N", "E", "S", "W")
# 모서리에서 깊이가 같으면 남북을 동서보다 먼저 본다.
_TIE_ORDER = {"N": 0, "S": 1, "W": 2, "E": 3}


def band_of(cx: float, cy: float, margin: float) -> str | None:
    """중심점이 바깥 띠(북·남·서·동)에 있으면 그 방향, 안쪽이면 None.

    좌표는 0~1로 자른다. 띠 경계(margin, 1 - margin) 위는 안쪽이다.
    모서리처럼 두 띠에 걸치면 띠 안으로 더 깊이 들어간 쪽을 고른다.
    """
    # 화면 밖으로 살짝 나간 좌표도 가장자리로 취급하도록 0~1로 자른다
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    # 걸쳐 있는 띠마다 "띠 안으로 얼마나 깊이 들어갔는지"를 모은다(y가 작을수록 북, x가 클수록 동)
    depths: dict[str, float] = {}
    if cy < margin:
        depths["N"] = margin - cy
    if cy > 1 - margin:
        depths["S"] = cy - (1 - margin)
    if cx < margin:
        depths["W"] = margin - cx
    if cx > 1 - margin:
        depths["E"] = cx - (1 - margin)
    # 어느 띠에도 걸리지 않으면 안쪽
    if not depths:
        return None
    # 0.98 - 0.95와 0.05 - 0.02처럼 수학으로는 같은 깊이가 부동소수점으로는 다를 수
    # 있어서, 반올림한 값으로 비교해야 동률 규칙이 맞게 적용된다.
    return max(depths, key=lambda d: (round(depths[d], 9), -_TIE_ORDER[d]))


@dataclass
class _Person:
    """추적 번호 하나(사람 한 명)의 진행 상태. EdgeCounter 안에서만 쓴다."""

    entry: str | None  # 처음 본 곳이 바깥 띠면 그 방향, 안쪽이면 None
    band: str | None  # 지금 있는 띠(안쪽이면 None)
    leaving: str | None  # 안쪽에서 띠로 나간 방향(나가는 중)
    last_seen: int  # 마지막으로 보인 프레임 번호
    first_seen: int  # 처음 보인 프레임 번호(체류 시간 계산용)


class EdgeCounter:
    """캠 하나의 유입/유출 판정(FNC-S-004). 캠마다 따로 만든다.

    프레임마다 update()에 추적 번호별 중심점을 넣고, 1초 결과를 만들 때
    pop_counts()로 그동안 쌓인 방향별 유입·유출 수와 유출한 사람들의 체류 시간
    (FNC-S-005)을 받아 간다.
    추적 번호가 없는 박스는 호출하는 쪽에서 빼고 넣는다.
    """

    def __init__(
        self,
        margin: float,
        lost_frames: int,
        fps: float | None = None,
        start_frames: int = 0,
    ) -> None:
        """margin: 판정선 위치(EDGE_MARGIN_RATIO), lost_frames: 끊김으로 확정하는 프레임 수(ANALYZE_FPS).

        fps: 초당 분석 프레임 수. 체류 시간(초)을 구할 때 쓰고, 주지 않으면
        lost_frames와 같다(명세에서 끊김은 1초 = ANALYZE_FPS 프레임).
        start_frames: 처음 받은 프레임부터 이만큼의 프레임(update 호출 횟수, 건너뛴 슬롯은
        세지 않음) 안에 처음 보인 사람은 시작 때 이미 화면에 있던 사람으로 보고 entry 없이
        시작한다(더미 영상의 시작 경계).
        """
        # fps를 안 주면 "끊김 1초 = lost_frames 프레임" 규칙에 따라 같은 값으로 본다
        if fps is None:
            fps = lost_frames
        # 체류 시간을 fps로 나누므로 0 이하면 계산이 깨진다
        if fps <= 0:
            raise ValueError(f"fps는 0보다 커야 합니다: {fps}")
        self._margin = margin
        self._lost_frames = lost_frames
        self._fps = fps
        self._start_frames = start_frames
        self._updates = 0  # 지금까지 받은 프레임 수
        # 추적 번호 -> 진행 상태. 지금 추적 중인 사람만 둔다
        self._people: dict[int, _Person] = {}
        self._frame = -1  # 첫 프레임 번호는 0부터 받을 수 있다
        # 다음 pop_counts까지 쌓는 방향별 유입·유출 수와 체류 시간 목록
        self._counts = self._zero()
        # update(분석 스레드)와 pop_counts(결과 만드는 쪽)가 다른 스레드일 수 있어 상태를 잠근다
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
                # 번호가 거꾸로 가면 "안 보인 프레임 수" 계산이 음수가 되어 판정이 깨진다
                raise ValueError(
                    f"프레임 번호는 늘어나야 합니다: {frame_no} <= {self._frame}"
                )
            self._frame = frame_no
            self._updates += 1
            # 먼저 오래 안 보인 사람을 확정(유출 집계)하고, 그다음 이번 프레임에 보인 사람을 갱신한다
            self._expire_missing(points)
            for track_id, (cx, cy) in points.items():
                self._observe(track_id, band_of(cx, cy, self._margin))

    def pop_counts(self) -> dict[str, Any]:
        """지난 호출 이후 쌓인 값을 돌려주고 비운다.

        {"in": {N,E,S,W}, "out": {N,E,S,W}, "dwell_sec": [초, ...]}
        dwell_sec은 그동안 유출이 확정된 사람마다 하나씩, 확정된 순서대로다.
        """
        with self._lock:
            # 쌓인 값을 꺼내면서 새 0 값으로 바꿔 끼운다(꺼낸 값과 이후 집계가 섞이지 않는다)
            counts, self._counts = self._counts, self._zero()
            return counts

    def _observe(self, track_id: int, band: str | None) -> None:
        """이번 프레임에 보인 사람 한 명의 위치(band)로 상태를 갱신하고, 유입이면 집계한다."""
        person = self._people.get(track_id)
        if person is not None and self._missed(person) - 1 >= self._lost_frames:
            # 지금 프레임 전에 이미 lost_frames만큼 안 보였던 사람이 같은 번호로 다시 나타남:
            # 이전 사람은 확정하고 새 사람으로 본다.
            self._finish(track_id)
            person = None
        # 처음 보는 사람이면 등록한다. 시작 직후(start_frames 이내)에 처음 보인 사람은
        # 이미 화면 안에 있던 사람이므로, 띠에서 보였어도 진입 방향(entry)을 기록하지 않는다
        if person is None:
            self._people[track_id] = _Person(
                entry=None if self._updates <= self._start_frames else band,
                band=band,
                leaving=None,
                last_seen=self._frame,
                first_seen=self._frame,
            )
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
        """이번 프레임에 안 보이고 lost_frames 이상 연속으로 안 보인 사람을 끊긴 것으로 확정한다."""
        # 순회 중에 사전을 지우면 안 되므로 대상을 먼저 모은 뒤 지운다
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
        """사람 한 명의 추적을 끝낸다. 나가던 중(leaving)이었다면 그 방향의 유출 1건과 체류 시간을 집계한다."""
        person = self._people.pop(track_id)
        if person.leaving:
            self._counts["out"][person.leaving] += 1
            # 체류(초) = 마지막 본 시각 - 처음 본 시각. 끊긴 뒤 확정까지 기다린 시간은 넣지 않는다.
            frames = person.last_seen - person.first_seen
            self._counts["dwell_sec"].append(round(frames / self._fps, 3))

    @staticmethod
    def _zero() -> dict[str, Any]:
        """방향별 in·out을 0으로, dwell_sec을 빈 목록으로 채운 새 집계 사전을 만든다."""
        counts: dict[str, Any] = {
            kind: {d: 0 for d in DIRECTIONS} for kind in ("in", "out")
        }
        counts["dwell_sec"] = []
        return counts
