"""FNC-S-003 중심점 거리로 같은 사람을 잇는 추적기"""

from dataclasses import dataclass

import numpy as np

MIN_RADIUS = 10.0
MAX_RADIUS = 60.0
RADIUS_SCALE = 1.2
LOST_RADIUS_GROWTH = 0.25
LOST_GROWTH_LIMIT = 4
MAX_RADIUS_FACTOR = 1.5
VELOCITY_SMOOTH = 0.6
SIZE_SMOOTH = 0.7
# 놓친 동안은 속도만큼 다 가지 않았다고 보고 절반만 반영한다(속도 추정이 흔들려도 멀리 튀지 않게).
LOST_DRIFT = 0.5
# 번호를 주는 데 필요한 관측 횟수(연속이 아니라 누적). 번호가 늦어질수록 가장자리로 들어오는
# 사람이 번호를 받기 전에 안쪽 띠로 옮겨 가서 유입을 놓친다(이전 추적기도 두 번째 프레임부터 줬다).
CONFIRM_HITS = 2


@dataclass
class _Track:
    id: int
    x: float  # 마지막으로 실제 본 위치
    y: float
    size: float
    vx: float = 0.0  # 프레임당 이동량
    vy: float = 0.0
    hits: int = 1
    lost: float = 0.0  # 마지막으로 본 뒤 지난 프레임 수


class PointTracker:
    """사람 위치(중심점)만으로 같은 사람을 이어 번호를 붙이는 추적기.

    박스가 겹치는지가 아니라 중심점 사이의 거리로 같은 사람인지 정한다. 드론 영상은
    사람 박스가 작아서 한 걸음만 움직여도 이전 박스와 겹치지 않는데, 이 방식은 그래도
    이어 준다. 좌표 단위는 호출하는 쪽이 정한다(반경 기준값이 이 단위의 픽셀이다).

    방식: 마지막으로 본 위치와 속도로 지금 위치를 예측 → 가까운 쌍부터 1:1로 확정 →
    짝이 없는 트랙은 잠시 두었다가 못 찾은 프레임이 `max_missed`를 넘으면 지움 → 짝이
    없는 탐지는 새 번호. 번호는 이 객체가 살아 있는 동안 계속 늘어나고 다시 쓰지 않는다.
    """

    def __init__(
        self,
        max_missed: int = 3,
        min_radius: float = MIN_RADIUS,
        max_radius: float = MAX_RADIUS,
        confirm_hits: int = CONFIRM_HITS,
    ) -> None:
        self._max_missed = max_missed
        self._min_radius = min_radius
        self._max_radius = max_radius
        self._confirm_hits = confirm_hits
        self._next_id = 1
        self._tracks: dict[int, _Track] = {}

    def update(
        self, centers: np.ndarray, sizes: np.ndarray, dt_frames: float = 1.0
    ) -> list[int | None]:
        """한 프레임의 탐지를 넣고, 탐지마다 추적 번호를 돌려준다.

        centers (N, 2) 중심점, sizes (N,) 사람 크기, dt_frames는 지난 호출 뒤 지나간
        프레임 수(분석에 실패해 건너뛴 프레임이 있으면 1보다 크다). 번호는 `confirm_hits`
        번 관측된 사람에게만 주고 나머지는 None이다. 한 번 번호를 받은 사람은 잠시 놓쳐도
        유지한다. 탐지가 하나도 없는 프레임도 넣어야 놓친 트랙이 시간에 맞춰 지워진다.
        """
        centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
        sizes = np.asarray(sizes, dtype=np.float64).reshape(-1)
        if len(centers) != len(sizes):
            raise ValueError("centers와 sizes의 개수가 다릅니다")
        if not (np.isfinite(centers).all() and np.isfinite(sizes).all()):
            raise ValueError("중심점과 크기는 유한한 값이어야 합니다")
        if dt_frames <= 0:
            raise ValueError("dt_frames는 0보다 커야 합니다")

        tracks = list(self._tracks.values())
        matches = self._match(tracks, centers, dt_frames)
        matched_tracks = {t for t, _ in matches}
        matched_detections = {d for _, d in matches}

        assigned: list[int | None] = [None] * len(centers)
        for track_index, detection_index in matches:
            track = tracks[track_index]
            self._observe(
                track, centers[detection_index], sizes[detection_index], dt_frames
            )
            assigned[detection_index] = track.id
        for track_index, track in enumerate(tracks):
            if track_index in matched_tracks:
                continue
            track.lost += dt_frames
            if track.lost > self._max_missed:
                del self._tracks[track.id]
        # 새 번호는 기존 트랙과의 매칭을 모두 끝낸 뒤에 준다(같은 프레임의 탐지끼리 짝지어지지 않게).
        for detection_index in range(len(centers)):
            if detection_index in matched_detections:
                continue
            track = _Track(
                id=self._next_id,
                x=float(centers[detection_index, 0]),
                y=float(centers[detection_index, 1]),
                size=float(sizes[detection_index]),
            )
            self._next_id += 1
            self._tracks[track.id] = track
            assigned[detection_index] = track.id
        return [
            (
                track_id
                if track_id is not None
                and self._tracks[track_id].hits >= self._confirm_hits
                else None
            )
            for track_id in assigned
        ]

    def _match(
        self, tracks: list[_Track], centers: np.ndarray, dt_frames: float
    ) -> list[tuple[int, int]]:
        if not tracks or not len(centers):
            return []
        # 마지막으로 본 위치 + 속도 × (이번에 지난 프레임 + 놓친 동안 지난 프레임의 절반)
        elapsed = np.array([dt_frames + LOST_DRIFT * t.lost for t in tracks])
        predicted_x = np.array([t.x + t.vx * e for t, e in zip(tracks, elapsed)])
        predicted_y = np.array([t.y + t.vy * e for t, e in zip(tracks, elapsed)])
        dx = predicted_x[:, None] - centers[None, :, 0]
        dy = predicted_y[:, None] - centers[None, :, 1]
        distance_squared = dx * dx + dy * dy
        radius = np.array([self._radius(track) for track in tracks])
        track_index, detection_index = np.nonzero(
            distance_squared <= (radius * radius)[:, None]
        )
        # 가까운 쌍부터. 거리가 같으면 번호가 작은 트랙, 앞선 탐지를 먼저 확정한다.
        order = np.lexsort(
            (
                detection_index,
                track_index,
                distance_squared[track_index, detection_index],
            )
        )
        matches: list[tuple[int, int]] = []
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for k in order:
            t, d = int(track_index[k]), int(detection_index[k])
            if t in used_tracks or d in used_detections:
                continue
            used_tracks.add(t)
            used_detections.add(d)
            matches.append((t, d))
        return matches

    def _radius(self, track: _Track) -> float:
        base = float(
            np.clip(RADIUS_SCALE * track.size, self._min_radius, self._max_radius)
        )
        grown = base * (1.0 + LOST_RADIUS_GROWTH * min(track.lost, LOST_GROWTH_LIMIT))
        return min(self._max_radius * MAX_RADIUS_FACTOR, grown)

    @staticmethod
    def _observe(
        track: _Track, center: np.ndarray, size: float, dt_frames: float
    ) -> None:
        # 속도는 마지막으로 실제 본 위치에서 지금까지 지난 프레임 수로 나눈다.
        elapsed = track.lost + dt_frames
        vx = (float(center[0]) - track.x) / elapsed
        vy = (float(center[1]) - track.y) / elapsed
        if track.hits > 1:
            track.vx = VELOCITY_SMOOTH * track.vx + (1 - VELOCITY_SMOOTH) * vx
            track.vy = VELOCITY_SMOOTH * track.vy + (1 - VELOCITY_SMOOTH) * vy
        else:
            track.vx, track.vy = vx, vy
        track.x, track.y = float(center[0]), float(center[1])
        track.size = SIZE_SMOOTH * track.size + (1 - SIZE_SMOOTH) * float(size)
        track.hits += 1
        track.lost = 0.0
