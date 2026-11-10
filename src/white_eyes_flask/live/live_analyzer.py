"""FNC-S-007 라이브 프레임 분석: 탐지·추적·유입/유출 -> 1초 결과"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from white_eyes_flask.analysis.detector import Detector
from white_eyes_flask.analysis.edge_counter import DIRECTIONS, EdgeCounter
from white_eyes_flask.analysis.result import FrameResult
from white_eyes_flask.analysis.tracker import CameraTracker
from white_eyes_flask.camera.kst import format_at
from white_eyes_flask.camera.result_queue import ResultSink

logger = logging.getLogger(__name__)


@dataclass
class _Second:
    """한 초 동안 모인 값. 프레임을 읽은 시각이 속한 초(epoch)로 나눈다."""

    at: int
    total: int = 0
    frames: int = 0
    in_: dict[str, int] = field(default_factory=lambda: dict.fromkeys(DIRECTIONS, 0))
    out: dict[str, int] = field(default_factory=lambda: dict.fromkeys(DIRECTIONS, 0))
    dwell_sec: list[float] = field(default_factory=list)

    def add_counts(self, counts: dict[str, Any]) -> None:
        for direction in DIRECTIONS:
            self.in_[direction] += counts["in"][direction]
            self.out[direction] += counts["out"][direction]
        self.dwell_sec.extend(counts["dwell_sec"])


class LiveAnalyzer:
    """캠 하나의 라이브 분석. 한 스레드가 프레임을 순서대로 넣는다.

    분석한 프레임의 읽은 시각(epoch 초)으로 초를 나눠, 초가 바뀔 때마다 그 초의 결과를
    전송 큐에 넣는다(더미 재생이 넣는 것과 같은 형식). 분석한 프레임이 없던 초는 보내지
    않는다(끊긴 동안을 0명으로 속이지 않고 빈 구간으로 둔다, FNC-S-006).
    스트림이 끊기면 end_session()으로 나가던 사람을 확정하고, 다시 연결되면 추적기를
    새로 시작한다(끊긴 사이 번호가 옛 사람과 이어지지 않게).
    """

    def __init__(
        self,
        camera_id: int,
        detector: Detector,
        sink: ResultSink,
        *,
        fps: int,
        conf_thres: float,
        edge_margin: float,
    ) -> None:
        self._camera_id = camera_id
        self._detector = detector
        self._sink = sink
        self._fps = fps
        self._conf_thres = conf_thres
        self._edge_margin = edge_margin
        self._tracker: CameraTracker | None = None
        self._edge: EdgeCounter | None = None
        self._last_slot = -1
        self._current: _Second | None = None

    def analyze(self, frame: np.ndarray, at: float) -> FrameResult | None:
        """프레임 하나를 분석한다. 탐지에 실패한 프레임은 None(추적기가 로그를 남김)."""
        self._roll(math.floor(at))
        if self._tracker is None or self._edge is None:
            self._tracker = CameraTracker(
                self._detector, self._conf_thres, track_buffer=self._fps
            )
            self._edge = EdgeCounter(
                self._edge_margin, lost_frames=self._fps, fps=self._fps
            )
            self._last_slot = -1
        result = self._tracker.update(frame)
        if result is None:
            return None
        # 분석 슬롯은 실제 시각 x fps. 프레임이 밀려도 번호는 늘어나야 한다(EdgeCounter 규칙).
        slot = max(self._last_slot + 1, math.floor(at * self._fps))
        self._last_slot = slot
        points = {
            box.track_id: (box.cx, box.cy)
            for box in result.boxes
            if box.track_id is not None
        }
        self._edge.update(points, frame_no=slot)
        if self._current is not None:
            self._current.total += result.count
            self._current.frames += 1
        return result

    def end_session(self) -> None:
        """스트림이 끊겼다. 끝난 뒤 1초 동안 안 보인 것으로 보고 나가던 사람을 확정한다."""
        if self._edge is not None and self._last_slot >= 0:
            self._edge.update({}, frame_no=self._last_slot + self._fps)
            self._collect_counts()
        self._tracker = None
        self._edge = None

    def flush_idle(self, now: float) -> None:
        """프레임이 안 들어오는 동안에도 지난 초의 결과를 보낸다(분석이 늦어 1초의 여유를 둔다)."""
        if self._current is not None and math.floor(now) > self._current.at + 1:
            self._collect_counts()
            self._emit()

    def close(self) -> None:
        self.end_session()
        if self._current is not None:
            self._emit()

    def _roll(self, second: int) -> None:
        """프레임이 속한 초가 바뀌었으면 이전 초를 보내고 새 초를 시작한다."""
        if self._current is None:
            self._current = _Second(second)
            return
        if second <= self._current.at:
            return  # 같은 초(시계가 되돌아간 경우도 지금 초로 둔다)
        self._collect_counts()
        self._emit()
        self._current = _Second(second)

    def _collect_counts(self) -> None:
        """추적기에 쌓인 유입/유출을 지금 초에 넣는다.

        지금 초가 없으면(이미 보냈거나 프레임이 없던 때) 다른 시각에 붙이지 않고 버린다.
        끊긴 동안은 빈 구간이라는 규칙과 같다. close()도 같은 경로라 같은 규칙이다.
        """
        if self._edge is None:
            return
        counts = self._edge.pop_counts()
        if self._current is not None:
            self._current.add_counts(counts)
        elif counts["dwell_sec"] or any(
            counts[kind][d] for kind in ("in", "out") for d in DIRECTIONS
        ):
            logger.info(
                "보낼 초가 없어 확정된 유입/유출을 버림 cameraId=%s in=%s out=%s",
                self._camera_id,
                counts["in"],
                counts["out"],
            )

    def _emit(self) -> None:
        current, self._current = self._current, None
        if current is None or current.frames == 0:
            return
        self._sink.put(
            {
                "cameraId": self._camera_id,
                "at": format_at(current.at),
                # 그 초 프레임들의 인원 평균을 반올림한다(0.5는 올림). 더미 분석과 같은 규칙
                "personCount": int(current.total / current.frames + 0.5),
                "in": current.in_,
                "out": current.out,
                "dwellSec": current.dwell_sec,
            }
        )
