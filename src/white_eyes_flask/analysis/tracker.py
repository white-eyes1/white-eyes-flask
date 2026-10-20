import logging
from types import SimpleNamespace

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from white_eyes_flask.analysis.detector import Detector
from white_eyes_flask.analysis.result import FrameResult, TrackedBox

logger = logging.getLogger(__name__)

TRACK_LOW_THRESH = 0.1
MATCH_THRESH = 0.8
PERSON_CLASS = 0


class CameraTracker:
    """캠 하나의 추적기. 캠마다 따로 만들고, 탐지기는 여러 캠이 공유해도 된다.

    같은 캠의 프레임은 한 스레드가 순서대로 넣어야 한다(추적기 상태를 갱신한다).
    """

    def __init__(
        self, detector: Detector, conf_thres: float, track_buffer: int
    ) -> None:
        """track_buffer: 추적이 끊긴 사람을 기억해 두는 프레임 수(1초 = ANALYZE_FPS)."""
        self._detector = detector
        self._tracker = BYTETracker(
            SimpleNamespace(
                track_high_thresh=conf_thres,
                track_low_thresh=TRACK_LOW_THRESH,
                new_track_thresh=conf_thres,
                track_buffer=track_buffer,
                match_thresh=MATCH_THRESH,
                fuse_score=True,
            )
        )

    def update(self, frame: np.ndarray) -> FrameResult | None:
        """프레임 하나를 분석한다. 탐지에 실패하면 로그를 남기고 None을 돌려준다."""
        try:
            detections = self._detector.detect(frame)
        except Exception:
            logger.exception("탐지 오류, 프레임을 건너뜀")
            return None
        height, width = frame.shape[:2]
        detections = _drop_degenerate(detections)
        track_ids = self._track(detections, (height, width))
        boxes = [
            _to_box(row, track_ids.get(i), width, height)
            for i, row in enumerate(detections)
        ]
        # 인원은 추적 결과가 아니라 탐지된 박스 수다(추적 시작 전 박스도 센다).
        return FrameResult(count=len(boxes), boxes=boxes)

    def _track(self, detections: np.ndarray, shape: tuple[int, int]) -> dict[int, int]:
        """탐지 순번 -> 추적 번호. 번호가 아직 없는 탐지는 들어 있지 않다."""
        boxes = Boxes(
            np.column_stack(
                [detections, np.full(len(detections), PERSON_CLASS, dtype=np.float32)]
            ).astype(np.float32),
            shape,
        )
        # 빈 탐지도 넘겨야 추적기가 프레임 진행과 끊긴 사람의 만료를 센다.
        tracked = self._tracker.update(boxes)
        # 열: x1, y1, x2, y2, track_id, score, cls, 탐지 순번
        return {int(row[7]): int(row[4]) for row in tracked}


def _drop_degenerate(detections: np.ndarray) -> np.ndarray:
    widths = detections[:, 2] - detections[:, 0]
    heights = detections[:, 3] - detections[:, 1]
    return detections[(widths > 0) & (heights > 0)]


def _to_box(
    row: np.ndarray, track_id: int | None, width: int, height: int
) -> TrackedBox:
    x1, y1, x2, y2 = (
        float(np.clip(row[0] / width, 0.0, 1.0)),
        float(np.clip(row[1] / height, 0.0, 1.0)),
        float(np.clip(row[2] / width, 0.0, 1.0)),
        float(np.clip(row[3] / height, 0.0, 1.0)),
    )
    return TrackedBox(
        track_id=track_id,
        x=x1,
        y=y1,
        w=x2 - x1,
        h=y2 - y1,
        cx=(x1 + x2) / 2,
        cy=(y1 + y2) / 2,
        conf=float(row[4]),
    )
