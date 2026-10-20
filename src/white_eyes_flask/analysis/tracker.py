import logging

import numpy as np

from white_eyes_flask.analysis.detector import Detector
from white_eyes_flask.analysis.point_tracker import PointTracker
from white_eyes_flask.analysis.result import FrameResult, TrackedBox

logger = logging.getLogger(__name__)

# PointTracker의 반경 기준값(픽셀)은 이 폭의 영상을 기준으로 정해져 있다. 4K 영상도,
# 작은 영상도 같은 비율로 맞추려고 중심점과 크기를 이 폭으로 환산해서 넣는다.
REFERENCE_WIDTH = 1000.0
MIN_PERSON_SIZE = 6.0
MIN_LOST_FRAMES = 3


class CameraTracker:
    """캠 하나의 추적기. 캠마다 따로 만들고, 탐지기는 여러 캠이 공유해도 된다.

    같은 캠의 프레임은 한 스레드가 순서대로 넣어야 한다(추적기 상태를 갱신한다).
    """

    def __init__(
        self, detector: Detector, conf_thres: float, track_buffer: int
    ) -> None:
        """track_buffer: 추적이 끊긴 사람을 기억해 두는 프레임 수의 기준(1초 = ANALYZE_FPS).

        실제로는 그 절반(최소 3프레임)을 넘게 못 찾으면 번호를 잊는다. 1초 분석 프레임이
        5장이면 3장이다. conf_thres 미만의 탐지는 쓰지 않는다.
        """
        self._detector = detector
        self._conf_thres = conf_thres
        self._tracker = PointTracker(max_missed=max(MIN_LOST_FRAMES, track_buffer // 2))
        self._failed_frames = 0

    def update(self, frame: np.ndarray) -> FrameResult | None:
        """프레임 하나를 분석한다. 탐지에 실패하면 로그를 남기고 None을 돌려준다.

        실패한 프레임도 시간은 흘렀으므로, 다음 성공한 프레임에서 그만큼을 추적기에 알린다
        (그래야 그 사이 움직인 사람을 같은 번호로 잇고, 놓친 사람의 만료도 맞는다).
        """
        try:
            detections = self._detector.detect(frame)
        except Exception:
            logger.exception("탐지 오류, 프레임을 건너뜀")
            self._failed_frames += 1
            return None
        height, width = frame.shape[:2]
        detections = _usable(detections, self._conf_thres)
        track_ids = self._track(detections, width)
        boxes = [
            _to_box(row, track_id, width, height)
            for row, track_id in zip(detections, track_ids)
        ]
        # 인원은 추적 결과가 아니라 탐지된 박스 수다(번호가 아직 없는 박스도 센다).
        return FrameResult(count=len(boxes), boxes=boxes)

    def _track(self, detections: np.ndarray, width: int) -> list[int | None]:
        """탐지마다 추적 번호. 번호가 아직 없으면 None이다."""
        scale = REFERENCE_WIDTH / width
        centers = np.column_stack(
            [
                (detections[:, 0] + detections[:, 2]) / 2,
                (detections[:, 1] + detections[:, 3]) / 2,
            ]
        )
        sizes = np.maximum(
            MIN_PERSON_SIZE,
            0.5
            * (
                (detections[:, 2] - detections[:, 0])
                + (detections[:, 3] - detections[:, 1])
            )
            * scale,
        )
        # 탐지가 없는 프레임도 넘겨야 추적기가 프레임 진행과 놓친 사람의 만료를 센다.
        dt_frames = 1 + self._failed_frames
        self._failed_frames = 0
        return self._tracker.update(centers * scale, sizes, dt_frames)


def _usable(detections: np.ndarray, conf_thres: float) -> np.ndarray:
    """신뢰도가 낮거나 폭·높이가 0 이하인 탐지를 뺀다."""
    widths = detections[:, 2] - detections[:, 0]
    heights = detections[:, 3] - detections[:, 1]
    keep = (detections[:, 4] >= conf_thres) & (widths > 0) & (heights > 0)
    return detections[keep]


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
