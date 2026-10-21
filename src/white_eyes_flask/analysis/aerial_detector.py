import threading
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from white_eyes_flask.analysis.boxes import (
    DEFAULT_DEDUPE_OVERLAP,
    dedupe_boxes,
    sanitize_boxes,
)

IOU_THRES = 0.45
MAX_DET = 3000
PERSON_CLASS_NAMES = ("pedestrian", "people")


class VisDroneDetector:
    """VisDrone YOLOv8m(위에서 내려다본 사람 몸 박스) 탐지기.

    모델 하나를 여러 캠 스레드가 공유하므로 추론은 잠금 안에서 한 번에 하나만 돈다.
    VisDrone의 pedestrian(걷는 사람)과 people(그 밖의 사람) 두 클래스를 사람 하나로 합친다.
    """

    def __init__(
        self,
        model: Any,
        imgsz: int,
        conf_thres: float,
        dedupe_overlap: float = DEFAULT_DEDUPE_OVERLAP,
    ) -> None:
        self._model = model
        self._imgsz = imgsz
        self._conf_thres = conf_thres
        self._dedupe_overlap = dedupe_overlap
        self._person_classes = _person_class_ids(model)
        self._lock = threading.Lock()

    @classmethod
    def from_weights(
        cls,
        weights: str,
        imgsz: int,
        conf_thres: float,
        dedupe_overlap: float = DEFAULT_DEDUPE_OVERLAP,
    ) -> "VisDroneDetector":
        """가중치 파일에서 불러온다. 신뢰하는 출처의 파일만 쓴다(파일 안의 코드가 실행될 수 있다)."""
        if not Path(weights).is_file():
            raise FileNotFoundError(f"가중치 파일이 없습니다: {weights}")
        return cls(YOLO(str(weights)), imgsz, conf_thres, dedupe_overlap)

    def detect(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"BGR 프레임(H, W, 3)이어야 합니다: {frame.shape}")
        with self._lock:
            result = self._model.predict(
                frame,
                imgsz=self._imgsz,
                conf=self._conf_thres,
                iou=IOU_THRES,
                classes=self._person_classes,
                agnostic_nms=True,  # 같은 자리의 두 클래스를 한 사람으로
                max_det=MAX_DET,
                device="cpu",
                verbose=False,
            )[0]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                return np.zeros((0, 5), dtype=np.float32)
            detections = np.column_stack(
                [boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()]
            )
        detections = sanitize_boxes(detections, frame.shape[:2])
        return dedupe_boxes(detections, self._dedupe_overlap)


def _person_class_ids(model: Any) -> list[int]:
    """모델의 클래스 이름에서 사람 클래스 번호를 찾는다. 없으면 다른 모델이므로 바로 실패한다."""
    raw = model.names
    pairs = raw.items() if isinstance(raw, dict) else enumerate(raw)
    names = {str(name): int(index) for index, name in pairs}
    missing = [name for name in PERSON_CLASS_NAMES if name not in names]
    if missing:
        raise ValueError(f"사람 클래스가 없는 모델입니다(없는 이름: {missing})")
    return [names[name] for name in PERSON_CLASS_NAMES]
