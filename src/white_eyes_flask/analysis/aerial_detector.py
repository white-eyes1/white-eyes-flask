"""FNC-S-003 VisDrone 사람 탐지기"""

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

# NMS(겹친 박스 정리)에서 두 박스를 같은 물체로 볼 겹침(IoU) 기준
IOU_THRES = 0.45
# 한 프레임에서 최대 몇 개까지 탐지할지. 군중 장면에서 박스가 잘리지 않게 넉넉히 잡는다
MAX_DET = 3000
# 사람으로 합칠 VisDrone 클래스 이름(모델의 names에 이 이름이 있어야 한다)
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
        """이미 불러온 모델과 추론 설정을 받는다. 사람 클래스가 없는 모델이면 여기서 바로 ValueError다."""
        self._model = model
        self._imgsz = imgsz
        self._conf_thres = conf_thres
        self._dedupe_overlap = dedupe_overlap
        # 추론 때 이 클래스만 남기려고 번호를 미리 구해 둔다
        self._person_classes = _person_class_ids(model)
        # 여러 캠 스레드가 같은 모델을 쓰므로 추론을 한 번에 하나씩만 돌리는 잠금
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
        # YOLO가 모호한 오류를 내기 전에, 파일이 없다는 사실을 먼저 분명히 알린다
        if not Path(weights).is_file():
            raise FileNotFoundError(f"가중치 파일이 없습니다: {weights}")
        return cls(YOLO(str(weights)), imgsz, conf_thres, dedupe_overlap)

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """BGR 프레임에서 사람 박스를 찾아 (N, 5) [x1, y1, x2, y2, conf] 배열(원본 픽셀 좌표)로 돌려준다."""
        # 흑백·알파 채널 등 잘못된 모양의 입력은 모델에 넣기 전에 막는다
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"BGR 프레임(H, W, 3)이어야 합니다: {frame.shape}")
        # 모델을 공유하므로 추론 구간만 잠근다(박스 정리는 잠금 밖에서 해 대기 시간을 줄인다)
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
            # 탐지가 없으면 모양만 맞춘 빈 배열을 돌려줘 호출한 쪽이 따로 분기하지 않게 한다
            if boxes is None or len(boxes) == 0:
                return np.zeros((0, 5), dtype=np.float32)
            # 좌표(xyxy)와 신뢰도를 한 배열로 묶는다. GPU 텐서일 수 있어 cpu()로 내린 뒤 numpy로 바꾼다
            detections = np.column_stack(
                [boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()]
            )
        # 이상한 값 제거와 프레임 밖 좌표 자르기, 그리고 한 사람을 둘로 센 박스 합치기(설정이 켜졌을 때만)
        detections = sanitize_boxes(detections, frame.shape[:2])
        return dedupe_boxes(detections, self._dedupe_overlap)


def _person_class_ids(model: Any) -> list[int]:
    """모델의 클래스 이름에서 사람 클래스 번호를 찾는다. 없으면 다른 모델이므로 바로 실패한다."""
    # model.names는 {번호: 이름} dict이거나 이름 리스트일 수 있어 둘 다 받는다
    raw = model.names
    pairs = raw.items() if isinstance(raw, dict) else enumerate(raw)
    # 이름 -> 번호 로 뒤집어 찾기 쉽게 만든다
    names = {str(name): int(index) for index, name in pairs}
    # VisDrone이 아닌 모델(예: COCO)을 잘못 넣었을 때 조용히 사람 0명으로 동작하지 않게 한다
    missing = [name for name in PERSON_CLASS_NAMES if name not in names]
    if missing:
        raise ValueError(f"사람 클래스가 없는 모델입니다(없는 이름: {missing})")
    return [names[name] for name in PERSON_CLASS_NAMES]
