import logging
import sys
import threading
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

logger = logging.getLogger(__name__)

VENDOR_DIR = Path(__file__).resolve().parents[3] / "vendor" / "yolo_crowd"
IOU_THRES = 0.45


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> np.ndarray:
        """BGR 프레임(H, W, 3)에서 사람 머리를 찾는다.

        원본 픽셀 좌표의 (N, 5) float32 배열 [x1, y1, x2, y2, conf]를 돌려준다.
        """


def _import_vendor() -> Any:
    # vendor/yolo_crowd는 `models`, `utils`를 최상위 패키지로 가정한 YOLOv5 코드라서
    # 같은 이름의 다른 패키지가 먼저 잡히지 않도록 경로 맨 앞에 둔다.
    vendor_path = str(VENDOR_DIR)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)
    import models.common
    import models.yolo  # noqa: F401  가중치 파일을 읽을 때 클래스가 필요하다
    import utils.datasets
    import utils.general

    return models.common, utils.datasets, utils.general


class YoloCrowdDetector:
    """YOLO-CROWD(YOLOv5 기반 머리 탐지, 클래스 people 하나) 탐지기.

    모델 하나를 여러 캠 스레드가 공유하므로 추론은 잠금 안에서 한 번에 하나만 돈다.
    """

    def __init__(
        self,
        model: Any,
        stride: int,
        imgsz: int,
        conf_thres: float,
        iou_thres: float = IOU_THRES,
    ) -> None:
        self._model = model
        self._stride = stride
        self._imgsz = imgsz
        self._conf_thres = conf_thres
        self._iou_thres = iou_thres
        self._lock = threading.Lock()
        _, self._datasets, self._general = _import_vendor()

    @classmethod
    def from_weights(
        cls, weights: str, imgsz: int, conf_thres: float
    ) -> "YoloCrowdDetector":
        """가중치 파일에서 불러온다. 신뢰하는 출처의 파일만 쓴다.

        torch 2.6부터 torch.load의 기본값이 weights_only=True라 이 가중치(모델 객체
        전체가 저장됨)를 못 읽는다. vendor의 attempt_load는 수정하지 않으므로 같은
        순서를 여기서 따로 실행한다. 이 호출은 파일 안의 코드를 실행할 수 있다.
        """
        path = Path(weights)
        if not path.is_file():
            raise FileNotFoundError(f"가중치 파일이 없습니다: {weights}")
        common, _, _ = _import_vendor()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = (
            checkpoint["ema" if checkpoint.get("ema") else "model"]
            .float()
            .fuse()
            .eval()
        )
        for module in model.modules():
            if type(module) in (
                torch.nn.Hardswish,
                torch.nn.LeakyReLU,
                torch.nn.ReLU,
                torch.nn.ReLU6,
                torch.nn.SiLU,
            ):
                module.inplace = True
            elif type(module) is common.Conv:
                module._non_persistent_buffers_set = set()
        return cls(model, int(model.stride.max()), imgsz, conf_thres)

    def detect(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"BGR 프레임(H, W, 3)이어야 합니다: {frame.shape}")
        # letterbox가 크기를 맞추고 남는 곳을 패딩으로 채운다(비율 유지).
        resized = self._datasets.letterbox(frame, self._imgsz, stride=self._stride)[0]
        chw = np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1))  # BGR->RGB
        tensor = torch.from_numpy(chw).float().unsqueeze(0) / 255.0
        with self._lock, torch.inference_mode():
            prediction = self._model(tensor)[0]
            det = self._general.non_max_suppression(
                prediction, self._conf_thres, self._iou_thres
            )[0]
            if det.shape[0] == 0:
                return np.zeros((0, 5), dtype=np.float32)
            # 패딩 제거 -> 비율 복원 -> 원본 경계로 자르기
            self._general.scale_coords(tensor.shape[2:], det[:, :4], frame.shape)
            return det[:, :5].numpy().astype(np.float32)
