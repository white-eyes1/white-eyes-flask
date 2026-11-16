"""FNC-S-003 사람 탐지기 (YOLO-CROWD)와 탐지기 공통 약속"""

import logging
import sys
import threading
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

logger = logging.getLogger(__name__)

# 외부 YOLO-CROWD 소스(vendor)의 위치. 이 파일에서 위로 3단계 올라간 프로젝트 루트 아래에 있다
VENDOR_DIR = Path(__file__).resolve().parents[3] / "vendor" / "yolo_crowd"
# NMS(겹친 박스 정리)에서 같은 물체로 볼 겹침(IoU) 기준
IOU_THRES = 0.45
# 입력 크기를 맞출 때 쓰는 최소 stride(배수 단위). 이유는 YoloCrowdDetector.__init__ 참고
MIN_PAD_STRIDE = 32


class Detector(Protocol):
    """탐지기가 지켜야 할 약속(구조적 타입). detect만 있으면 어떤 클래스든 탐지기로 쓸 수 있다."""

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """BGR 프레임(H, W, 3)에서 사람을 찾는다(VisDrone은 몸 박스, YOLO-CROWD는 머리 박스).

        원본 픽셀 좌표의 (N, 5) float32 배열 [x1, y1, x2, y2, conf]를 돌려준다.
        """


def _import_vendor() -> Any:
    """vendor의 YOLO-CROWD 모듈을 불러와 (models.common, utils.datasets, utils.general)을 돌려준다."""
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
        """불러온 모델과 추론 설정을 받는다. iou_thres는 NMS 기준이다."""
        self._model = model
        # 입력 가로·세로는 32의 배수여야 한다. YOLO-CROWD는 model.stride.max()가 16이지만
        # 내부에 32배로 줄이는 층이 있어서, 16의 배수만 맞추면 입력 크기에 따라 층을 합칠 때
        # 텐서 크기가 어긋나 오류가 난다(예: 1080x1920 영상을 입력 640으로 돌릴 때).
        self._stride = max(stride, MIN_PAD_STRIDE)
        self._imgsz = imgsz
        self._conf_thres = conf_thres
        self._iou_thres = iou_thres
        # 모델 하나를 여러 캠 스레드가 공유하므로 추론을 한 번에 하나씩만 돌리는 잠금
        self._lock = threading.Lock()
        # letterbox(크기 맞춤)와 NMS·좌표 복원 함수를 vendor에서 가져와 둔다
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
        # 가중치 안의 클래스를 복원하려면 vendor 모듈이 먼저 import되어 있어야 한다
        common, _, _ = _import_vendor()
        # weights_only=False는 파일 안의 코드를 실행할 수 있으므로 신뢰하는 파일에만 쓴다(위 설명 참고)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        # 학습 때 쓴 이동평균(ema) 가중치가 있으면 그것을, 없으면 일반 model을 쓴다.
        # float()로 정밀도를 맞추고, fuse()로 합성곱+정규화 층을 합쳐 추론을 빠르게 하며, eval()로 추론 모드로 바꾼다
        model = (
            checkpoint["ema" if checkpoint.get("ema") else "model"]
            .float()
            .fuse()
            .eval()
        )
        # vendor attempt_load가 하는 후처리를 그대로 따른다: 활성화 함수는 메모리를 아끼는 inplace로 바꾸고,
        # Conv는 옛 버전 호환용 버퍼 집합을 초기화한다
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
        """BGR 프레임에서 머리 박스를 찾아 (N, 5) [x1, y1, x2, y2, conf] 배열(원본 픽셀 좌표)로 돌려준다."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"BGR 프레임(H, W, 3)이어야 합니다: {frame.shape}")
        # letterbox가 크기를 맞추고 남는 곳을 패딩으로 채운다(비율 유지).
        resized = self._datasets.letterbox(frame, self._imgsz, stride=self._stride)[0]
        chw = np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1))  # BGR->RGB
        # 배치 차원을 붙이고 0~255 픽셀을 0~1로 바꿔 모델 입력 모양(1, 3, H, W)을 만든다
        tensor = torch.from_numpy(chw).float().unsqueeze(0) / 255.0
        # 잠금으로 한 번에 하나만 추론하고, inference_mode로 학습용 기록을 끄고 메모리를 아낀다
        with self._lock, torch.inference_mode():
            prediction = self._model(tensor)[0]
            # NMS: 신뢰도 낮은 박스를 버리고 겹친 박스는 하나만 남긴다. 결과는 배치의 첫 장
            det = self._general.non_max_suppression(
                prediction, self._conf_thres, self._iou_thres
            )[0]
            # 탐지가 없으면 모양만 맞춘 빈 배열을 돌려준다
            if det.shape[0] == 0:
                return np.zeros((0, 5), dtype=np.float32)
            # 패딩 제거 -> 비율 복원 -> 원본 경계로 자르기
            self._general.scale_coords(tensor.shape[2:], det[:, :4], frame.shape)
            return det[:, :5].numpy().astype(np.float32)
