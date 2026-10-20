import logging
import threading
from pathlib import Path

from white_eyes_flask.analysis.detector import Detector, YoloCrowdDetector

logger = logging.getLogger(__name__)


class DetectorUnavailableError(Exception):
    """가중치가 없거나 불러오지 못해 탐지기를 쓸 수 없음."""


class LazyDetectorProvider:
    """탐지 모델을 처음 쓸 때 한 번만 불러오고 모든 캠이 공유하게 한다.

    불러오기에 실패해도 실패를 기억하지 않는다. 가중치 파일을 고치면 다음 시작
    때 다시 시도한다.
    """

    def __init__(self, weights: str | None, imgsz: int, conf_thres: float) -> None:
        self._weights = weights
        self._imgsz = imgsz
        self._conf_thres = conf_thres
        self._lock = threading.Lock()
        self._detector: Detector | None = None

    def __call__(self) -> Detector:
        with self._lock:
            if self._detector is not None:
                return self._detector
            if not self._weights or not Path(self._weights).is_file():
                raise DetectorUnavailableError("가중치 파일이 없음(YOLO_WEIGHTS)")
            try:
                self._detector = YoloCrowdDetector.from_weights(
                    self._weights, self._imgsz, self._conf_thres
                )
            except Exception as error:
                logger.exception("탐지 모델 불러오기 실패")
                raise DetectorUnavailableError("가중치를 불러오지 못함") from error
            return self._detector
