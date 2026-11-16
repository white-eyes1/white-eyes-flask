"""FNC-S-003 탐지 모델을 처음 쓸 때 한 번만 불러오는 공급자"""

import logging
import threading
from pathlib import Path

from white_eyes_flask.analysis.aerial_detector import VisDroneDetector
from white_eyes_flask.analysis.boxes import DEFAULT_DEDUPE_OVERLAP
from white_eyes_flask.analysis.detector import Detector, YoloCrowdDetector
from white_eyes_flask.logging_setup import quiet_library_loggers

logger = logging.getLogger(__name__)

# DETECTOR 환경 변수로 고를 수 있는 탐지기 이름(visdrone: 몸 박스, yolo-crowd: 머리 박스)
DETECTOR_VISDRONE = "visdrone"
DETECTOR_YOLO_CROWD = "yolo-crowd"
DETECTOR_KINDS = (DETECTOR_VISDRONE, DETECTOR_YOLO_CROWD)


class DetectorUnavailableError(Exception):
    """가중치가 없거나 불러오지 못해 탐지기를 쓸 수 없음."""


class LazyDetectorProvider:
    """탐지 모델을 처음 쓸 때 한 번만 불러오고 모든 캠이 공유하게 한다.

    불러오기에 실패해도 실패를 기억하지 않는다. 가중치 파일을 고치면 다음 시작
    때 다시 시도한다. 알 수 없는 탐지기 이름은 만들 때 바로 ValueError다.
    """

    def __init__(
        self,
        weights: str | None,
        imgsz: int,
        conf_thres: float,
        kind: str = DETECTOR_VISDRONE,
        dedupe_overlap: float = DEFAULT_DEDUPE_OVERLAP,
    ) -> None:
        """모델은 아직 불러오지 않고 설정만 저장한다. 이름이 잘못되면 서버가 뜰 때 바로 알 수 있게 여기서 검사한다."""
        if kind not in DETECTOR_KINDS:
            raise ValueError(
                f"DETECTOR는 {', '.join(DETECTOR_KINDS)} 중 하나여야 합니다: {kind}"
            )
        self._weights = weights
        self._imgsz = imgsz
        self._conf_thres = conf_thres
        self._kind = kind
        self._dedupe_overlap = dedupe_overlap
        # 여러 캠이 동시에 처음 요청해도 모델을 한 번만 불러오게 하는 잠금
        self._lock = threading.Lock()
        # 불러온 탐지기. None이면 아직 안 불러왔다
        self._detector: Detector | None = None

    def __call__(self) -> Detector:
        """탐지기를 돌려준다. 처음 호출될 때만 불러오고 이후에는 같은 객체를 돌려준다."""
        with self._lock:
            # 이미 불러왔으면 바로 재사용
            if self._detector is not None:
                return self._detector
            # 가중치가 설정되지 않았거나 파일이 없으면 사용 불가로 알린다
            if not self._weights or not Path(self._weights).is_file():
                raise DetectorUnavailableError("가중치 파일이 없음(YOLO_WEIGHTS)")
            try:
                self._detector = self._load(self._weights)
            except Exception as error:
                logger.exception("탐지 모델 불러오기 실패")
                raise DetectorUnavailableError("가중치를 불러오지 못함") from error
            # 모델을 불러오며 라이브러리가 로거를 다시 설정했을 수 있다. 로깅 설정 실패가 모델 로딩 실패로 보고되지 않게 try 밖에 둔다
            quiet_library_loggers()
            return self._detector

    def _load(self, weights: str) -> Detector:
        """설정한 종류에 맞는 탐지기를 가중치 파일에서 불러온다."""
        if self._kind == DETECTOR_YOLO_CROWD:
            return YoloCrowdDetector.from_weights(
                weights, self._imgsz, self._conf_thres
            )
        return VisDroneDetector.from_weights(
            weights, self._imgsz, self._conf_thres, self._dedupe_overlap
        )
