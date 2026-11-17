"""FNC-S-003 탐지기 공통 약속"""

from typing import Protocol

import numpy as np


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> np.ndarray:
        """BGR 프레임(H, W, 3)에서 사람을 찾는다(VisDrone은 몸 박스).

        원본 픽셀 좌표의 (N, 5) float32 배열 [x1, y1, x2, y2, conf]를 돌려준다.
        """
