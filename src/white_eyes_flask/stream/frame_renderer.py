"""FNC-S-011 최신 프레임에 박스·인원수를 그려 JPEG로 만든다."""

import functools

import cv2
import numpy as np

BOX_COLOR = (0, 255, 0)  # BGR 초록
TEXT_COLOR = (0, 255, 0)
LIVE_WIDTH = 960
LIVE_QUALITY = 70
SNAPSHOT_WIDTH = 320
SNAPSHOT_QUALITY = 60


class FrameEncodeError(Exception):
    """프레임을 JPEG로 만들지 못함."""


def render_jpeg(
    image: np.ndarray,
    boxes: np.ndarray,
    person_count: int,
    *,
    width: int,
    quality: int,
) -> bytes:
    """image(BGR)를 가로 width로 줄이고 박스·인원수를 그려 JPEG 바이트로 만든다.

    boxes는 (N, 4 이상) [x1, y1, x2, y2, ...]이고 원본 image 좌표다. 원본은 고치지 않는다.
    원본이 width보다 좁으면 늘리지 않는다.
    """
    height, source_width = image.shape[:2]
    scale = min(1.0, width / source_width)
    if scale < 1.0:
        canvas = cv2.resize(
            image,
            (width, max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        canvas = image.copy()

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, boxes_columns(boxes))
    for x1, y1, x2, y2 in (boxes[:, :4] * scale).round().astype(int):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), BOX_COLOR, 1)
    cv2.putText(
        canvas,
        f"Count: {person_count}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )

    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise FrameEncodeError("JPEG 변환 실패")
    return encoded.tobytes()


@functools.cache
def no_signal_jpeg() -> bytes:
    """첫 프레임이 오기 전(또는 오래 끊긴 동안) 연결을 살려 두려고 보내는 회색 이미지."""
    canvas = np.full((360, 640, 3), 40, dtype=np.uint8)
    cv2.putText(
        canvas,
        "NO SIGNAL",
        (200, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )
    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 60])
    if not ok:
        raise FrameEncodeError("JPEG 변환 실패")
    return encoded.tobytes()


def boxes_columns(boxes: np.ndarray) -> int:
    """박스 배열의 열 수. 비어 있으면 4로 본다."""
    boxes = np.asarray(boxes)
    return boxes.shape[-1] if boxes.ndim == 2 and boxes.size else 4
