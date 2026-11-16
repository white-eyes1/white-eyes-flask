"""FNC-S-011 최신 프레임에 박스·인원수를 그려 JPEG로 만든다."""

import functools

import cv2
import numpy as np

BOX_COLOR = (0, 255, 0)  # BGR 초록
TEXT_COLOR = (0, 255, 0)
# 실시간 화면(MJPEG)의 가로 크기(px)와 JPEG 품질(1~100)
LIVE_WIDTH = 960
LIVE_QUALITY = 70
# 카드 미리보기(스냅샷)는 목록에 작게 여러 장 뜨므로 더 작고 낮은 품질로 만든다
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
    # 줄일 비율. min(1.0, ...)으로 원본보다 키우지는 않는다
    scale = min(1.0, width / source_width)
    if scale < 1.0:
        # 축소에는 INTER_AREA가 화질이 좋다. 높이는 비율을 유지하되 최소 1px
        canvas = cv2.resize(
            image,
            (width, max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        # 그리기는 canvas에만 하고 원본(여러 스레드가 공유하는 프레임)은 건드리지 않도록 복사한다
        canvas = image.copy()

    # 박스가 없을 때도 (0, 열 수) 모양이 되게 맞춘다. 좌표는 축소 비율을 곱해 canvas 기준으로 바꾼다
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, boxes_columns(boxes))
    for x1, y1, x2, y2 in (boxes[:, :4] * scale).round().astype(int):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), BOX_COLOR, 1)
    # 왼쪽 위에 인원수를 적는다
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

    # 메모리 상에서 JPEG로 인코딩한다(파일을 거치지 않는다)
    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise FrameEncodeError("JPEG 변환 실패")
    return encoded.tobytes()


# 매번 같은 이미지이므로 한 번 만든 결과를 재사용한다
@functools.cache
def no_signal_jpeg() -> bytes:
    """첫 프레임이 오기 전(또는 오래 끊긴 동안) 연결을 살려 두려고 보내는 회색 이미지."""
    # 640x360, 어두운 회색(40) 단색 바탕에 문구를 얹는다
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
    # 빈 배열은 열 수를 알 수 없어 4(x1, y1, x2, y2)로 가정한다. reshape(-1, 0) 오류를 막는 용도다
    return boxes.shape[-1] if boxes.ndim == 2 and boxes.size else 4
