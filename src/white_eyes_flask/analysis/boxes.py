"""탐지 박스 후처리 도우미. 박스는 (N, 5) [x1, y1, x2, y2, conf] 배열(원본 픽셀 좌표)이다."""

import numpy as np

DEDUPE_OVERLAP = 0.25  # 합치기를 켤 때의 기준
DEFAULT_DEDUPE_OVERLAP = 0  # 기본은 끔(결정 0015)


def sanitize_boxes(boxes: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """탐지 박스 (N, 5) [x1, y1, x2, y2, conf]를 안전한 값만 남긴다.

    유한하지 않은 값이 든 박스를 버리고, 좌표를 프레임 안으로 자른 뒤 폭이나 높이가
    0 이하인 박스를 버린다. 결과는 항상 (M, 5) float32다.
    """
    height, width = shape
    # 빈 입력이나 모양이 어긋난 입력도 항상 (N, 5) float32로 맞춘다
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
    # NaN·무한대가 든 행 제거. 이후 clip·비교가 엉뚱한 값을 만들지 않게 먼저 거른다.
    # 불리언 마스크 인덱싱은 복사본을 만들지만, 아래에서 값을 고치므로 원본 보호를 위해 copy를 명시한다
    boxes = boxes[np.isfinite(boxes).all(axis=1)].copy()
    # x 좌표(0, 2열)는 0~폭, y 좌표(1, 3열)는 0~높이 안으로 자른다
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    # 자르고 나서 폭이나 높이가 0 이하(프레임 밖에 완전히 걸친 박스 등)가 된 박스를 버린다
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[keep]


def dedupe_boxes(boxes: np.ndarray, overlap: float = DEDUPE_OVERLAP) -> np.ndarray:
    """같은 사람을 둘로 센 박스를 합친다.

    사람이 화면에 크게 찍히면 모델이 몸통과 다리를 따로 잡아 한 사람이 둘로 세어진다.
    점수가 높은 박스부터 남기고, 이미 남은 박스와 아래 둘 중 하나에 해당하면 버린다.
    - 한쪽 박스의 중심이 다른 쪽 박스 안에 들어간다
    - 겹친 넓이가 둘 중 작은 박스 넓이의 `overlap` 이상이다
    `overlap`이 0 이하면 합치지 않는다. 점수가 같으면 입력 순서가 앞선 박스를 남긴다.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
    # 기능이 꺼져 있거나(overlap <= 0) 비교할 상대가 없으면 그대로 돌려준다
    if overlap <= 0 or len(boxes) < 2:
        return boxes
    # 신뢰도 높은 순으로 처리한다. stable 정렬이라 점수가 같으면 입력 순서가 앞선 박스가 먼저다
    order = np.argsort(-boxes[:, 4], kind="stable")
    # 지금까지 남긴 박스의 좌표를 쌓아 두는 미리 잡은 버퍼(반복마다 배열을 새로 만들지 않으려는 것)
    kept = np.empty((len(boxes), 4), dtype=np.float32)
    kept_count = 0
    # 남긴 박스의 원래 인덱스. 마지막에 원래 순서로 되돌리는 데 쓴다
    kept_index: list[int] = []
    for index in order:
        x1, y1, x2, y2 = boxes[index, :4]
        # 이미 남은 박스 중 하나와 같은 사람으로 판정되면 이 박스(점수가 더 낮다)는 버린다
        if kept_count and _is_duplicate((x1, y1, x2, y2), kept[:kept_count], overlap):
            continue
        kept[kept_count] = (x1, y1, x2, y2)
        kept_count += 1
        kept_index.append(int(index))
    # 점수 순이 아니라 입력 순서대로 돌려줘 결과가 입력과 같은 순서를 유지하게 한다
    return boxes[sorted(kept_index)]


def _is_duplicate(box, kept: np.ndarray, overlap: float) -> bool:
    """box가 이미 남긴 박스(kept, (K, 4)) 중 하나와 같은 사람인지 판정한다. 기준은 dedupe_boxes 설명과 같다."""
    x1, y1, x2, y2 = box
    # 새 박스의 중심, 그리고 남긴 박스들의 중심
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    kept_cx = (kept[:, 0] + kept[:, 2]) / 2
    kept_cy = (kept[:, 1] + kept[:, 3]) / 2
    # 조건 1-a: 새 박스의 중심이 남긴 박스 안에 있다
    box_in_kept = (
        (kept[:, 0] <= cx)
        & (cx <= kept[:, 2])
        & (kept[:, 1] <= cy)
        & (cy <= kept[:, 3])
    )
    # 조건 1-b: 반대로 남긴 박스의 중심이 새 박스 안에 있다
    kept_in_box = (x1 <= kept_cx) & (kept_cx <= x2) & (y1 <= kept_cy) & (kept_cy <= y2)
    # 겹친 영역의 폭·높이. 음수면 겹치지 않는 것이라 겹친 넓이를 0으로 둔다
    inter_w = np.minimum(x2, kept[:, 2]) - np.maximum(x1, kept[:, 0])
    inter_h = np.minimum(y2, kept[:, 3]) - np.maximum(y1, kept[:, 1])
    inter = np.where((inter_w > 0) & (inter_h > 0), inter_w * inter_h, 0.0)
    # 조건 2의 분모: 둘 중 작은 박스의 넓이. 일반 IoU가 아니라 "작은 쪽 기준 겹침"을 쓰는 이유는,
    # 큰 박스(몸통) 안에 작은 박스(다리)가 들어간 경우 IoU는 작아도 같은 사람이기 때문이다
    smaller = np.minimum(
        (x2 - x1) * (y2 - y1), (kept[:, 2] - kept[:, 0]) * (kept[:, 3] - kept[:, 1])
    )
    smaller = np.maximum(smaller, np.finfo(np.float32).tiny)  # 넓이 0인 박스 방어
    return bool(np.any(box_in_kept | kept_in_box | (inter / smaller >= overlap)))
