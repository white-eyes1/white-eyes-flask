import numpy as np

DEDUPE_OVERLAP = 0.25


def sanitize_boxes(boxes: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """탐지 박스 (N, 5) [x1, y1, x2, y2, conf]를 안전한 값만 남긴다.

    유한하지 않은 값이 든 박스를 버리고, 좌표를 프레임 안으로 자른 뒤 폭이나 높이가
    0 이하인 박스를 버린다. 결과는 항상 (M, 5) float32다.
    """
    height, width = shape
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
    boxes = boxes[np.isfinite(boxes).all(axis=1)].copy()
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
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
    if overlap <= 0 or len(boxes) < 2:
        return boxes
    order = np.argsort(-boxes[:, 4], kind="stable")
    kept = np.empty((len(boxes), 4), dtype=np.float32)
    kept_count = 0
    kept_index: list[int] = []
    for index in order:
        x1, y1, x2, y2 = boxes[index, :4]
        if kept_count and _is_duplicate((x1, y1, x2, y2), kept[:kept_count], overlap):
            continue
        kept[kept_count] = (x1, y1, x2, y2)
        kept_count += 1
        kept_index.append(int(index))
    return boxes[sorted(kept_index)]


def _is_duplicate(box, kept: np.ndarray, overlap: float) -> bool:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    kept_cx = (kept[:, 0] + kept[:, 2]) / 2
    kept_cy = (kept[:, 1] + kept[:, 3]) / 2
    box_in_kept = (
        (kept[:, 0] <= cx)
        & (cx <= kept[:, 2])
        & (kept[:, 1] <= cy)
        & (cy <= kept[:, 3])
    )
    kept_in_box = (x1 <= kept_cx) & (kept_cx <= x2) & (y1 <= kept_cy) & (kept_cy <= y2)
    inter_w = np.minimum(x2, kept[:, 2]) - np.maximum(x1, kept[:, 0])
    inter_h = np.minimum(y2, kept[:, 3]) - np.maximum(y1, kept[:, 1])
    inter = np.where((inter_w > 0) & (inter_h > 0), inter_w * inter_h, 0.0)
    smaller = np.minimum(
        (x2 - x1) * (y2 - y1), (kept[:, 2] - kept[:, 0]) * (kept[:, 3] - kept[:, 1])
    )
    smaller = np.maximum(smaller, np.finfo(np.float32).tiny)  # 넓이 0인 박스 방어
    return bool(np.any(box_in_kept | kept_in_box | (inter / smaller >= overlap)))
