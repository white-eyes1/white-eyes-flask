"""FNC-S-008 라이브 좌표 파일(.json.gz). 형식은 conventions '좌표 파일 형식'."""

import gzip
import json
import os
from pathlib import Path

from white_eyes_flask.analysis.result import TrackedBox

BOX_FILE_VERSION = 1
_DIGITS = 4


def _ratio(value: float) -> float:
    """좌표는 영상 크기 대비 0~1 비율, 소수 4자리."""
    return round(min(1.0, max(0.0, value)), _DIGITS)


def _row(box: TrackedBox) -> list:
    return [box.track_id, _ratio(box.x), _ratio(box.y), _ratio(box.w), _ratio(box.h)]


def write_box_file(
    path: Path, fps: int, frames: list[tuple[float, list[TrackedBox]]]
) -> None:
    """frames는 (그 분 파일 시작부터의 초, 박스 목록). 임시 이름에 쓴 뒤 올려 반쪽 파일을 남기지 않는다."""
    # 박스가 거꾸로 도착해도 다시보기가 "t ≤ 재생 위치인 마지막 프레임"을 찾을 수 있게 t 순서로 쓴다
    ordered = sorted(frames, key=lambda frame: frame[0])
    body = {
        "version": BOX_FILE_VERSION,
        "fps": fps,
        "frames": [
            {"t": round(max(0.0, t), _DIGITS), "b": [_row(box) for box in boxes]}
            for t, boxes in ordered
        ],
    }
    temp = path.with_name(path.name + ".tmp")
    with gzip.open(temp, "wt", encoding="utf-8", compresslevel=6) as stream:
        json.dump(body, stream, separators=(",", ":"))
    os.replace(temp, path)
