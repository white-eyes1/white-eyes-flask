"""FNC-S-008 라이브 좌표 파일(.json.gz). 형식은 conventions '좌표 파일 형식'."""

import gzip
import json
import os
from pathlib import Path

from white_eyes_flask.analysis.result import TrackedBox

# 좌표 파일 형식의 버전(파일의 version 필드). 형식이 바뀌면 올린다.
BOX_FILE_VERSION = 1
# 좌표를 반올림하는 소수 자릿수. 파일 크기를 줄인다.
_DIGITS = 4


def _ratio(value: float) -> float:
    """좌표는 영상 크기 대비 0~1 비율, 소수 4자리."""
    # 0~1 밖의 값은 잘라 낸 뒤 반올림한다.
    return round(min(1.0, max(0.0, value)), _DIGITS)


def _row(box: TrackedBox) -> list:
    """박스 하나를 [추적 번호, x, y, w, h] 한 줄로 만든다(좌표는 0~1 비율)."""
    return [box.track_id, _ratio(box.x), _ratio(box.y), _ratio(box.w), _ratio(box.h)]


def write_box_file(
    path: Path, fps: int, frames: list[tuple[float, list[TrackedBox]]]
) -> None:
    """frames는 (그 분 파일 시작부터의 초, 박스 목록). 임시 이름에 쓴 뒤 올려 반쪽 파일을 남기지 않는다."""
    # 박스가 거꾸로 도착해도 다시보기가 "t ≤ 재생 위치인 마지막 프레임"을 찾을 수 있게 t 순서로 쓴다
    ordered = sorted(frames, key=lambda frame: frame[0])
    # 파일 본문: 형식 버전, 분석 fps, 프레임 목록. 프레임의 t는 그 분 파일 시작부터의 초, b는 그 시각의 박스 목록이다.
    body = {
        "version": BOX_FILE_VERSION,
        "fps": fps,
        "frames": [
            {"t": round(max(0.0, t), _DIGITS), "b": [_row(box) for box in boxes]}
            for t, boxes in ordered
        ],
    }
    # 같은 폴더의 임시 이름(.tmp)에 먼저 쓴다.
    temp = path.with_name(path.name + ".tmp")
    # gzip으로 압축하고, 공백 없는 JSON으로 써서 파일 크기를 줄인다.
    with gzip.open(temp, "wt", encoding="utf-8", compresslevel=6) as stream:
        json.dump(body, stream, separators=(",", ":"))
    # 다 쓴 뒤 이름만 바꿔 올린다. 같은 파일시스템 안에서는 교체가 한 번에 일어나 읽는 쪽이 반쯤 쓰인 파일을 볼 수 없다.
    os.replace(temp, path)
