import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2

from white_eyes_flask.analysis.detector import Detector
from white_eyes_flask.analysis.dummy_cache import DummyResultWriter
from white_eyes_flask.analysis.edge_counter import DIRECTIONS, EdgeCounter
from white_eyes_flask.analysis.tracker import CameraTracker

PROGRESS_STEP = 5
PROGRESS_CAP = 95  # 100은 파일을 올린 뒤에 호출하는 쪽이 보고한다
_SLOT_EPSILON = 1e-6
# 컨테이너가 알려 준 전체 프레임 수 대비 이만큼은 읽어야 끝까지 읽은 것으로 본다
# (프레임 수 정보가 조금 어긋나는 영상이 있어 여유를 둔다)
_MIN_READ_RATIO = 0.98


class VideoReadError(Exception):
    """영상을 열 수 없거나 읽은 프레임이 하나도 없음."""


class AnalysisStopped(Exception):
    """정지 요청으로 분석을 중단함."""


@dataclass(frozen=True)
class AnalysisResult:
    seconds: list[dict[str, Any]]  # 영상 시각 기준 초별 결과
    duration_sec: float  # 영상 길이(읽은 프레임 수 / 영상 fps)


def analyze_video(
    path: str | Path,
    detector: Detector,
    *,
    fps: int,
    conf_thres: float,
    edge_margin: float,
    writer: DummyResultWriter,
    stop: threading.Event,
    on_progress: Callable[[int], None],
) -> AnalysisResult:
    """영상을 처음부터 끝까지 분석해 초별 결과를 돌려주고 프레임별 박스는 writer에 쓴다.

    분석 프레임은 영상 시각 기준으로 초당 fps개를 고른다(슬롯 = 영상 시각 x fps).
    영상 fps가 fps의 배수가 아니어도 되고, 오류로 건너뛴 슬롯은 시간이 지난 것으로
    센다. 추적기는 새로 시작한다. 영상 끝에서 아직 확정되지 않은 유출은 끝난 뒤
    1초 동안 안 보인 것으로 보고 마지막 초에 넣는다.
    """
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VideoReadError(f"영상을 열 수 없음: {path}")
        video_fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(video_fps) or video_fps <= 0:
            raise VideoReadError(f"영상 fps를 알 수 없음: {path}")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return _analyze(
            capture,
            video_fps,
            total_frames,
            detector,
            fps,
            conf_thres,
            edge_margin,
            writer,
            stop,
            on_progress,
        )
    finally:
        capture.release()


def _analyze(
    capture: Any,
    video_fps: float,
    total_frames: int,
    detector: Detector,
    fps: int,
    conf_thres: float,
    edge_margin: float,
    writer: DummyResultWriter,
    stop: threading.Event,
    on_progress: Callable[[int], None],
) -> AnalysisResult:
    tracker = CameraTracker(detector, conf_thres, track_buffer=fps)
    edge = EdgeCounter(edge_margin, lost_frames=fps, fps=fps)
    person_totals: dict[int, list[int]] = {}  # 초 -> [인원 합, 프레임 수]
    edge_by_second: dict[int, dict[str, Any]] = {}
    current_second: int | None = None
    last_slot = -1
    last_edge_slot: int | None = None
    next_progress = PROGRESS_STEP
    index = -1

    while True:
        if stop.is_set():
            raise AnalysisStopped()
        ok, frame = capture.read()
        if not ok:
            break  # 정상 끝(중간에 깨진 프레임도 여기서 끝으로 본다)
        index += 1
        next_progress = _report_progress(
            index, total_frames, next_progress, on_progress
        )
        slot = int(index / video_fps * fps + _SLOT_EPSILON)
        if slot <= last_slot:
            continue
        last_slot = slot
        second = slot // fps
        if second != current_second:
            if current_second is not None:
                _merge(edge_by_second, current_second, edge.pop_counts())
            current_second = second

        result = tracker.update(frame)
        if result is None:
            continue  # 탐지 오류 프레임은 건너뜀(추적기가 로그를 남김)
        points = {
            box.track_id: (box.cx, box.cy)
            for box in result.boxes
            if box.track_id is not None
        }
        edge.update(points, frame_no=slot)
        last_edge_slot = slot
        writer.add_frame(slot / fps, result.boxes)
        totals = person_totals.setdefault(second, [0, 0])
        totals[0] += result.count
        totals[1] += 1

    if current_second is None:
        raise VideoReadError("읽은 프레임이 없음")
    if total_frames > 0 and index + 1 < total_frames * _MIN_READ_RATIO:
        # 끝까지 읽지 못하고 중간에 끊김: 일부만 캐시하면 안 된다
        raise VideoReadError(f"영상을 중간까지만 읽음: {index + 1}/{total_frames}")
    if last_edge_slot is not None:
        edge.update({}, frame_no=last_edge_slot + fps)  # 끝난 뒤 1초 동안 안 보임
    _merge(edge_by_second, current_second, edge.pop_counts())
    seconds = _build_seconds(current_second, person_totals, edge_by_second)
    # 영상 길이 L = 읽은 프레임 수 / 영상 fps. 재생(FNC-S-010)이 반복 경계로 쓴다.
    duration = min((index + 1) / video_fps, float(len(seconds)))
    return AnalysisResult(seconds=seconds, duration_sec=round(duration, 3))


def _report_progress(
    index: int,
    total_frames: int,
    next_progress: int,
    on_progress: Callable[[int], None],
) -> int:
    if total_frames <= 0:
        return next_progress  # 전체 프레임 수를 모르면 진행률을 보고하지 않는다
    step = min((index + 1) * 100 // total_frames, PROGRESS_CAP)
    step = step // PROGRESS_STEP * PROGRESS_STEP
    if step >= next_progress:
        on_progress(step)
        return step + PROGRESS_STEP
    return next_progress


def _merge(
    by_second: dict[int, dict[str, Any]], second: int, counts: dict[str, Any]
) -> None:
    bucket = by_second.setdefault(
        second,
        {
            "in": {d: 0 for d in DIRECTIONS},
            "out": {d: 0 for d in DIRECTIONS},
            "dwellSec": [],
        },
    )
    for kind in ("in", "out"):
        for direction in DIRECTIONS:
            bucket[kind][direction] += counts[kind][direction]
    bucket["dwellSec"].extend(counts["dwell_sec"])


def _build_seconds(
    last_second: int,
    person_totals: dict[int, list[int]],
    edge_by_second: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    seconds = []
    for second in range(last_second + 1):
        total, frames = person_totals.get(second, (0, 0))
        edge = edge_by_second.get(second)
        seconds.append(
            {
                "sec": second,
                # 그 초 프레임들의 인원 평균을 반올림한다(0.5는 올림)
                "personCount": int(total / frames + 0.5) if frames else 0,
                "in": edge["in"] if edge else {d: 0 for d in DIRECTIONS},
                "out": edge["out"] if edge else {d: 0 for d in DIRECTIONS},
                "dwellSec": edge["dwellSec"] if edge else [],
            }
        )
    return seconds
