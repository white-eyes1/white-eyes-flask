"""FNC-S-009 더미 영상 분석: 영상을 끝까지 분석해 초별 결과를 만든다"""

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

# 진행률을 몇 % 단위로 보고할지
PROGRESS_STEP = 5
PROGRESS_CAP = 95  # 100은 파일을 올린 뒤에 호출하는 쪽이 보고한다
# 슬롯 번호 계산에서 부동소수점 오차로 경계 프레임이 앞 슬롯으로 밀리지 않게 더하는 아주 작은 값
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
    """영상 분석의 최종 결과. 캐시 JSON(초별 결과)과 영상 길이가 된다."""

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
    센다. 추적기는 새로 시작한다. 영상은 반복 재생되므로 경계에서 유입/유출이 생기지
    않게 한다: 시작 때 첫 두 분석 프레임에 처음 보인 사람은 유입으로 세지 않고,
    끝에서 아직 확정되지 않은 유출은 세지 않는다.
    """
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VideoReadError(f"영상을 열 수 없음: {path}")
        # 영상 fps가 0·NaN이면 프레임 번호를 시각으로 바꿀 수 없어 분석을 시작하지 않는다
        video_fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(video_fps) or video_fps <= 0:
            raise VideoReadError(f"영상 fps를 알 수 없음: {path}")
        # 전체 프레임 수는 진행률 계산과 중간 끊김 검사에만 쓴다(컨테이너가 모르면 0 이하일 수 있다)
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
        # 성공·실패·중단 어느 경우든 영상 파일 핸들을 반드시 닫는다
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
    """영상을 한 프레임씩 읽으며 탐지·추적·유입유출 집계를 하고 초별 결과로 묶는다(analyze_video의 본체)."""
    # 캠 하나분의 상태를 새로 만든다(추적기·유입유출 집계기). 시작 직후 2프레임의 사람은 이미 있던 사람으로 본다
    tracker = CameraTracker(detector, conf_thres, track_buffer=fps)
    edge = EdgeCounter(edge_margin, lost_frames=fps, fps=fps, start_frames=2)
    person_totals: dict[int, list[int]] = {}  # 초 -> [인원 합, 프레임 수]
    # 초 -> 그 초에 쌓인 유입·유출·체류 시간
    edge_by_second: dict[int, dict[str, Any]] = {}
    current_second: int | None = None
    # 마지막으로 분석한 슬롯 번호. 같은 슬롯에 해당하는 나머지 프레임은 건너뛰는 기준
    last_slot = -1
    next_progress = PROGRESS_STEP
    # 지금까지 읽은 프레임의 0부터 센 번호
    index = -1

    while True:
        # 프레임마다 정지 요청을 확인해 캠을 끄면 곧바로 분석을 멈춘다
        if stop.is_set():
            raise AnalysisStopped()
        ok, frame = capture.read()
        if not ok:
            break  # 정상 끝(중간에 깨진 프레임도 여기서 끝으로 본다)
        index += 1
        # 읽은 프레임 기준으로 진행률을 보고한다(분석 대상이 아닌 프레임도 읽었으니 포함한다)
        next_progress = _report_progress(
            index, total_frames, next_progress, on_progress
        )
        # 영상 시각(index / video_fps)에 초당 분석 수(fps)를 곱한 값이 슬롯 번호다.
        # 영상 fps가 분석 fps의 배수가 아니어도 시각 기준으로 고르게 솎아 낸다
        slot = int(index / video_fps * fps + _SLOT_EPSILON)
        # 이미 분석한 슬롯이면(프레임이 더 촘촘한 경우) 이 프레임은 건너뛴다
        if slot <= last_slot:
            continue
        last_slot = slot
        # 슬롯이 fps개 모일 때마다 영상 시각 1초
        second = slot // fps
        # 초가 바뀌는 순간, 끝난 초 동안 쌓인 유입·유출을 그 초에 귀속시킨다
        if second != current_second:
            if current_second is not None:
                _merge(edge_by_second, current_second, edge.pop_counts())
            current_second = second

        result = tracker.update(frame)
        if result is None:
            continue  # 탐지 오류 프레임은 건너뜀(추적기가 로그를 남김)
        # 추적 번호가 있는 박스만 유입유출 판정에 넣는다(번호 없는 박스는 인원에만 센다)
        points = {
            box.track_id: (box.cx, box.cy)
            for box in result.boxes
            if box.track_id is not None
        }
        # 슬롯 번호를 프레임 번호로 넘겨, 오류로 건너뛴 슬롯만큼 시간이 흐른 것으로 센다
        edge.update(points, frame_no=slot)
        # 프레임별 박스 좌표는 영상 시각(초)과 함께 좌표 파일에 바로 흘려 쓴다
        writer.add_frame(slot / fps, result.boxes)
        # 그 초의 인원 평균을 내려고 합계와 프레임 수를 누적한다
        totals = person_totals.setdefault(second, [0, 0])
        totals[0] += result.count
        totals[1] += 1

    # 분석 슬롯이 한 번도 없었다 = 읽은 프레임이 없다
    if current_second is None:
        raise VideoReadError("읽은 프레임이 없음")
    if total_frames > 0 and index + 1 < total_frames * _MIN_READ_RATIO:
        # 끝까지 읽지 못하고 중간에 끊김: 일부만 캐시하면 안 된다
        raise VideoReadError(f"영상을 중간까지만 읽음: {index + 1}/{total_frames}")
    # 영상 끝에서 확정되지 않은 '나가는 중'은 유출로 세지 않는다(FNC-S-004 업무 규칙 ②)
    # 마지막 초에 쌓인 값만 꺼내 기록한다. 아직 끊김이 확정되지 않은 사람은 _finish가 불리지 않았으므로 유출로 세어지지 않는다
    _merge(edge_by_second, current_second, edge.pop_counts())
    seconds = _build_seconds(current_second, person_totals, edge_by_second)
    # 영상 길이 L = 읽은 프레임 수 / 영상 fps. 재생(FNC-S-010)이 반복 경계로 쓴다.
    # 길이가 초별 결과 개수를 넘으면 캐시 검증(_validate)에서 거부되므로 개수로 상한을 둔다
    duration = min((index + 1) / video_fps, float(len(seconds)))
    return AnalysisResult(seconds=seconds, duration_sec=round(duration, 3))


def _report_progress(
    index: int,
    total_frames: int,
    next_progress: int,
    on_progress: Callable[[int], None],
) -> int:
    """진행률이 다음 보고 단계(5% 단위)에 닿으면 on_progress로 알리고, 다음 보고 기준값을 돌려준다."""
    if total_frames <= 0:
        return next_progress  # 전체 프레임 수를 모르면 진행률을 보고하지 않는다
    # 읽은 비율(%)을 95%까지만 센다(100%는 파일 저장 후 호출하는 쪽이 알린다)
    step = min((index + 1) * 100 // total_frames, PROGRESS_CAP)
    # 5% 단위로 내림해 보고 횟수를 줄인다(프레임마다 보고하면 호출이 너무 잦다)
    step = step // PROGRESS_STEP * PROGRESS_STEP
    if step >= next_progress:
        on_progress(step)
        return step + PROGRESS_STEP
    return next_progress


def _merge(
    by_second: dict[int, dict[str, Any]], second: int, counts: dict[str, Any]
) -> None:
    """EdgeCounter.pop_counts 결과(counts)를 해당 초의 버킷에 더한다. 같은 초가 여러 번 들어와도 합쳐진다."""
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
    """0초부터 마지막 초까지 빠짐없이 초별 결과를 만든다. 분석된 프레임이 없던 초는 0과 빈 값으로 채운다."""
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
