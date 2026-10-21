"""FNC-S-009 더미 영상 분석 (analyze_video)"""

import gzip
import json
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from white_eyes_flask.analysis.dummy_cache import DummyCache
from white_eyes_flask.analysis.video_analysis import (
    AnalysisStopped,
    VideoReadError,
    analyze_video,
)

SOURCE = "dummy/videos/clip.mp4"
WIDTH, HEIGHT = 160, 120
FPS = 5
CONF = 0.3
MARGIN = 0.05


def make_video(path: Path, frames: int, fps: float = 30.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT)
    )
    assert writer.isOpened(), "이 환경에서 mp4v 인코더를 쓸 수 없음"
    for _ in range(frames):
        writer.write(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
    writer.release()
    return path


def person(cx, cy, size=20, conf=0.9):
    return [cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2, conf]


class ScriptedDetector:
    """호출 순서(슬롯 순서)마다 정해 둔 탐지 목록을 돌려준다. 예외 객체면 던진다."""

    def __init__(self, script):
        self.script = script
        self.calls = 0
        self.on_call = None

    def detect(self, frame):
        index = self.calls
        self.calls += 1
        if self.on_call:
            self.on_call(index)
        item = self.script(index) if callable(self.script) else self.script[index]
        if isinstance(item, Exception):
            raise item
        return np.array(item, dtype=np.float32).reshape(-1, 5)


@pytest.fixture
def cache(tmp_path: Path) -> DummyCache:
    return DummyCache(cache_dir=tmp_path / "cache", media_root=tmp_path / "media")


def run(video, detector, cache, stop=None, progress=None):
    stop = stop or threading.Event()
    with cache.writer(SOURCE, fps=FPS) as writer:
        result = analyze_video(
            video,
            detector,
            fps=FPS,
            conf_thres=CONF,
            edge_margin=MARGIN,
            writer=writer,
            stop=stop,
            on_progress=progress or (lambda percent: None),
        )
        writer.commit(result.seconds, result.duration_sec)
    run.last_duration = result.duration_sec
    return result.seconds


def test_duration_is_the_number_of_frames_read_divided_by_video_fps(tmp_path, cache):
    video = make_video(tmp_path / "clip.mp4", frames=90, fps=30)
    run(video, ScriptedDetector(empty_frames), cache)
    assert run.last_duration == pytest.approx(3.0)


def test_duration_of_a_video_whose_length_is_not_whole_seconds(tmp_path, cache):
    video = make_video(tmp_path / "clip.mp4", frames=75, fps=30)
    run(video, ScriptedDetector(empty_frames), cache)
    assert run.last_duration == pytest.approx(2.5)


def read_boxes(cache):
    with gzip.open(cache.boxes_path(SOURCE), "rt", encoding="utf-8") as stream:
        return json.load(stream)


def empty_frames(_):
    return []


def test_video_is_sampled_at_analyze_fps_by_video_time(tmp_path, cache):
    video = make_video(tmp_path / "clip.mp4", frames=90, fps=30)  # 3초
    detector = ScriptedDetector(empty_frames)
    run(video, detector, cache)
    assert detector.calls == 15
    times = [frame["t"] for frame in read_boxes(cache)["frames"]]
    assert times == pytest.approx([i / 5 for i in range(15)])


def test_sampling_works_when_video_fps_is_not_a_multiple(tmp_path, cache):
    video = make_video(tmp_path / "clip.mp4", frames=24, fps=24)  # 1초
    detector = ScriptedDetector(empty_frames)
    run(video, detector, cache)
    assert detector.calls == 5


def test_one_result_per_second_with_rounded_average_count(tmp_path, cache):
    video = make_video(tmp_path / "clip.mp4", frames=90, fps=30)
    counts = [1, 1, 2, 2, 2] + [0] * 5 + [3] * 5  # 초 0: 평균 1.6, 초 1: 0, 초 2: 3

    def script(i):
        return [person(30 + 20 * k, 60) for k in range(counts[i])]

    seconds = run(video, ScriptedDetector(script), cache)
    assert [s["sec"] for s in seconds] == [0, 1, 2]
    assert [s["personCount"] for s in seconds] == [2, 0, 3]  # 1.6은 반올림해 2


def test_entering_from_the_top_band_is_counted_in_north(tmp_path, cache):
    video = make_video(tmp_path / "clip.mp4", frames=90, fps=30)

    def script(i):
        # 박스가 화면 안에 있게 작게 만들고(화면 밖으로 잘리면 중심이 달라짐) 위에서 아래로.
        # 번호는 두 번째 프레임부터 받으므로 슬롯 0·1(y=2, 4)이 위쪽 띠 안이어야 한다.
        return [person(80, 2 + 2 * min(i, 14), size=8)]

    seconds = run(video, ScriptedDetector(script), cache)
    assert sum(s["in"]["N"] for s in seconds) == 1
    assert sum(sum(s["out"].values()) for s in seconds) == 0


def test_leaving_through_the_bottom_band_counts_out_and_dwell(tmp_path, cache):
    video = make_video(tmp_path / "clip.mp4", frames=90, fps=30)

    def script(i):
        if i <= 12:
            # 안쪽에서 아래로 이동해 슬롯 12에서 y=116(아래 띠는 y > 114)
            return [person(80, 80 + 3 * i, size=8)]
        return []

    seconds = run(video, ScriptedDetector(script), cache)
    assert sum(s["out"]["S"] for s in seconds) == 1
    dwell = [d for s in seconds for d in s["dwellSec"]]
    # 번호는 두 번째로 보인 슬롯 1부터 받는다: 슬롯 1에서 처음, 슬롯 12에서 마지막 = 11 / 5
    assert dwell == [2.2]


def test_leaving_right_before_the_end_is_confirmed_after_the_last_frame(
    tmp_path, cache
):
    video = make_video(tmp_path / "clip.mp4", frames=90, fps=30)  # 슬롯 0~14

    def script(i):
        # 마지막 슬롯(y=116)에 아래 띠로 나감 -> 영상이 끝나 확정은 그 뒤
        return [person(80, 60 + 4 * i, size=8)]

    seconds = run(video, ScriptedDetector(script), cache)
    assert seconds[-1]["out"]["S"] == 1
    assert seconds[-1]["dwellSec"] == [2.6]  # 슬롯 1부터 슬롯 14까지 = 13 / 5


def test_box_file_has_one_frame_per_slot_with_normalized_boxes(tmp_path, cache):
    video = make_video(tmp_path / "clip.mp4", frames=30, fps=30)
    run(video, ScriptedDetector(lambda i: [person(80, 60)]), cache)
    data = read_boxes(cache)
    assert data["version"] == 1 and data["fps"] == 5
    assert len(data["frames"]) == 5
    track_id, x, y, w, h = data["frames"][0]["b"][0]
    assert track_id is None  # 처음 보인 프레임에는 아직 번호가 없다
    assert (x, y, w, h) == pytest.approx(
        (70 / 160, 50 / 120, 20 / 160, 20 / 120), abs=1e-4
    )
    assert all(frame["b"][0][0] == 1 for frame in data["frames"][1:])  # 번호 유지


def test_detection_error_skips_that_slot_but_continues(tmp_path, cache):
    video = make_video(tmp_path / "clip.mp4", frames=30, fps=30)

    def script(i):
        return RuntimeError("추론 실패") if i == 2 else [person(80, 60)]

    seconds = run(video, ScriptedDetector(script), cache)
    times = [frame["t"] for frame in read_boxes(cache)["frames"]]
    assert times == pytest.approx([0.0, 0.2, 0.6, 0.8])
    assert seconds[0]["personCount"] == 1


def test_progress_is_reported_in_five_percent_steps_below_one_hundred(tmp_path, cache):
    video = make_video(tmp_path / "clip.mp4", frames=90, fps=30)
    reported = []
    run(video, ScriptedDetector(empty_frames), cache, progress=reported.append)
    assert reported == sorted(set(reported))
    assert all(p % 5 == 0 and 0 < p < 100 for p in reported)
    assert reported[-1] >= 90


def test_stop_before_start_raises(tmp_path, cache):
    video = make_video(tmp_path / "clip.mp4", frames=30, fps=30)
    stop = threading.Event()
    stop.set()
    with pytest.raises(AnalysisStopped):
        run(video, ScriptedDetector(empty_frames), cache, stop=stop)
    assert cache.has(SOURCE) is False


def test_stop_in_the_middle_raises_and_leaves_no_files(tmp_path, cache):
    video = make_video(tmp_path / "clip.mp4", frames=90, fps=30)
    stop = threading.Event()
    detector = ScriptedDetector(empty_frames)
    detector.on_call = lambda index: stop.set() if index == 4 else None
    with pytest.raises(AnalysisStopped):
        run(video, detector, cache, stop=stop)
    assert detector.calls == 5
    assert [p for p in tmp_path.rglob("*") if p.is_file() and p.suffix != ".mp4"] == []


def test_unreadable_video_raises_read_error(tmp_path, cache):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"")
    with pytest.raises(VideoReadError):
        run(broken, ScriptedDetector(empty_frames), cache)


def test_missing_video_raises_read_error(tmp_path, cache):
    with pytest.raises(VideoReadError):
        run(tmp_path / "none.mp4", ScriptedDetector(empty_frames), cache)


class _FailingCapture:
    """90프레임이라고 알리지만 10번째 프레임부터 읽지 못하는 영상."""

    def __init__(self, path):
        self.reads = 0

    def isOpened(self):
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: 30.0, cv2.CAP_PROP_FRAME_COUNT: 90.0}.get(prop, 0.0)

    def read(self):
        self.reads += 1
        if self.reads > 10:
            return False, None
        return True, np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    def release(self):
        pass


def test_video_that_stops_being_readable_halfway_is_an_error(
    tmp_path, cache, monkeypatch
):
    monkeypatch.setattr(
        "white_eyes_flask.analysis.video_analysis.cv2.VideoCapture", _FailingCapture
    )
    with pytest.raises(VideoReadError):
        run(tmp_path / "any.mp4", ScriptedDetector(empty_frames), cache)
    assert cache.has(SOURCE) is False
