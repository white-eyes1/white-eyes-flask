"""FNC-S-010 5번 더미 재생의 실시간 화면·미리보기 프레임 공급 (DummyPreview)"""

import gzip
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from white_eyes_flask import create_app
from white_eyes_flask.analysis.dummy_cache import DummyCache
from white_eyes_flask.analysis.result import TrackedBox
from white_eyes_flask.camera.dummy_player import DummyPlayer, _Repetition
from white_eyes_flask.camera.dummy_preview import (
    FAILURE_COOLDOWN_SEC,
    IDLE_CLOSE_SEC,
    MAX_FORWARD_SKIP,
    BoxTimeline,
    DummyPreview,
    PreviewUnavailable,
    VideoFrames,
    to_pixels,
)
from white_eyes_flask.camera.result_queue import ResultQueue
from white_eyes_flask.config import Config
from white_eyes_flask.stream.stream_service import StreamService

SOURCE = "dummy/videos/clip.mp4"
FPS = 10
FRAMES = 30  # 3초
WIDTH, HEIGHT = 64, 48
STEP = 8  # i번째 프레임의 밝기 = i * 8 (0~232)


def make_video(path: Path, fourcc: str = "MJPG") -> Path:
    """i번째 프레임이 밝기 i*8의 단색인 3초짜리 작은 영상. 밝기로 프레임 번호를 되찾는다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*fourcc), float(FPS), (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        pytest.skip(f"이 환경에서는 {fourcc} 영상을 만들 수 없음")
    for index in range(FRAMES):
        writer.write(np.full((HEIGHT, WIDTH, 3), index * STEP, dtype=np.uint8))
    writer.release()
    return path


def frame_number(image) -> int:
    """영상 프레임의 밝기에서 프레임 번호를 되찾는다(손실 압축이라 가까운 값으로 반올림)."""
    return round(float(image.mean()) / STEP)


@pytest.fixture
def video(tmp_path):
    return make_video(tmp_path / "clip.avi")


# ---- VideoFrames ----


def test_frame_at_maps_the_position_to_the_frame_number(video):
    frames = VideoFrames(video)
    try:
        assert frame_number(frames.frame_at(0.0)) == 0
        assert frame_number(frames.frame_at(1.0)) == 10
        assert frame_number(frames.frame_at(2.5)) == 25
    finally:
        frames.close()


def test_a_position_past_the_end_gives_the_last_frame(video):
    frames = VideoFrames(video)
    try:
        assert frame_number(frames.frame_at(99.0)) == FRAMES - 1
    finally:
        frames.close()


def test_going_back_in_the_video_still_gives_the_right_frame(video):
    frames = VideoFrames(video)
    try:
        assert frame_number(frames.frame_at(2.0)) == 20
        assert frame_number(frames.frame_at(0.5)) == 5  # 반복이 다시 시작된 경우
    finally:
        frames.close()


class FakeCapture:
    """읽은 방식(시크·건너뛰기·읽기)을 세는 가짜 VideoCapture."""

    def __init__(self, fps=10.0, total=100, opened=True):
        self.fps, self.total, self.opened = fps, total, opened
        self.position = 0
        self.seeks, self.grabs, self.reads = [], 0, 0
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return {
            cv2.CAP_PROP_FPS: self.fps,
            cv2.CAP_PROP_FRAME_COUNT: self.total,
            cv2.CAP_PROP_POS_FRAMES: self.position,  # 다음에 읽을 프레임 번호
        }[prop]

    def set(self, prop, value):
        assert prop == cv2.CAP_PROP_POS_FRAMES
        self.seeks.append(int(value))
        self.position = int(value)
        return True

    def grab(self):
        self.grabs += 1
        self.position += 1
        return True

    def read(self):
        self.reads += 1
        image = np.full((4, 4, 3), self.position % 256, dtype=np.uint8)
        self.position += 1
        return True, image

    def release(self):
        self.released = True


def frames_with(capture):
    return VideoFrames(Path("clip.mp4"), opener=lambda path: capture)


def test_a_close_forward_move_reads_ahead_instead_of_seeking():
    capture = FakeCapture()
    frames = frames_with(capture)
    frames.frame_at(0.0)  # 처음에는 시크한다
    assert capture.seeks == [0]
    image = frames.frame_at(0.5)  # 프레임 5: 4장을 건너뛰고 읽는다
    assert capture.seeks == [0]  # 다시 시크하지 않았다
    assert capture.grabs == 4
    assert int(image[0, 0, 0]) == 5


def test_a_far_forward_move_seeks():
    capture = FakeCapture(total=1000)
    frames = frames_with(capture)
    frames.frame_at(0.0)
    frames.frame_at((MAX_FORWARD_SKIP + 10) / 10)
    assert len(capture.seeks) == 2


def test_the_same_frame_is_not_decoded_twice():
    capture = FakeCapture()
    frames = frames_with(capture)
    frames.frame_at(1.0)
    reads = capture.reads
    frames.frame_at(1.04)  # 영상 fps(10)보다 자주 요청해도 같은 프레임
    assert capture.reads == reads


def test_a_video_that_cannot_be_opened_is_reported():
    frames = frames_with(FakeCapture(opened=False))
    with pytest.raises(PreviewUnavailable):
        frames.frame_at(0.0)


@pytest.mark.parametrize("fps", [0.0, float("nan"), -5.0])
def test_a_video_with_a_broken_fps_is_reported(fps):
    frames = frames_with(FakeCapture(fps=fps))
    with pytest.raises(PreviewUnavailable):
        frames.frame_at(0.0)


def test_close_releases_the_capture():
    capture = FakeCapture()
    frames = frames_with(capture)
    frames.frame_at(0.0)
    frames.close()
    assert capture.released


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg가 설치돼 있지 않음")
def test_real_h264_video_with_a_long_gop_gives_the_right_frames(tmp_path):
    """실제 영상은 대부분 H.264에 긴 키프레임 간격이다(MJPG와 달리 시크가 정확하지 않을 수 있다)."""
    path = tmp_path / "gop.mp4"
    raw = b"".join(
        np.full((HEIGHT, WIDTH), index * STEP, dtype=np.uint8).tobytes()
        for index in range(FRAMES)
    )
    subprocess.run(
        [
            shutil.which("ffmpeg"),
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-s",
            f"{WIDTH}x{HEIGHT}",
            "-r",
            str(FPS),
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-g",
            "60",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(path),
        ],
        input=raw,
        check=True,
    )
    frames = VideoFrames(path)
    try:
        for second in (0.0, 1.0, 2.0, 0.5, 2.9):  # 앞으로, 뒤로 섞어서
            expected = int(second * FPS)
            assert abs(frame_number(frames.frame_at(second)) - expected) <= 1
    finally:
        frames.close()


# ---- BoxTimeline / to_pixels ----


def write_boxes(path: Path, frames) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump({"version": 1, "fps": 5, "frames": frames}, stream)
    return path


def test_boxes_use_the_last_frame_at_or_before_the_position(tmp_path):
    path = write_boxes(
        tmp_path / "b.json.gz",
        [
            {"t": 0.2, "b": [[1, 0.1, 0.1, 0.2, 0.2]]},
            {"t": 1.0, "b": [[2, 0.5, 0.5, 0.1, 0.1]]},
        ],
    )
    timeline = BoxTimeline(path)
    assert timeline.rows_at(0.0) == []  # 첫 프레임 이전
    assert timeline.rows_at(0.2)[0][0] == 1
    assert timeline.rows_at(0.99)[0][0] == 1  # 다음 프레임까지 직전 박스를 유지
    assert timeline.rows_at(1.0)[0][0] == 2
    assert timeline.rows_at(50.0)[0][0] == 2


def test_boxes_frames_out_of_order_are_sorted(tmp_path):
    path = write_boxes(
        tmp_path / "b.json.gz",
        [
            {"t": 1.0, "b": [[2, 0.5, 0.5, 0.1, 0.1]]},
            {"t": 0.2, "b": [[1, 0.1, 0.1, 0.2, 0.2]]},
        ],
    )
    assert BoxTimeline(path).rows_at(0.5)[0][0] == 1


def test_an_unreadable_box_file_means_no_boxes_not_an_error(tmp_path, caplog):
    path = tmp_path / "broken.json.gz"
    path.write_bytes(b"not gzip")
    with caplog.at_level("WARNING"):
        assert BoxTimeline(path).rows_at(1.0) == []
    assert any("박스 없이" in record.message for record in caplog.records)
    assert BoxTimeline(tmp_path / "missing.json.gz").rows_at(1.0) == []


def test_to_pixels_converts_ratios_and_clips_to_the_image():
    boxes = to_pixels([[1, 0.25, 0.5, 0.25, 0.25], [2, 0.9, 0.9, 0.5, 0.5]], 100, 200)
    assert boxes.shape == (2, 4)
    assert boxes[0].tolist() == [25.0, 100.0, 50.0, 150.0]
    assert boxes[1].tolist() == [90.0, 180.0, 100.0, 200.0]  # 영상 밖은 잘린다


def test_a_box_without_a_track_id_is_still_drawn():
    boxes = to_pixels([[None, 0.1, 0.1, 0.2, 0.2]], 100, 100)
    assert boxes.shape == (1, 4)


def test_broken_rows_are_skipped():
    rows = [
        [1, "x", 0, 0, 0],
        [1, 0.1],
        [1, float("nan"), 0, 1, 1],
        [1, 0.1, 0.1, 0.2, 0.2],
    ]
    assert to_pixels(rows, 10, 10).shape == (1, 4)
    assert to_pixels([], 10, 10).shape == (0, 4)


# ---- 반복 위치(_Repetition) ----


def test_position_is_none_before_playback_starts_or_before_s():
    state = _Repetition(3.0)
    assert state.position(100.0) is None
    state.start = 100
    assert state.position(99.0) is None  # 시계가 S보다 앞서면 만들지 않는다


def test_position_is_the_time_since_s_and_stays_at_the_end_until_the_next_repetition():
    state = _Repetition(3.0)
    state.start = 100
    assert state.position(100.0) == 0.0
    assert state.position(101.5) == 1.5
    assert state.position(103.4) == pytest.approx(
        2.999
    )  # 길이를 넘어도 마지막에 머문다
    state.start = 104  # 재생 루프가 다음 반복을 시작했다
    assert state.position(104.2) == pytest.approx(0.2)


def test_position_with_a_fractional_length_never_reaches_the_length():
    state = _Repetition(2.4)
    state.start = 0
    assert state.position(10.0) < 2.4


# ---- DummyPreview.step ----


class FixedPosition:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


def seconds_for(count_by_second):
    return [{"sec": i, "personCount": count} for i, count in enumerate(count_by_second)]


def make_preview(tmp_path, video, boxes=None, position=1.5, **kwargs):
    stream = StreamService().open(7)
    boxes_path = write_boxes(
        tmp_path / "b.json.gz",
        (
            boxes
            if boxes is not None
            else [{"t": 0.0, "b": []}, {"t": 1.0, "b": [[1, 0.25, 0.25, 0.5, 0.5]]}]
        ),
    )
    pos = FixedPosition(position)
    preview = DummyPreview(
        stream,
        video,
        boxes_path,
        seconds_for([3, 14, 25]),
        pos,
        fps=5,
        stop=threading.Event(),
        **kwargs,
    )
    return preview, stream, pos


def published(stream):
    return stream._frame


def test_step_publishes_the_frame_boxes_and_count_for_the_position(tmp_path, video):
    preview, stream, _ = make_preview(tmp_path, video, position=1.5)
    assert preview.step() is True
    frame = published(stream)
    assert frame_number(frame.image) == 15  # 위치 1.5초의 영상 프레임
    assert frame.person_count == 14  # 캐시 k=1초의 인원
    assert frame.boxes.shape == (1, 4)
    assert frame.boxes[0].tolist() == [16.0, 12.0, 48.0, 36.0]  # 64x48 영상 기준


def test_step_does_nothing_before_playback_starts(tmp_path, video):
    preview, stream, _ = make_preview(tmp_path, video, position=None)
    assert preview.step() is False
    assert published(stream) is None


def test_step_shows_the_frame_without_boxes_if_the_box_file_is_broken(tmp_path, video):
    preview, stream, _ = make_preview(tmp_path, video)
    (tmp_path / "b.json.gz").write_bytes(b"broken")
    assert preview.step() is True
    assert published(stream).boxes.shape == (0, 4)


def test_person_count_stays_in_range_for_a_position_past_the_cache(tmp_path, video):
    preview, stream, _ = make_preview(tmp_path, video, position=2.999)
    assert preview.step() is True
    assert published(stream).person_count == 25


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_repeated_failures_rest_for_a_while_and_then_try_again(tmp_path):
    clock = Clock()
    missing = tmp_path / "later.avi"
    preview, stream, _ = make_preview(tmp_path, missing, position=1.0, clock=clock)
    for _ in range(3):
        assert preview.step() is False  # 영상이 아직 없다
    make_video(missing)  # 파일이 나중에 생겼다
    assert preview.step() is False  # 쉬는 중이라 시도하지 않는다
    clock.now += FAILURE_COOLDOWN_SEC + 0.1
    assert preview.step() is True  # 영구히 끄지 않았다
    assert frame_number(published(stream).image) == 10


def test_a_failure_does_not_count_after_a_success(tmp_path, video):
    preview, _, _ = make_preview(tmp_path, video)
    assert preview.step() is True
    assert preview._failures == 0


# ---- 스레드: wants_frames일 때만 만든다 ----


def run_preview(preview, name="cam-7-preview"):
    preview.start(name)
    return preview


def wait_until(condition, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_no_frames_are_made_while_nobody_is_watching(tmp_path, video):
    preview, stream, _ = make_preview(tmp_path, video)
    preview._interval = 0.01
    run_preview(preview)
    try:
        time.sleep(0.6)
        assert published(stream) is None  # 시청자도 최근 스냅샷 요청도 없다
    finally:
        preview._stop.set()
        preview.join()


def test_a_snapshot_request_makes_frames_and_the_next_snapshot_succeeds(
    tmp_path, video
):
    preview, stream, _ = make_preview(tmp_path, video)
    preview._interval = 0.01
    run_preview(preview)
    try:
        assert stream.snapshot_jpeg() is None  # 첫 요청: 아직 프레임이 없다(503의 원인)
        assert wait_until(lambda: published(stream) is not None)
        time.sleep(1.05)  # 스냅샷 재사용 1초가 지난 뒤 다시 요청
        jpeg = stream.snapshot_jpeg()
        assert jpeg is not None and jpeg[:2] == b"\xff\xd8"
    finally:
        preview._stop.set()
        preview.join()


def test_an_mjpeg_viewer_makes_frames_and_leaving_stops_them(tmp_path, video):
    preview, stream, _ = make_preview(tmp_path, video)
    preview._interval = 0.01
    run_preview(preview)
    viewer = stream.live_jpegs()
    try:
        assert next(viewer)  # 시청자가 들어오면 프레임이 생기고 JPEG가 나온다
    finally:
        viewer.close()
        preview._stop.set()
        preview.join()


def test_the_thread_ends_and_releases_the_video_when_stopped(tmp_path):
    capture = FakeCapture()
    stream = StreamService().open(7)
    stop = threading.Event()
    preview = DummyPreview(
        stream,
        Path("clip.mp4"),
        tmp_path / "none.json.gz",
        seconds_for([1]),
        FixedPosition(0.5),
        fps=100,
        stop=stop,
        opener=lambda path: capture,
    )
    stream.snapshot_jpeg()  # 요청이 있어 프레임을 만든다
    run_preview(preview)
    assert wait_until(lambda: capture.reads > 0)
    stop.set()
    preview.join()
    assert capture.released
    assert not preview._thread.is_alive()


# ---- DummyPlayer 통합 ----


class QuietRegistrar:
    def register(self, camera_id, started_ms, video_path, box_path):
        return None


class QuietReporter:
    def report(self, *args, **kwargs):
        return True


def prepare_cache(tmp_path, source=SOURCE, make=True):
    cache = DummyCache(cache_dir=tmp_path / "cache", media_root=tmp_path / "media")
    if make:
        make_video(tmp_path / "media" / source, fourcc="mp4v")
    with cache.writer(source, fps=5) as writer:
        writer.add_frame(0.0, [TrackedBox(1, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.9)])
        writer.commit(
            [
                {
                    "sec": i,
                    "personCount": 10 + i,
                    "in": dict.fromkeys("NESW", 0),
                    "out": dict.fromkeys("NESW", 0),
                    "dwellSec": [],
                }
                for i in range(3)
            ],
            3.0,
        )
    return cache


def test_a_playing_dummy_camera_feeds_the_stream_and_stops_cleanly(tmp_path):
    cache = prepare_cache(tmp_path)
    streams = StreamService()
    stream = streams.open(7)
    player = DummyPlayer(
        cache,
        QuietRegistrar(),
        ResultQueue(maxlen=100),
        QuietReporter(),
        streams=streams,
        preview_fps=50,
        inline_registration=True,
    )
    stop = threading.Event()
    result = []
    thread = threading.Thread(
        target=lambda: result.append(player.play(7, SOURCE, stop)), daemon=True
    )
    thread.start()
    try:
        stream.snapshot_jpeg()  # 미리보기 요청
        assert wait_until(lambda: stream._frame is not None, timeout=5.0)
        frame = stream._frame
        assert frame.person_count in (10, 11, 12)  # 캐시의 인원
        assert frame.boxes.shape == (1, 4)
    finally:
        stop.set()
        thread.join(5)
    assert not thread.is_alive()
    assert result  # play가 돌아왔다
    assert not [t for t in threading.enumerate() if t.name == "cam-7-preview"]


def test_without_a_stream_service_no_preview_thread_is_started(tmp_path):
    cache = prepare_cache(tmp_path)
    player = DummyPlayer(
        cache,
        QuietRegistrar(),
        ResultQueue(maxlen=100),
        QuietReporter(),
        inline_registration=True,
    )
    stop = threading.Event()
    thread = threading.Thread(target=lambda: player.play(7, SOURCE, stop), daemon=True)
    thread.start()
    time.sleep(0.3)
    try:
        assert not [t for t in threading.enumerate() if t.name == "cam-7-preview"]
    finally:
        stop.set()
        thread.join(5)


# ---- 앱 통합: 더미 캠을 가동하고 /snapshot을 부르면 JPEG ----


def test_snapshot_of_a_started_dummy_camera_becomes_a_jpeg(tmp_path):
    prepare_cache(tmp_path)
    app = create_app(
        Config(
            internal_token="t",
            yolo_weights=None,
            host="127.0.0.1",
            port=5000,
            media_root=str(tmp_path / "media"),
            cache_dir=str(tmp_path / "cache"),
            analyze_fps=20,
        )
    )
    client = app.test_client()
    headers = {"X-Internal-Token": "t"}
    try:
        started = client.post(
            "/analyzer/v1/cameras/7/start",
            json={"sourceType": "DUMMY", "sourceUri": SOURCE},
            headers=headers,
        )
        assert started.status_code == 200
        response = client.get("/analyzer/v1/cameras/7/snapshot", headers=headers)
        assert response.status_code in (
            200,
            503,
        )  # 첫 요청은 아직 프레임이 없을 수 있다
        deadline = time.time() + 8
        while response.status_code != 200 and time.time() < deadline:
            time.sleep(0.3)
            response = client.get("/analyzer/v1/cameras/7/snapshot", headers=headers)
        assert response.status_code == 200
        assert response.mimetype == "image/jpeg"
        assert response.data[:2] == b"\xff\xd8"
    finally:
        client.post("/analyzer/v1/cameras/7/stop", headers=headers)
        app.extensions["camera_service"].shutdown()


# ---- 리뷰 반영: 정리·복구·주기 ----


def test_a_box_that_has_no_area_or_is_inside_out_is_skipped():
    rows = [[1, 0.5, 0.5, -0.2, 0.2], [2, 0.5, 0.5, 0.0, 0.2], [3, 0.1, 0.1, 0.2, 0.2]]
    assert to_pixels(rows, 100, 100).shape == (1, 4)


def test_a_row_that_is_not_a_list_is_skipped():
    assert to_pixels([{"x": 1}, 7, None, [1, 0.1, 0.1, 0.2, 0.2]], 100, 100).shape == (
        1,
        4,
    )


def test_a_failed_open_releases_the_capture_even_if_info_cannot_be_read():
    capture = FakeCapture()
    capture.get = lambda prop: (_ for _ in ()).throw(RuntimeError("backend error"))
    frames = frames_with(capture)
    with pytest.raises(PreviewUnavailable):
        frames.frame_at(0.0)
    assert capture.released


def test_repeated_failures_close_the_capture_and_open_a_new_one_later(tmp_path):
    opened = []

    def opener(path):
        capture = FakeCapture()
        capture.read = lambda: (False, None)  # 항상 읽기 실패
        opened.append(capture)
        return capture

    clock = Clock()
    stream = StreamService().open(7)
    preview = DummyPreview(
        stream,
        Path("clip.mp4"),
        tmp_path / "none.json.gz",
        seconds_for([1]),
        FixedPosition(0.5),
        fps=5,
        stop=threading.Event(),
        clock=clock,
        opener=opener,
    )
    for _ in range(3):
        assert preview.step() is False
    assert opened[0].released  # 고장 난 캡처는 닫았다
    clock.now += FAILURE_COOLDOWN_SEC + 0.1
    preview.step()
    assert len(opened) == 2  # 쉰 뒤에는 새로 열었다


def test_shutdown_ends_the_thread_even_if_the_job_stop_never_comes(tmp_path, video):
    preview, stream, _ = make_preview(tmp_path, video)
    preview._interval = 0.01
    stream.snapshot_jpeg()
    run_preview(preview)
    assert wait_until(lambda: published(stream) is not None)
    preview.shutdown()  # 재생 루프가 예외로 끝나 stop이 서지 않은 경우를 흉내
    assert not preview._thread.is_alive()
    assert not preview._stop.is_set()


def test_the_preview_thread_does_not_outlive_a_crashed_playback_loop(
    tmp_path, monkeypatch
):
    cache = prepare_cache(tmp_path)
    streams = StreamService()
    stream = streams.open(7)
    player = DummyPlayer(
        cache,
        QuietRegistrar(),
        ResultQueue(maxlen=100),
        QuietReporter(),
        streams=streams,
        preview_fps=50,
        inline_registration=True,
    )

    def crash(*args, **kwargs):
        stream.snapshot_jpeg()  # 미리보기가 돌기 시작한 뒤에 터진다
        time.sleep(0.3)
        raise RuntimeError("playback crashed")

    monkeypatch.setattr(player, "_loop", crash)
    with pytest.raises(RuntimeError):
        player.play(7, SOURCE, threading.Event())  # job의 stop은 서지 않는다
    assert wait_until(
        lambda: not [t for t in threading.enumerate() if t.name == "cam-7-preview"]
    )


def test_the_frame_rate_is_kept_even_if_making_a_frame_takes_time(tmp_path, video):
    preview, stream, _ = make_preview(tmp_path, video)
    preview._interval = 0.1  # 초당 10번
    calls = []

    def slow_step():
        calls.append(time.monotonic())
        time.sleep(0.04)  # 만드는 데 걸리는 시간
        return True

    preview.step = slow_step
    stream.snapshot_jpeg()
    run_preview(preview)
    time.sleep(1.2)
    preview.shutdown()
    # 처리 시간을 더해 기다렸다면 초당 약 7번(1.2초에 8~9번)이다. 보정하면 약 12번
    assert len(calls) >= 10


def test_a_failed_read_near_the_end_reuses_the_last_frame():
    """메타 프레임 수가 실제보다 크면 마지막 프레임이 안 읽힌다. 끝에서 멈추지 않고 직전 프레임을 쓴다."""
    capture = FakeCapture(total=100)
    frames = frames_with(capture)
    last = frames.frame_at(9.8)  # 98번 프레임은 읽힌다
    real_read = capture.read
    capture.read = lambda: (False, None)  # 99번은 실제로는 없다
    assert frames.frame_at(9.9) is last
    capture.read = real_read


def test_a_failed_read_far_from_the_end_is_still_an_error():
    capture = FakeCapture(total=100)
    frames = frames_with(capture)
    frames.frame_at(1.0)
    capture.read = lambda: (False, None)
    with pytest.raises(PreviewUnavailable):
        frames.frame_at(5.0)  # 끝이 아닌 곳의 실패는 진짜 문제다


def test_the_video_is_closed_when_nobody_watches_and_opened_again_when_watched(
    tmp_path,
):
    opened = []

    def opener(path):
        capture = FakeCapture()
        opened.append(capture)
        return capture

    stream = StreamService().open(7)
    preview = DummyPreview(
        stream,
        Path("clip.mp4"),
        tmp_path / "none.json.gz",
        seconds_for([1]),
        FixedPosition(0.5),
        fps=100,
        stop=threading.Event(),
        opener=opener,
        idle_close_sec=0.2,
    )
    stream.snapshot_jpeg()  # 요청이 있어 프레임을 만든다
    run_preview(preview)
    try:
        assert wait_until(lambda: len(opened) == 1)
        # 요청이 10초를 지나야 wants_frames가 꺼지므로 시청자 없이 시간이 흐른 상황을 직접 만든다
        stream._last_snapshot_request = time.monotonic() - 3600
        assert wait_until(
            lambda: opened[0].released, timeout=3.0
        )  # 안 보는 동안 닫았다
        stream.snapshot_jpeg()  # 다시 본다
        assert wait_until(lambda: len(opened) == 2, timeout=3.0)  # 새로 열었다
    finally:
        preview.shutdown()


def test_resting_after_repeated_failures_is_logged(tmp_path, caplog):
    clock = Clock()
    preview, _, _ = make_preview(
        tmp_path, tmp_path / "missing.avi", position=1.0, clock=clock
    )
    with caplog.at_level("WARNING"):
        for _ in range(3):
            preview.step()
    assert any("초 쉼" in record.message for record in caplog.records)


def test_the_default_idle_close_time_is_ten_seconds():
    assert IDLE_CLOSE_SEC == 10.0


def test_a_video_that_was_never_opened_is_not_opened_while_nobody_watches(tmp_path):
    opened = []

    def opener(path):
        opened.append(path)
        return FakeCapture()

    stream = StreamService().open(7)
    preview = DummyPreview(
        stream,
        Path("clip.mp4"),
        tmp_path / "none.json.gz",
        seconds_for([1]),
        FixedPosition(0.5),
        fps=100,
        stop=threading.Event(),
        opener=opener,
        idle_close_sec=0.1,
    )
    run_preview(preview)  # 요청도 시청자도 없다
    try:
        time.sleep(0.6)  # 안 열린 채로 닫을 시간이 지난다
        assert opened == []  # 한 번도 열지 않았다
    finally:
        preview.shutdown()


def test_closing_a_video_that_is_not_open_is_harmless():
    frames = frames_with(FakeCapture())
    frames.close()  # 열린 적 없어도 오류가 나지 않는다
    frames.close()  # 두 번 닫아도 마찬가지
