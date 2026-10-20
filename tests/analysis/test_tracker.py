import numpy as np
import pytest

from white_eyes_flask.analysis.tracker import CameraTracker

WIDTH, HEIGHT = 1000, 500
FRAME = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


class FakeDetector:
    """미리 정해 둔 탐지 목록을 프레임마다 하나씩 돌려준다. 예외 객체면 던진다."""

    def __init__(self, frames):
        self._frames = list(frames)

    def detect(self, frame):
        item = self._frames.pop(0)
        if isinstance(item, Exception):
            raise item
        return np.array(item, dtype=np.float32).reshape(-1, 5)


def person(x, y, size=40, conf=0.9):
    return [x, y, x + size, y + size, conf]


def make_tracker(frames, buffer=5, conf=0.3):
    return CameraTracker(FakeDetector(frames), conf_thres=conf, track_buffer=buffer)


def run(tracker, count, frame=FRAME):
    result = None
    for _ in range(count):
        result = tracker.update(frame)
    return result


def test_box_is_normalized_with_top_left_and_center():
    tracker = make_tracker([[[100, 50, 200, 100, 0.8]]])
    box = tracker.update(FRAME).boxes[0]
    assert (box.x, box.y, box.w, box.h) == pytest.approx((0.1, 0.1, 0.1, 0.1))
    assert (box.cx, box.cy) == pytest.approx((0.15, 0.15))
    assert box.conf == pytest.approx(0.8)


def test_box_outside_frame_is_clipped_to_unit_range():
    tracker = make_tracker([[[-20, -10, 1100, 520, 0.8]]])
    box = tracker.update(FRAME).boxes[0]
    assert (box.x, box.y, box.w, box.h) == pytest.approx((0.0, 0.0, 1.0, 1.0))


def test_degenerate_box_is_dropped_from_count():
    tracker = make_tracker([[person(100, 100), [300, 300, 300, 340, 0.9]]])
    result = tracker.update(FRAME)
    assert result.count == 1
    assert len(result.boxes) == 1


def test_detection_below_confidence_threshold_is_not_counted():
    tracker = make_tracker([[person(100, 100, conf=0.9), person(400, 300, conf=0.2)]])
    assert tracker.update(FRAME).count == 1


def test_count_is_number_of_detections():
    tracker = make_tracker([[person(100, 100), person(400, 300), person(800, 50)]])
    result = tracker.update(FRAME)
    assert result.count == 3
    assert len(result.boxes) == 3


def test_empty_frame_gives_zero_count_and_no_boxes():
    tracker = make_tracker([[]])
    result = tracker.update(FRAME)
    assert result.count == 0
    assert result.boxes == []


def test_moving_person_keeps_same_track_id_once_numbered():
    frames = [[person(100 + step * 6, 200)] for step in range(6)]
    tracker = make_tracker(frames)
    ids = [tracker.update(FRAME).boxes[0].track_id for _ in frames]
    assert ids[0] is None  # 처음엔 번호가 없다
    assert ids[1] is not None
    assert len(set(ids[1:])) == 1


def test_two_people_get_different_track_ids():
    frames = [[person(100, 100), person(600, 300)] for _ in range(3)]
    result = run(make_tracker(frames), 3)
    ids = {box.track_id for box in result.boxes}
    assert None not in ids
    assert len(ids) == 2


def test_person_appearing_later_has_no_track_id_until_seen_again():
    frames = [
        [person(100, 100)],
        [person(100, 100)],
        [person(100, 100), person(600, 300)],
        [person(100, 100), person(600, 300)],
    ]
    tracker = make_tracker(frames)
    run(tracker, 2)
    third = tracker.update(FRAME)
    newcomer = max(third.boxes, key=lambda box: box.x)
    old = min(third.boxes, key=lambda box: box.x)
    assert third.count == 2
    assert newcomer.track_id is None
    assert old.track_id is not None
    fourth = tracker.update(FRAME)
    assert max(fourth.boxes, key=lambda box: box.x).track_id is not None


def test_detection_error_skips_frame_and_keeps_tracking(caplog):
    frames = [
        [person(100, 100)],
        [person(102, 100)],
        [person(104, 100)],
        RuntimeError("추론 실패"),
        [person(106, 100)],
    ]
    tracker = make_tracker(frames)
    first = run(tracker, 3).boxes[0].track_id
    assert tracker.update(FRAME) is None
    assert "탐지 오류" in caplog.text
    assert tracker.update(FRAME).boxes[0].track_id == first


def test_failed_frames_count_as_elapsed_time_for_following_the_person():
    # 30px씩 움직이는 사람이 분석 실패 2번을 지나 90px 떨어진 곳에서 다시 보인다.
    frames = [
        [person(100, 100)],
        [person(130, 100)],
        [person(160, 100)],
        RuntimeError("추론 실패"),
        RuntimeError("추론 실패"),
        [person(250, 100)],
    ]
    tracker = make_tracker(frames, buffer=10)
    first = run(tracker, 3).boxes[0].track_id
    assert first is not None
    assert tracker.update(FRAME) is None
    assert tracker.update(FRAME) is None
    assert tracker.update(FRAME).boxes[0].track_id == first


def test_track_is_forgotten_after_buffer_frames():
    frames = (
        [[person(100, 100)]] * 3 + [[] for _ in range(8)] + [[person(100, 100)]] * 3
    )
    tracker = make_tracker(frames, buffer=3)
    first = run(tracker, 3).boxes[0].track_id
    run(tracker, 8)
    again = run(tracker, 3).boxes[0].track_id
    assert again != first


def test_short_gap_keeps_the_same_number():
    frames = [[person(100, 100)]] * 3 + [[]] + [[person(100, 100)]]
    tracker = make_tracker(frames, buffer=5)
    first = run(tracker, 3).boxes[0].track_id
    tracker.update(FRAME)
    assert tracker.update(FRAME).boxes[0].track_id == first


def test_tracking_does_not_depend_on_video_resolution():
    # 같은 사람이 같은 비율로 움직이면 해상도가 달라도 같은 번호로 이어진다.
    for width, height in ((500, 250), (1000, 500), (3840, 2160)):
        scale = width / 1000
        frames = [
            [
                [
                    (100 + 10 * i) * scale,
                    100 * scale,
                    (140 + 10 * i) * scale,
                    140 * scale,
                    0.9,
                ]
            ]
            for i in range(6)
        ]
        tracker = make_tracker(frames)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        ids = [tracker.update(frame).boxes[0].track_id for _ in frames]
        assert len(set(ids[2:])) == 1 and ids[2] is not None, (width, height)


def test_cameras_do_not_share_track_state():
    busy = make_tracker([[person(100, 100), person(400, 100), person(700, 100)]] * 3)
    quiet = make_tracker([[person(500, 200)]] * 3)
    run(busy, 3)
    # 다른 캠에서 번호를 3개 썼어도 이 캠의 첫 사람 번호는 1부터 시작한다.
    assert run(quiet, 3).boxes[0].track_id == 1
