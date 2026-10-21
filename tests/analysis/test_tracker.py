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


def make_tracker(frames, buffer=5):
    return CameraTracker(FakeDetector(frames), conf_thres=0.3, track_buffer=buffer)


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


def test_moving_person_keeps_same_track_id():
    frames = [[person(100 + step * 6, 200)] for step in range(6)]
    tracker = make_tracker(frames)
    ids = [tracker.update(FRAME).boxes[0].track_id for _ in frames]
    assert ids[0] is not None
    assert len(set(ids)) == 1


def test_two_people_get_different_track_ids():
    frames = [[person(100, 100), person(600, 300)] for _ in range(3)]
    tracker = make_tracker(frames)
    result = None
    for _ in frames:
        result = tracker.update(FRAME)
    ids = {box.track_id for box in result.boxes}
    assert None not in ids
    assert len(ids) == 2


def test_person_appearing_later_has_no_track_id_until_matched_again():
    frames = [
        [person(100, 100)],
        [person(100, 100), person(600, 300)],
        [person(100, 100), person(600, 300)],
    ]
    tracker = make_tracker(frames)
    tracker.update(FRAME)
    second = tracker.update(FRAME)
    newcomer = max(second.boxes, key=lambda box: box.x)
    assert second.count == 2
    assert newcomer.track_id is None
    third = tracker.update(FRAME)
    assert max(third.boxes, key=lambda box: box.x).track_id is not None


def test_detection_error_skips_frame_and_keeps_tracking(caplog):
    frames = [
        [person(100, 100)],
        RuntimeError("추론 실패"),
        [person(104, 100)],
    ]
    tracker = make_tracker(frames)
    first = tracker.update(FRAME).boxes[0].track_id
    assert tracker.update(FRAME) is None
    assert "탐지 오류" in caplog.text
    assert tracker.update(FRAME).boxes[0].track_id == first


def test_track_is_forgotten_after_buffer_frames():
    frames = [[person(100, 100)]] + [[] for _ in range(8)] + [[person(100, 100)]]
    tracker = make_tracker(frames, buffer=3)
    first = tracker.update(FRAME).boxes[0].track_id
    for _ in range(8):
        tracker.update(FRAME)
    again = tracker.update(FRAME).boxes[0].track_id
    assert again != first


def test_cameras_do_not_share_track_state():
    busy = make_tracker([[person(100, 100), person(400, 100), person(700, 100)]])
    quiet = make_tracker([[person(500, 200)]])
    busy.update(FRAME)
    # 다른 캠에서 번호를 3개 썼어도 이 캠의 첫 사람 번호는 1부터 시작한다.
    assert quiet.update(FRAME).boxes[0].track_id == 1
