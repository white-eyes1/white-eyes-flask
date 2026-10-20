import numpy as np
import pytest

from white_eyes_flask.analysis.point_tracker import PointTracker

SIZE = 20.0


def step(tracker, points, dt=1.0):
    points = np.array(points, dtype=float).reshape(-1, 2)
    return tracker.update(points, np.full(len(points), SIZE), dt)


def confirmed_ids(tracker, point, times=3):
    ids = None
    for _ in range(times):
        ids = step(tracker, [point])
    return ids


def test_no_number_until_seen_twice():
    tracker = PointTracker()
    assert step(tracker, [[100, 100]]) == [None]
    assert step(tracker, [[100, 100]]) == [1]


def test_confirm_hits_can_be_raised():
    tracker = PointTracker(confirm_hits=3)
    assert step(tracker, [[100, 100]]) == [None]
    assert step(tracker, [[100, 100]]) == [None]
    assert step(tracker, [[100, 100]]) == [1]


def test_moving_person_keeps_one_number():
    tracker = PointTracker()
    ids = [step(tracker, [[100 + 8 * i, 200]])[0] for i in range(8)]
    assert ids[0] is None
    assert set(ids[1:]) == {1}


def test_two_people_get_different_numbers():
    tracker = PointTracker()
    ids = None
    for _ in range(3):
        ids = step(tracker, [[100, 100], [600, 300]])
    assert sorted(ids) == [1, 2]


def test_one_detection_is_never_given_to_two_tracks():
    tracker = PointTracker()
    for _ in range(3):
        step(tracker, [[100, 100], [112, 100]])
    ids = step(tracker, [[106, 100]])  # 두 트랙의 반경 안에 하나만 있다
    assert len(ids) == 1
    # 남은 트랙은 놓친 것으로 세어져 곧 사라지지 않는다
    assert len(tracker._tracks) == 2


def test_new_detections_are_not_matched_with_each_other_in_the_same_frame():
    tracker = PointTracker()
    step(tracker, [[100, 100], [105, 100]])
    assert len(tracker._tracks) == 2


def test_numbers_increase_and_are_never_reused():
    tracker = PointTracker(max_missed=1)
    first = confirmed_ids(tracker, [100, 100])[0]
    for _ in range(3):
        step(tracker, [])  # 사라짐
    second = confirmed_ids(tracker, [100, 100])[0]
    assert second > first


def test_person_is_remembered_through_a_short_gap_and_keeps_number():
    tracker = PointTracker(max_missed=3)
    first = confirmed_ids(tracker, [100, 100])[0]
    step(tracker, [])
    step(tracker, [])
    assert step(tracker, [[100, 100]]) == [first]


def test_person_is_forgotten_after_too_many_missed_frames():
    tracker = PointTracker(max_missed=3)
    first = confirmed_ids(tracker, [100, 100])[0]
    for _ in range(4):
        step(tracker, [])
    assert tracker._tracks == {}
    assert confirmed_ids(tracker, [100, 100])[0] != first


def test_empty_frames_advance_expiry():
    tracker = PointTracker(max_missed=2)
    step(tracker, [[100, 100]])
    for _ in range(3):
        step(tracker, [])
    assert tracker._tracks == {}


def test_confirmed_person_keeps_number_after_a_miss():
    tracker = PointTracker()
    first = confirmed_ids(tracker, [100, 100])[0]
    step(tracker, [])
    assert step(tracker, [[100, 100]]) == [first]  # 다시 3번을 볼 필요 없다


def test_moving_person_is_followed_through_a_gap_using_velocity():
    tracker = PointTracker(max_missed=3)
    ids = None
    for i in range(4):
        ids = step(tracker, [[100 + 15 * i, 100]])
    number = ids[0]
    step(tracker, [])  # 한 프레임 놓침. 15px씩 움직이는 사람은 이제 175 근처
    assert step(tracker, [[100 + 15 * 5, 100]]) == [number]


def test_prediction_is_not_applied_twice_after_a_miss():
    # 속도가 큰 사람이 두 프레임 놓친 뒤 예측 위치 근처에서 다시 나타난다.
    tracker = PointTracker(max_missed=3, max_radius=40)
    for i in range(4):
        step(tracker, [[100 + 10 * i, 100]])  # 마지막 관측 130, 속도 10
    step(tracker, [])
    step(tracker, [])
    # 마지막 관측 130에서 3프레임 뒤이므로 예측은 130 + 10 * 3 * (1 + 0.5 * 2) / ... 가 아니라
    # 130 + 10 * (1 + 0.5 * 2) = 150. 반경 안이어야 한다.
    ids = step(tracker, [[152, 100]])
    assert ids[0] is not None


def test_far_detection_is_a_new_person():
    tracker = PointTracker()
    first = confirmed_ids(tracker, [100, 100])[0]
    ids = step(tracker, [[100, 100], [900, 400]])
    assert ids[0] == first
    assert ids[1] is None  # 새 사람은 아직 번호가 없다


def test_match_radius_scales_with_person_size():
    tracker = PointTracker()
    for _ in range(3):
        tracker.update(np.array([[100.0, 100.0]]), np.array([50.0]))
    # 크기 50이면 반경 60(최대), 30px 떨어져도 같은 사람
    assert tracker.update(np.array([[130.0, 100.0]]), np.array([50.0])) == [1]


def test_equal_distance_ties_are_resolved_deterministically():
    results = set()
    for _ in range(5):
        tracker = PointTracker()
        for _ in range(3):
            step(tracker, [[100, 100], [140, 100]])
        results.add(tuple(step(tracker, [[120, 100]])))
    assert len(results) == 1


def test_irregular_frame_gap_uses_elapsed_frames_for_prediction():
    tracker = PointTracker()
    for i in range(4):
        step(tracker, [[100 + 10 * i, 100]])  # 속도 10/프레임, 마지막 130
    # 3프레임이 건너뛰어진 뒤(dt=3): 예측은 130 + 30 = 160
    assert step(tracker, [[160, 100]], dt=3.0)[0] is not None
    assert len(tracker._tracks) == 1


def test_missed_time_adds_up_the_skipped_frames():
    tracker = PointTracker(max_missed=3)
    step(tracker, [[100, 100]])
    step(tracker, [], dt=2.0)
    assert len(tracker._tracks) == 1
    step(tracker, [], dt=2.0)  # 4프레임 동안 못 봄 > 3
    assert tracker._tracks == {}


def test_speed_uses_the_time_since_the_last_real_sighting():
    tracker = PointTracker(max_missed=10)
    for i in range(4):
        step(tracker, [[100 + 10 * i, 100]])  # 속도 10, 마지막 130
    step(tracker, [], dt=5.0)
    # 마지막 관측 뒤 6프레임이 지났다. 그 사이 60을 움직였다면 속도는 그대로 10이다.
    step(tracker, [[190, 100]])
    (track,) = tracker._tracks.values()
    assert track.vx == pytest.approx(10.0)


def test_mismatched_inputs_are_rejected():
    tracker = PointTracker()
    with pytest.raises(ValueError):
        tracker.update(np.zeros((2, 2)), np.zeros(3))


def test_non_finite_input_is_rejected():
    tracker = PointTracker()
    with pytest.raises(ValueError):
        tracker.update(np.array([[np.nan, 1.0]]), np.array([10.0]))


def test_non_positive_dt_is_rejected():
    tracker = PointTracker()
    with pytest.raises(ValueError):
        tracker.update(np.zeros((0, 2)), np.zeros(0), dt_frames=0)


def test_trackers_do_not_share_state():
    a, b = PointTracker(), PointTracker()
    confirmed_ids(a, [100, 100])
    assert confirmed_ids(b, [500, 500])[0] == 1
