"""FNC-S-004 유입/유출 판정 (EdgeCounter)"""

import threading

import pytest

from white_eyes_flask.analysis.edge_counter import EdgeCounter, band_of

MARGIN = 0.05
LOST = 5


def make(lost=LOST):
    return EdgeCounter(margin=MARGIN, lost_frames=lost)


def move(counter, path, track_id=1):
    """한 사람이 path의 점들을 프레임마다 하나씩 지나간다."""
    for point in path:
        counter.update({track_id: point})


def idle(counter, frames):
    """아무도 안 보이는 프레임을 frames개 보낸다."""
    for _ in range(frames):
        counter.update({})


def totals(counter):
    """유입·유출 수만 비교한다(체류 시간은 test_dwell_time.py에서 본다)."""
    counts = counter.pop_counts()
    del counts["dwell_sec"]
    return counts


def empty():
    zero = {"N": 0, "E": 0, "S": 0, "W": 0}
    return {"in": dict(zero), "out": dict(zero)}


def expect(**changes):
    result = empty()
    for key, count in changes.items():
        kind, direction = key.split("_")
        result[kind][direction] = count
    return result


@pytest.mark.parametrize(
    "point,band",
    [
        ((0.5, 0.02), "N"),
        ((0.5, 0.98), "S"),
        ((0.02, 0.5), "W"),
        ((0.98, 0.5), "E"),
        ((0.5, 0.5), None),
        ((0.5, 0.05), None),  # 경계값은 안쪽
        ((0.5, 0.95), None),
        ((0.05, 0.5), None),
        ((0.95, 0.5), None),
        ((0.5, 0.0499), "N"),
        ((0.02, 0.04), "W"),  # 모서리: 띠 깊이가 더 깊은 쪽
        ((0.04, 0.02), "N"),
        ((0.98, 0.96), "E"),
        ((0.96, 0.98), "S"),
        ((0.02, 0.02), "N"),  # 깊이가 같으면 남북이 먼저
        ((0.98, 0.98), "S"),
        ((0.02, 0.98), "S"),
        ((0.98, 0.02), "N"),  # 부동소수점 오차가 있어도 동률은 남북 우선
        ((0.97, 0.03), "N"),
        ((0.03, 0.97), "S"),
        ((-0.2, 0.5), "W"),  # 범위 밖은 0~1로 자름
        ((0.5, 1.3), "S"),
    ],
)
def test_band_of_classifies_position(point, band):
    assert band_of(*point, MARGIN) == band


@pytest.mark.parametrize(
    "direction,path",
    [
        ("N", [(0.5, 0.02), (0.5, 0.1), (0.5, 0.3)]),
        ("S", [(0.5, 0.98), (0.5, 0.9), (0.5, 0.7)]),
        ("W", [(0.02, 0.5), (0.1, 0.5), (0.3, 0.5)]),
        ("E", [(0.98, 0.5), (0.9, 0.5), (0.7, 0.5)]),
    ],
)
def test_person_entering_from_band_counts_in_for_that_direction(direction, path):
    counter = make()
    move(counter, path)
    assert totals(counter) == expect(**{f"in_{direction}": 1})


@pytest.mark.parametrize(
    "direction,path",
    [
        ("N", [(0.5, 0.3), (0.5, 0.1), (0.5, 0.02)]),
        ("S", [(0.5, 0.7), (0.5, 0.9), (0.5, 0.98)]),
        ("W", [(0.3, 0.5), (0.1, 0.5), (0.02, 0.5)]),
        ("E", [(0.7, 0.5), (0.9, 0.5), (0.98, 0.5)]),
    ],
)
def test_person_leaving_through_band_counts_out_after_lost(direction, path):
    counter = make()
    move(counter, path)
    assert totals(counter) == empty()  # 아직 끊기지 않았으므로 확정 전
    idle(counter, LOST)
    assert totals(counter) == expect(**{f"out_{direction}": 1})


def test_person_first_seen_inside_does_not_count_in():
    counter = make()
    move(counter, [(0.5, 0.5), (0.5, 0.4), (0.5, 0.3)])
    idle(counter, LOST)
    assert totals(counter) == empty()


def test_person_first_seen_inside_and_vanishing_inside_counts_nothing():
    counter = make()
    move(counter, [(0.5, 0.5), (0.52, 0.5)])
    idle(counter, LOST + 3)
    assert totals(counter) == empty()


def test_person_seen_only_in_band_and_vanishing_counts_nothing():
    counter = make()
    move(counter, [(0.5, 0.02), (0.5, 0.03)])
    idle(counter, LOST + 3)
    assert totals(counter) == empty()


def test_person_passing_through_counts_in_then_out():
    counter = make()
    move(counter, [(0.5, 0.02), (0.5, 0.3), (0.5, 0.6), (0.5, 0.98)])
    idle(counter, LOST)
    assert totals(counter) == expect(in_N=1, out_S=1)


def test_returning_inside_cancels_leaving():
    counter = make()
    move(counter, [(0.5, 0.5), (0.5, 0.02), (0.5, 0.3)])
    idle(counter, LOST)
    assert totals(counter) == empty()


def test_in_is_not_counted_twice_when_jittering_at_the_line():
    counter = make()
    move(
        counter,
        [(0.5, 0.02), (0.5, 0.1), (0.5, 0.03), (0.5, 0.1), (0.5, 0.04), (0.5, 0.2)],
    )
    assert totals(counter) == expect(in_N=1)


def test_last_leaving_direction_wins_when_moving_between_bands():
    counter = make()
    move(counter, [(0.5, 0.5), (0.5, 0.02), (0.98, 0.5)])
    idle(counter, LOST)
    assert totals(counter) == expect(out_E=1)


def test_entering_through_another_band_than_first_seen_counts_no_in():
    counter = make()
    move(counter, [(0.5, 0.02), (0.98, 0.5), (0.7, 0.5)])
    assert totals(counter) == empty()


def test_reappearing_before_lost_frames_continues_same_person():
    counter = make()
    move(counter, [(0.5, 0.5), (0.5, 0.02)])
    idle(counter, LOST - 1)
    move(counter, [(0.5, 0.02)])  # 4프레임 쉬고 다시 나타남
    idle(counter, LOST)
    assert totals(counter) == expect(out_N=1)


def test_person_is_confirmed_on_the_fifth_missing_frame():
    counter = make()
    move(counter, [(0.5, 0.5), (0.5, 0.02)])
    idle(counter, LOST - 1)
    assert totals(counter) == empty()
    idle(counter, 1)
    assert totals(counter) == expect(out_N=1)


def test_person_is_counted_out_only_once():
    counter = make()
    move(counter, [(0.5, 0.5), (0.5, 0.02)])
    idle(counter, LOST + 10)
    assert totals(counter) == expect(out_N=1)
    assert totals(counter) == empty()


def test_missing_slots_count_as_elapsed_time():
    counter = make()
    counter.update({1: (0.5, 0.5)}, frame_no=1)
    counter.update({1: (0.5, 0.02)}, frame_no=2)
    # 3~6번 슬롯이 오류로 건너뛰어짐(4개 누락): 같은 사람이 이어짐
    counter.update({1: (0.5, 0.02)}, frame_no=7)
    assert totals(counter) == empty()
    # 8~13번 슬롯 누락 후 다시 나타남: 이미 끊긴 사람이라 새 사람이고, 이전 사람은 out 확정
    counter.update({1: (0.5, 0.5)}, frame_no=14)
    assert totals(counter) == expect(out_N=1)


def test_frame_numbers_can_start_from_zero():
    counter = make()
    counter.update({1: (0.5, 0.02)}, frame_no=0)
    counter.update({1: (0.5, 0.3)}, frame_no=1)
    assert totals(counter) == expect(in_N=1)


def test_person_first_seen_within_start_frames_does_not_count_in():
    counter = EdgeCounter(margin=MARGIN, lost_frames=LOST, start_frames=2)
    counter.update({}, frame_no=0)
    counter.update(
        {1: (0.5, 0.02)}, frame_no=1
    )  # 두 번째 분석 프레임: 시작 때 있던 사람
    counter.update({1: (0.5, 0.3)}, frame_no=2)
    assert totals(counter) == expect()


def test_person_first_seen_on_third_frame_counts_in():
    counter = EdgeCounter(margin=MARGIN, lost_frames=LOST, start_frames=2)
    counter.update({}, frame_no=0)
    counter.update({}, frame_no=1)
    counter.update({1: (0.5, 0.02)}, frame_no=2)  # 세 번째 분석 프레임
    counter.update({1: (0.5, 0.3)}, frame_no=3)
    assert totals(counter) == expect(in_N=1)


def test_start_frames_count_from_the_first_frame_received():
    counter = EdgeCounter(margin=MARGIN, lost_frames=LOST, start_frames=2)
    counter.update({}, frame_no=10)
    counter.update({1: (0.5, 0.02)}, frame_no=11)
    counter.update({1: (0.5, 0.3)}, frame_no=12)
    assert totals(counter) == expect()


def test_skipped_slots_do_not_use_up_start_frames():
    counter = EdgeCounter(margin=MARGIN, lost_frames=LOST, start_frames=2)
    counter.update({}, frame_no=5)
    counter.update(
        {1: (0.5, 0.02)}, frame_no=7
    )  # 슬롯은 건너뛰었지만 두 번째 분석 프레임
    counter.update({1: (0.5, 0.3)}, frame_no=8)
    assert totals(counter) == expect()


def test_start_frames_do_not_affect_leaving():
    counter = EdgeCounter(margin=MARGIN, lost_frames=LOST, start_frames=2)
    counter.update({1: (0.5, 0.5)}, frame_no=0)
    counter.update({1: (0.5, 0.98)}, frame_no=1)
    idle(counter, LOST)
    assert totals(counter) == expect(out_S=1)


def test_frame_numbers_must_increase():
    counter = make()
    counter.update({}, frame_no=5)
    with pytest.raises(ValueError):
        counter.update({}, frame_no=5)


def test_people_are_tracked_independently():
    counter = make()
    counter.update({1: (0.5, 0.02), 2: (0.5, 0.98)})
    counter.update({1: (0.5, 0.2), 2: (0.5, 0.8)})
    assert totals(counter) == expect(in_N=1, in_S=1)


def test_pop_counts_resets_and_returns_a_copy():
    counter = make()
    move(counter, [(0.5, 0.02), (0.5, 0.2)])
    first = counter.pop_counts()
    first["in"]["N"] = 99
    assert totals(counter) == empty()


def test_counts_are_not_lost_when_popped_from_another_thread():
    counter = make()
    collected = {"in": 0}
    stop = threading.Event()

    def popper():
        while not stop.is_set():
            collected["in"] += sum(counter.pop_counts()["in"].values())

    thread = threading.Thread(target=popper)
    thread.start()
    try:
        for person in range(200):
            counter.update({person: (0.5, 0.02)})
            counter.update({person: (0.5, 0.3)})
    finally:
        stop.set()
        thread.join()
    collected["in"] += sum(counter.pop_counts()["in"].values())
    assert collected["in"] == 200
