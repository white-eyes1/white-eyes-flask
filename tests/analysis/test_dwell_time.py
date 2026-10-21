"""FNC-S-005 체류 시간 계산 (EdgeCounter)"""

import pytest

from white_eyes_flask.analysis.edge_counter import EdgeCounter

MARGIN = 0.05
LOST = 5
INSIDE = (0.5, 0.5)
BAND_N = (0.5, 0.02)


def make(fps=None):
    return EdgeCounter(margin=MARGIN, lost_frames=LOST, fps=fps)


def see(counter, point, frames, track_id=1):
    """한 사람이 같은 자리에서 frames개 프레임 동안 보인다."""
    for _ in range(frames):
        counter.update({track_id: point})


def idle(counter, frames):
    for _ in range(frames):
        counter.update({})


def dwell(counter):
    return counter.pop_counts()["dwell_sec"]


def test_dwell_is_last_seen_minus_first_seen_in_seconds():
    counter = make()
    see(counter, INSIDE, 10)  # 프레임 0~9
    see(counter, BAND_N, 1)  # 프레임 10에서 띠로 나감
    idle(counter, LOST)
    assert dwell(counter) == [2.0]  # (10 - 0) / 5프레임


def test_dwell_is_not_known_until_out_is_confirmed():
    counter = make()
    see(counter, INSIDE, 10)
    see(counter, BAND_N, 1)
    idle(counter, LOST - 1)
    assert dwell(counter) == []
    idle(counter, 1)
    assert dwell(counter) == [2.0]


def test_dwell_does_not_include_the_frames_after_last_seen():
    counter = make()
    see(counter, INSIDE, 5)
    see(counter, BAND_N, 1)  # 마지막으로 본 프레임 5
    idle(counter, LOST + 20)  # 한참 뒤에 확정돼도 영향 없음
    assert dwell(counter) == [1.0]


def test_dwell_uses_fps_when_given():
    counter = make(fps=10)
    see(counter, INSIDE, 10)
    see(counter, BAND_N, 1)
    idle(counter, LOST)
    assert dwell(counter) == [1.0]  # (10 - 0) / 10프레임


def test_fps_defaults_to_lost_frames():
    counter = make()
    see(counter, INSIDE, 5)
    see(counter, BAND_N, 1)
    idle(counter, LOST)
    assert dwell(counter) == [1.0]


def test_person_who_entered_from_a_band_counts_from_first_sighting():
    counter = make()
    see(counter, BAND_N, 2)  # 프레임 0, 1: 띠에서 처음 봄
    see(counter, INSIDE, 3)  # 프레임 2~4
    see(counter, (0.5, 0.98), 1)  # 프레임 5: 남쪽 띠로 나감
    idle(counter, LOST)
    assert dwell(counter) == [1.0]


def test_person_vanishing_inside_has_no_dwell():
    counter = make()
    see(counter, INSIDE, 10)
    idle(counter, LOST + 3)
    assert dwell(counter) == []


def test_person_only_seen_in_a_band_has_no_dwell():
    counter = make()
    see(counter, BAND_N, 4)
    idle(counter, LOST + 3)
    assert dwell(counter) == []


def test_person_who_returned_inside_has_no_dwell_while_still_tracked():
    counter = make()
    see(counter, INSIDE, 3)
    see(counter, BAND_N, 1)
    see(counter, INSIDE, 2)  # 나가다 돌아옴
    idle(counter, LOST)
    assert dwell(counter) == []


def test_short_gap_before_lost_frames_keeps_the_same_person():
    counter = make()
    see(counter, INSIDE, 5)  # 프레임 0~4
    see(counter, BAND_N, 1)  # 프레임 5
    idle(counter, LOST - 1)  # 프레임 6~9 안 보임
    see(counter, BAND_N, 1)  # 프레임 10에 다시 보임
    idle(counter, LOST)
    assert dwell(counter) == [2.0]  # (10 - 0) / 5


def test_same_track_id_after_lost_is_a_new_person_with_new_first_seen():
    counter = make()
    see(counter, INSIDE, 3)
    see(counter, BAND_N, 1)  # 프레임 3
    idle(counter, LOST + 2)  # 확정
    first = dwell(counter)
    see(counter, INSIDE, 5)  # 같은 번호로 다시 나타남(새 사람)
    see(counter, BAND_N, 1)
    idle(counter, LOST)
    assert first == [0.6]
    assert dwell(counter) == [1.0]


def test_each_confirmed_person_adds_one_entry_in_confirmation_order():
    counter = make()
    for _ in range(4):
        counter.update({1: INSIDE, 2: INSIDE})
    counter.update({1: BAND_N, 2: INSIDE})  # 1번이 먼저 나감
    for _ in range(2):
        counter.update({2: INSIDE})
    counter.update({2: (0.5, 0.98)})  # 2번도 나감
    idle(counter, LOST)
    assert dwell(counter) == [0.8, 1.4]


def test_pop_counts_clears_dwell_list():
    counter = make()
    see(counter, INSIDE, 5)
    see(counter, BAND_N, 1)
    idle(counter, LOST)
    assert counter.pop_counts()["dwell_sec"] == [1.0]
    assert counter.pop_counts()["dwell_sec"] == []


def test_returned_dwell_list_is_a_copy():
    counter = make()
    see(counter, INSIDE, 5)
    see(counter, BAND_N, 1)
    idle(counter, LOST)
    first = counter.pop_counts()["dwell_sec"]
    first.append(99.0)
    assert counter.pop_counts()["dwell_sec"] == []


def test_skipped_slots_count_as_elapsed_time():
    counter = make()
    counter.update({1: INSIDE}, frame_no=0)
    counter.update({1: INSIDE}, frame_no=5)  # 1~4번 슬롯이 오류로 건너뜀
    counter.update({1: BAND_N}, frame_no=10)
    counter.update({}, frame_no=15)
    assert dwell(counter) == [2.0]


@pytest.mark.parametrize("fps", [0, -5])
def test_fps_must_be_positive(fps):
    with pytest.raises(ValueError):
        EdgeCounter(margin=MARGIN, lost_frames=LOST, fps=fps)
