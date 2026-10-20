import numpy as np
import pytest

from white_eyes_flask.analysis.boxes import dedupe_boxes, sanitize_boxes


def boxes(*rows):
    return np.array(rows, dtype=np.float32).reshape(-1, 5)


class TestSanitize:
    def test_non_finite_rows_are_dropped(self):
        result = sanitize_boxes(
            boxes(
                [10, 10, 50, 50, 0.9], [np.nan, 0, 5, 5, 0.9], [0, 0, np.inf, 5, 0.9]
            ),
            (100, 100),
        )
        assert len(result) == 1

    def test_coordinates_are_clipped_to_frame(self):
        result = sanitize_boxes(boxes([-20, -10, 150, 120, 0.8]), (100, 200))
        assert result[0, :4] == pytest.approx([0, 0, 150, 100])

    def test_box_fully_outside_frame_is_dropped(self):
        assert len(sanitize_boxes(boxes([300, 300, 400, 400, 0.9]), (100, 100))) == 0

    def test_zero_or_negative_size_is_dropped(self):
        result = sanitize_boxes(
            boxes([10, 10, 10, 50, 0.9], [20, 20, 10, 40, 0.9], [30, 30, 60, 60, 0.9]),
            (100, 100),
        )
        assert len(result) == 1

    def test_empty_input_gives_empty_float32_array(self):
        result = sanitize_boxes(np.zeros((0, 5)), (100, 100))
        assert result.shape == (0, 5)
        assert result.dtype == np.float32


class TestDedupe:
    def test_boxes_far_apart_are_kept(self):
        result = dedupe_boxes(boxes([0, 0, 20, 40, 0.9], [100, 0, 120, 40, 0.8]))
        assert len(result) == 2

    def test_box_whose_center_is_inside_a_better_box_is_dropped(self):
        # 몸통 박스 안에 다리만 잡힌 작은 박스
        big = [100, 100, 140, 220, 0.9]
        legs = [110, 190, 130, 230, 0.6]
        result = dedupe_boxes(boxes(legs, big))
        assert result.shape == (1, 5)
        assert result[0, 4] == pytest.approx(0.9)

    def test_large_overlap_relative_to_smaller_box_is_dropped(self):
        a = [0, 0, 40, 40, 0.9]
        b = [28, 0, 68, 40, 0.8]  # 중심은 서로 밖, 겹침은 작은 박스의 30%
        assert len(dedupe_boxes(boxes(a, b))) == 1

    def test_small_overlap_is_kept(self):
        a = [0, 0, 40, 40, 0.9]
        b = [34, 0, 74, 40, 0.8]  # 겹침 15%
        assert len(dedupe_boxes(boxes(a, b))) == 2

    def test_higher_score_survives_regardless_of_input_order(self):
        a = [0, 0, 40, 40, 0.5]
        b = [5, 5, 45, 45, 0.9]
        assert dedupe_boxes(boxes(a, b))[0, 4] == pytest.approx(0.9)
        assert dedupe_boxes(boxes(b, a))[0, 4] == pytest.approx(0.9)

    def test_equal_scores_keep_the_earlier_box(self):
        a = [0, 0, 40, 40, 0.8]
        b = [5, 5, 45, 45, 0.8]
        result = dedupe_boxes(boxes(a, b))
        assert result[0, :4] == pytest.approx(a[:4])

    def test_overlap_zero_disables_merging(self):
        a = [0, 0, 40, 40, 0.9]
        assert len(dedupe_boxes(boxes(a, a), overlap=0)) == 2

    def test_zero_area_box_does_not_break_the_overlap_check(self):
        with np.errstate(all="raise"):
            result = dedupe_boxes(boxes([10, 10, 10, 10, 0.9], [50, 50, 90, 90, 0.8]))
        assert len(result) == 2

    def test_one_dimensional_empty_array_is_accepted(self):
        assert dedupe_boxes(np.array([])).shape == (0, 5)
        assert sanitize_boxes(np.array([]), (10, 10)).shape == (0, 5)

    def test_empty_and_single_inputs_pass_through(self):
        assert dedupe_boxes(np.zeros((0, 5))).shape == (0, 5)
        assert len(dedupe_boxes(boxes([0, 0, 10, 10, 0.9]))) == 1

    def test_result_keeps_input_order(self):
        a = [0, 0, 20, 20, 0.5]
        b = [100, 0, 120, 20, 0.9]
        c = [200, 0, 220, 20, 0.7]
        result = dedupe_boxes(boxes(a, b, c))
        assert result[:, 0].tolist() == [0, 100, 200]
