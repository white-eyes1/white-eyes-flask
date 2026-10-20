import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from white_eyes_flask.analysis.aerial_detector import VisDroneDetector

NAMES = {0: "pedestrian", 1: "people", 2: "bicycle", 3: "car"}
FRAME = np.zeros((200, 400, 3), dtype=np.uint8)


class Tensor:
    """ultralytics 결과의 텐서처럼 .cpu().numpy()만 되는 값."""

    def __init__(self, array):
        self._array = np.array(array, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self._array

    def __len__(self):
        return len(self._array)


class FakeModel:
    def __init__(self, rows=(), names=None, delay=0.0):
        self.names = NAMES if names is None else names
        self.rows = np.array(rows, dtype=np.float32).reshape(-1, 5)
        self.calls = []
        self.delay = delay
        self.active = 0
        self.max_active = 0

    def predict(self, frame, **kwargs):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        time.sleep(self.delay)
        self.calls.append(kwargs)
        self.active -= 1
        return [SimpleNamespace(boxes=_Boxes(self.rows))]


class _Boxes:
    def __init__(self, rows):
        self.xyxy = Tensor(rows[:, :4])
        self.conf = Tensor(rows[:, 4])
        self._n = len(rows)

    def __len__(self):
        return self._n


def make(rows=(), **kwargs):
    model = FakeModel(rows)
    return VisDroneDetector(model, 960, 0.3, **kwargs), model


def test_model_is_called_with_person_classes_and_settings():
    detector, model = make([[10, 10, 50, 90, 0.9]])
    detector.detect(FRAME)
    call = model.calls[0]
    assert call["classes"] == [0, 1]
    assert call["imgsz"] == 960
    assert call["conf"] == 0.3
    assert call["agnostic_nms"] is True
    assert call["device"] == "cpu"
    assert call["max_det"] >= 1000


def test_returns_pixel_boxes_with_confidence():
    detector, _ = make([[10, 10, 50, 90, 0.9], [200, 20, 240, 100, 0.5]])
    result = detector.detect(FRAME)
    assert result.shape == (2, 5)
    assert result.dtype == np.float32
    assert result[0] == pytest.approx([10, 10, 50, 90, 0.9])


def test_no_detection_gives_empty_array():
    detector, _ = make()
    assert detector.detect(FRAME).shape == (0, 5)


def test_boxes_are_clipped_and_bad_values_dropped():
    detector, _ = make(
        [[-30, -5, 60, 90, 0.9], [100, 100, np.nan, 150, 0.9], [300, 20, 300, 80, 0.9]]
    )
    result = detector.detect(FRAME)
    assert result.shape == (1, 5)
    assert result[0, :4] == pytest.approx([0, 0, 60, 90])


def test_duplicate_boxes_of_one_person_are_merged():
    detector, _ = make([[100, 20, 140, 140, 0.9], [105, 100, 135, 150, 0.6]])
    assert len(detector.detect(FRAME)) == 1


def test_dedupe_can_be_turned_off():
    detector, _ = make(
        [[100, 20, 140, 140, 0.9], [105, 100, 135, 150, 0.6]], dedupe_overlap=0
    )
    assert len(detector.detect(FRAME)) == 2


def test_model_without_person_classes_is_rejected():
    with pytest.raises(ValueError):
        VisDroneDetector(FakeModel(names={0: "person"}), 960, 0.3)


def test_person_class_numbers_come_from_model_names():
    model = FakeModel(names={0: "car", 4: "pedestrian", 7: "people"})
    VisDroneDetector(model, 960, 0.3).detect(FRAME)
    assert model.calls[0]["classes"] == [4, 7]


def test_class_names_given_as_a_list_are_accepted():
    model = FakeModel(names=["pedestrian", "people", "car"])
    VisDroneDetector(model, 960, 0.3).detect(FRAME)
    assert model.calls[0]["classes"] == [0, 1]


def test_frame_must_be_three_channel_image():
    detector, _ = make()
    with pytest.raises(ValueError):
        detector.detect(np.zeros((200, 400), dtype=np.uint8))


def test_inference_is_one_at_a_time_when_shared():
    model = FakeModel([[10, 10, 50, 90, 0.9]], delay=0.02)
    detector = VisDroneDetector(model, 960, 0.3)
    threads = [
        threading.Thread(target=detector.detect, args=(FRAME,)) for _ in range(6)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert len(model.calls) == 6
    assert model.max_active == 1


def test_missing_weights_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        VisDroneDetector.from_weights(str(tmp_path / "none.pt"), 960, 0.3)
