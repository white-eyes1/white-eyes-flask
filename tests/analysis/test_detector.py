import numpy as np
import pytest
import torch

from white_eyes_flask.analysis.detector import YoloCrowdDetector

STRIDE = 32
IMGSZ = 960


class FakeModel:
    """letterbox된 입력 크기를 기록하고, 정해 둔 박스를 YOLO 출력 형식으로 돌려준다.

    박스는 입력 이미지 좌표의 (cx, cy, w, h, obj, cls)다.
    """

    def __init__(self, rows):
        self.rows = rows
        self.input_shape = None

    def __call__(self, tensor):
        self.input_shape = tuple(tensor.shape)
        prediction = torch.tensor([self.rows], dtype=torch.float32)
        return prediction, None


def make_detector(rows, conf=0.3):
    model = FakeModel(rows)
    return YoloCrowdDetector(model, STRIDE, IMGSZ, conf), model


def letterbox_pad(height, width):
    """vendor letterbox(auto=True)와 같은 결과를 따로 계산한다: (비율, 왼쪽 패딩, 위쪽 패딩)."""
    ratio = min(IMGSZ / height, IMGSZ / width)
    new_w, new_h = round(width * ratio), round(height * ratio)
    pad_w = (IMGSZ - new_w) % STRIDE / 2
    pad_h = (IMGSZ - new_h) % STRIDE / 2
    return ratio, pad_w, pad_h


@pytest.mark.parametrize(
    "height,width", [(1080, 1920), (1920, 1080), (720, 1280), (500, 500)]
)
def test_boxes_are_mapped_back_to_original_pixels(height, width):
    original = (width * 0.30, height * 0.40, width * 0.30 + 24, height * 0.40 + 24)
    ratio, pad_w, pad_h = letterbox_pad(height, width)
    cx = (original[0] + original[2]) / 2 * ratio + pad_w
    cy = (original[1] + original[3]) / 2 * ratio + pad_h
    w, h = 24 * ratio, 24 * ratio
    detector, _ = make_detector([[cx, cy, w, h, 0.9, 0.9]])
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    result = detector.detect(frame)

    assert result.shape == (1, 5)
    assert result.dtype == np.float32
    assert result[0, :4] == pytest.approx(original, abs=0.6)
    assert result[0, 4] == pytest.approx(0.81, abs=1e-3)  # obj * cls


def test_input_is_resized_with_padding_to_multiple_of_stride():
    detector, model = make_detector([[100, 100, 10, 10, 0.9, 0.9]])
    detector.detect(np.zeros((1080, 1920, 3), dtype=np.uint8))
    _, channels, height, width = model.input_shape
    assert channels == 3
    assert width == IMGSZ
    assert height % STRIDE == 0 and height < IMGSZ


@pytest.mark.parametrize(
    "height,width",
    [(1500, 2000), (765, 1360), (1080, 1920), (1920, 1080), (720, 1280), (500, 500)],
)
@pytest.mark.parametrize("imgsz", [640, 800, 960, 1280])
def test_input_is_a_multiple_of_32_even_when_the_model_reports_stride_16(
    height, width, imgsz
):
    # YOLO-CROWD는 model.stride.max()가 16이지만 내부에서 32배로 줄이는 층이 있어서
    # 입력의 가로·세로가 32의 배수가 아니면 텐서 크기가 어긋나 오류가 난다.
    model = FakeModel([[100, 100, 10, 10, 0.9, 0.9]])
    detector = YoloCrowdDetector(model, 16, imgsz, 0.3)
    detector.detect(np.zeros((height, width, 3), dtype=np.uint8))
    _, _, input_height, input_width = model.input_shape
    assert input_height % 32 == 0
    assert input_width % 32 == 0
    assert max(input_height, input_width) <= imgsz


def test_boxes_are_mapped_back_correctly_when_the_model_reports_stride_16():
    height, width = 765, 1360
    original = (width * 0.30, height * 0.40, width * 0.30 + 24, height * 0.40 + 24)
    ratio, pad_w, pad_h = letterbox_pad(height, width)
    cx = (original[0] + original[2]) / 2 * ratio + pad_w
    cy = (original[1] + original[3]) / 2 * ratio + pad_h
    model = FakeModel([[cx, cy, 24 * ratio, 24 * ratio, 0.9, 0.9]])
    detector = YoloCrowdDetector(model, 16, IMGSZ, 0.3)
    result = detector.detect(np.zeros((height, width, 3), dtype=np.uint8))
    assert result[0, :4] == pytest.approx(original, abs=0.6)


def test_low_confidence_boxes_are_removed():
    detector, _ = make_detector(
        [[100, 100, 10, 10, 0.9, 0.9], [300, 300, 10, 10, 0.2, 0.9]], conf=0.3
    )
    result = detector.detect(np.zeros((540, 960, 3), dtype=np.uint8))
    assert len(result) == 1


def test_overlapping_boxes_are_merged_by_nms():
    detector, _ = make_detector(
        [[100, 100, 20, 20, 0.9, 0.9], [101, 100, 20, 20, 0.8, 0.9]]
    )
    result = detector.detect(np.zeros((540, 960, 3), dtype=np.uint8))
    assert len(result) == 1


def test_no_detection_gives_empty_array():
    detector, _ = make_detector([[100, 100, 10, 10, 0.1, 0.1]])
    result = detector.detect(np.zeros((540, 960, 3), dtype=np.uint8))
    assert result.shape == (0, 5)


def test_frame_must_be_three_channel_image():
    detector, _ = make_detector([[100, 100, 10, 10, 0.9, 0.9]])
    with pytest.raises(ValueError):
        detector.detect(np.zeros((540, 960), dtype=np.uint8))


def test_missing_weights_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        YoloCrowdDetector.from_weights(str(tmp_path / "none.pt"), IMGSZ, 0.3)
