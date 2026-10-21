import threading

import pytest

from white_eyes_flask.analysis import detector_provider
from white_eyes_flask.analysis.detector_provider import (
    DetectorUnavailableError,
    LazyDetectorProvider,
)


@pytest.fixture
def loads(monkeypatch):
    """YoloCrowdDetector.from_weights를 가짜로 바꾸고 호출 기록을 돌려준다."""
    record = {"calls": [], "fail": None}

    def fake_from_weights(weights, imgsz, conf_thres):
        record["calls"].append((weights, imgsz, conf_thres))
        if record["fail"]:
            raise record["fail"]
        return object()

    monkeypatch.setattr(
        detector_provider.YoloCrowdDetector, "from_weights", fake_from_weights
    )
    return record


def test_no_weights_configured_is_unavailable():
    provider = LazyDetectorProvider(None, 960, 0.3)
    with pytest.raises(DetectorUnavailableError):
        provider()


def test_missing_weights_file_is_unavailable(tmp_path):
    provider = LazyDetectorProvider(str(tmp_path / "none.pt"), 960, 0.3)
    with pytest.raises(DetectorUnavailableError):
        provider()


def test_detector_is_loaded_once_and_shared(tmp_path, loads):
    weights = tmp_path / "yolo-crowd.pt"
    weights.write_bytes(b"x")
    provider = LazyDetectorProvider(str(weights), 960, 0.3)
    first, second = provider(), provider()
    assert first is second
    assert loads["calls"] == [(str(weights), 960, 0.3)]


def test_nothing_is_loaded_until_first_use(tmp_path, loads):
    weights = tmp_path / "yolo-crowd.pt"
    weights.write_bytes(b"x")
    LazyDetectorProvider(str(weights), 960, 0.3)
    assert loads["calls"] == []


def test_concurrent_first_use_loads_only_once(tmp_path, loads):
    weights = tmp_path / "yolo-crowd.pt"
    weights.write_bytes(b"x")
    provider = LazyDetectorProvider(str(weights), 960, 0.3)
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(provider())) for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert len(loads["calls"]) == 1
    assert len({id(r) for r in results}) == 1


def test_load_failure_is_unavailable_and_retried_next_time(tmp_path, loads):
    weights = tmp_path / "yolo-crowd.pt"
    weights.write_bytes(b"x")
    provider = LazyDetectorProvider(str(weights), 960, 0.3)
    loads["fail"] = RuntimeError("깨진 가중치")
    with pytest.raises(DetectorUnavailableError):
        provider()
    loads["fail"] = None
    assert provider() is not None  # 파일을 고치면 다음 시작 때 다시 시도
    assert len(loads["calls"]) == 2
