import gzip
import json
import threading
import time
from pathlib import Path

import pytest

from white_eyes_flask.analysis.dummy_cache import DummyCache
from white_eyes_flask.analysis.result import TrackedBox

SOURCE = "dummy/videos/festival.mp4"


@pytest.fixture
def cache(tmp_path: Path) -> DummyCache:
    return DummyCache(cache_dir=tmp_path / "cache", media_root=tmp_path / "media")


def box(track_id, x=0.1, y=0.2, w=0.05, h=0.1):
    return TrackedBox(
        track_id=track_id, x=x, y=y, w=w, h=h, cx=x + w / 2, cy=y + h / 2, conf=0.9
    )


def publish(cache: DummyCache, seconds=None):
    with cache.writer(SOURCE, fps=5) as writer:
        writer.add_frame(0.0, [box(1)])
        writer.commit(seconds if seconds is not None else [{"sec": 0}])


def test_file_name_is_the_video_name_without_extension(cache):
    assert cache.name("dummy/videos/festival.mp4") == "festival"
    assert cache.name("dummy/videos/sub/a.b.mp4") == "a.b"


def test_paths_follow_the_spec_layout(cache, tmp_path):
    assert cache.result_path(SOURCE) == tmp_path / "cache" / "festival.json"
    assert (
        cache.boxes_path(SOURCE)
        == tmp_path / "media" / "dummy" / "boxes" / "festival.json.gz"
    )
    assert cache.boxes_relative_path(SOURCE) == "dummy/boxes/festival.json.gz"


def test_has_is_false_before_anything_is_written(cache):
    assert cache.has(SOURCE) is False


def test_commit_publishes_both_files_and_has_becomes_true(cache):
    publish(cache)
    assert cache.has(SOURCE) is True
    assert cache.result_path(SOURCE).is_file()
    assert cache.boxes_path(SOURCE).is_file()


def test_has_requires_both_files(cache):
    publish(cache)
    cache.boxes_path(SOURCE).unlink()
    assert cache.has(SOURCE) is False


def test_result_file_holds_seconds_and_fps(cache):
    seconds = [{"sec": 0, "personCount": 3, "in": {}, "out": {}, "dwellSec": [1.0]}]
    publish(cache, seconds)
    data = json.loads(cache.result_path(SOURCE).read_text(encoding="utf-8"))
    assert data == {"version": 1, "fps": 5, "seconds": seconds}


def test_boxes_file_follows_the_coordinate_format(cache):
    with cache.writer(SOURCE, fps=5) as writer:
        writer.add_frame(0.2, [box(17, 0.41234, 0.22101, 0.015, 0.03), box(None)])
        writer.add_frame(0.4, [])
        writer.commit([])
    with gzip.open(cache.boxes_path(SOURCE), "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    assert data == {
        "version": 1,
        "fps": 5,
        "frames": [
            {
                "t": 0.2,
                "b": [[17, 0.4123, 0.221, 0.015, 0.03], [None, 0.1, 0.2, 0.05, 0.1]],
            },
            {"t": 0.4, "b": []},
        ],
    }


def test_boxes_file_with_no_frames_is_valid_json(cache):
    with cache.writer(SOURCE, fps=5) as writer:
        writer.commit([])
    with gzip.open(cache.boxes_path(SOURCE), "rt", encoding="utf-8") as stream:
        assert json.load(stream)["frames"] == []


def test_leaving_the_writer_without_commit_removes_temp_files(cache, tmp_path):
    with cache.writer(SOURCE, fps=5) as writer:
        writer.add_frame(0.0, [box(1)])
    assert cache.has(SOURCE) is False
    leftovers = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert leftovers == []


def test_exception_inside_writer_removes_temp_files_and_keeps_old_cache(
    cache, tmp_path
):
    publish(cache, [{"sec": 0}])
    with pytest.raises(RuntimeError):
        with cache.writer(SOURCE, fps=5) as writer:
            writer.add_frame(0.0, [box(1)])
            raise RuntimeError("분석 실패")
    assert cache.has(SOURCE) is True
    names = sorted(p.name for p in tmp_path.rglob("*") if p.is_file())
    assert names == ["festival.json", "festival.json.gz"]


def test_nothing_is_visible_until_commit(cache):
    with cache.writer(SOURCE, fps=5) as writer:
        writer.add_frame(0.0, [box(1)])
        assert cache.has(SOURCE) is False
        assert not cache.boxes_path(SOURCE).exists()
        writer.commit([])
    assert cache.has(SOURCE) is True


def test_lock_is_exclusive_per_video_name(cache):
    stop = threading.Event()
    entered = []
    with cache.lock(SOURCE, stop) as first:
        assert first is True

        def other():
            with cache.lock(SOURCE, stop) as second:
                entered.append(second)

        thread = threading.Thread(target=other)
        thread.start()
        time.sleep(0.2)
        assert entered == []  # 첫 번째가 놓을 때까지 기다림
    thread.join(timeout=3)
    assert entered == [True]


def test_different_videos_do_not_block_each_other(cache):
    stop = threading.Event()
    with cache.lock("dummy/videos/a.mp4", stop) as first:
        with cache.lock("dummy/videos/b.mp4", stop) as second:
            assert (first, second) == (True, True)


def test_waiting_for_lock_gives_up_when_stop_is_set(cache):
    stop = threading.Event()
    results = []
    with cache.lock(SOURCE, threading.Event()):

        def other():
            with cache.lock(SOURCE, stop) as acquired:
                results.append(acquired)

        thread = threading.Thread(target=other)
        thread.start()
        time.sleep(0.1)
        stop.set()
        thread.join(timeout=3)
    assert results == [False]
    assert not thread.is_alive()
