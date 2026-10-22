"""FNC-S-008 라이브 녹화: 파일 이름 규칙·좌표 파일·ffmpeg 명령"""

import gzip
import json
from datetime import datetime
from pathlib import Path

from white_eyes_flask.analysis.result import TrackedBox
from white_eyes_flask.camera.kst import KST
from white_eyes_flask.recording.box_file import write_box_file
from white_eyes_flask.recording.ffmpeg import RecorderSettings, build_command
from white_eyes_flask.recording.naming import (
    file_stem,
    minute_start_ms,
    nearest_minute_ms,
    relative_dir,
)


def ms(hour, minute, second=0, milli=0) -> int:
    moment = datetime(2026, 10, 12, hour, minute, second, tzinfo=KST)
    return int(moment.timestamp() * 1000) + milli


def test_file_name_uses_the_start_of_the_minute_in_kst():
    assert file_stem(7, ms(14, 3, 27)) == "cam7_20261012_1403"
    assert relative_dir(7, ms(14, 3, 27)) == "recordings/live/cam7/20261012"


def test_file_name_date_follows_kst_not_utc():
    # UTC로는 10월 11일 15시 30분
    assert file_stem(7, ms(0, 30)) == "cam7_20261012_0030"
    assert relative_dir(7, ms(0, 30)).endswith("/20261012")


def test_minute_start_drops_seconds_and_millis():
    assert minute_start_ms(ms(14, 3, 59, 999)) == ms(14, 3)


def test_nearest_minute_rounds_to_the_closest_boundary():
    assert nearest_minute_ms(ms(14, 3, 59, 200)) == ms(14, 4)  # 경계 0.8초 전
    assert nearest_minute_ms(ms(14, 4, 0, 400)) == ms(14, 4)  # 경계 0.4초 뒤
    assert nearest_minute_ms(ms(14, 3, 20)) == ms(14, 3)


def test_box_file_follows_the_coordinate_file_format(tmp_path: Path):
    path = tmp_path / "a.json.gz"
    write_box_file(
        path,
        5,
        [
            (0.2, [TrackedBox(17, 0.41234, 0.221, 0.015, 0.03, 0.42, 0.23, 0.9)]),
            (0.4, []),
        ],
    )
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        body = json.load(stream)
    assert body == {
        "version": 1,
        "fps": 5,
        "frames": [
            {"t": 0.2, "b": [[17, 0.4123, 0.221, 0.015, 0.03]]},
            {"t": 0.4, "b": []},
        ],
    }


def test_box_file_clamps_coordinates_and_negative_time(tmp_path: Path):
    path = tmp_path / "a.json.gz"
    write_box_file(
        path,
        5,
        [(-0.3, [TrackedBox(1, -0.01, 1.0001, 0.5, 2.0, 0.2, 0.9, 0.9)])],
    )
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        frame = json.load(stream)["frames"][0]
    assert frame["t"] == 0.0
    assert frame["b"] == [[1, 0.0, 1.0, 0.5, 1.0]]


def test_box_file_leaves_no_temporary_file(tmp_path: Path):
    write_box_file(tmp_path / "a.json.gz", 5, [])
    assert [p.name for p in tmp_path.iterdir()] == ["a.json.gz"]


def arg_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_ffmpeg_command_follows_the_spec(tmp_path: Path):
    settings = RecorderSettings(width=1280, height=720, fps=10)
    command = build_command(
        settings, 1920, 1080, tmp_path / "seg_%05d.mp4", tmp_path / "list.csv"
    )
    assert command[0] == "ffmpeg"
    assert arg_after(command, "-s") == "1920x1080"
    assert arg_after(command, "-i") == "pipe:0"
    assert arg_after(command, "-vf") == "scale=1280:720,fps=10"
    assert arg_after(command, "-c:v") == "libx264"
    assert arg_after(command, "-f") == "rawvideo"
    assert "segment" in command
    assert arg_after(command, "-segment_time") == "60"
    assert arg_after(command, "-segment_atclocktime") == "1"
    assert arg_after(command, "-reset_timestamps") == "1"
    assert arg_after(command, "-segment_format_options") == "movflags=+faststart"
    assert arg_after(command, "-segment_list_type") == "csv"
    assert command[-1].endswith("seg_%05d.mp4")


def test_ffmpeg_command_cuts_keyframes_every_second_and_uses_input_time(tmp_path):
    command = build_command(
        RecorderSettings(fps=10), 640, 360, tmp_path / "s_%05d.mp4", tmp_path / "l.csv"
    )
    assert arg_after(command, "-g") == "10"
    assert arg_after(command, "-keyint_min") == "10"
    assert arg_after(command, "-use_wallclock_as_timestamps") == "1"
    assert arg_after(command, "-pix_fmt") in {"bgr24", "yuv420p"}


def test_ffmpeg_binary_comes_from_the_settings(tmp_path):
    command = build_command(
        RecorderSettings(ffmpeg_bin="/opt/ffmpeg"),
        640,
        360,
        tmp_path / "s_%05d.mp4",
        tmp_path / "l.csv",
    )
    assert command[0] == "/opt/ffmpeg"
