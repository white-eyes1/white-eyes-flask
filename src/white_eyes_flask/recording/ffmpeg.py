"""FNC-S-008 녹화용 ffmpeg 명령

프레임은 stdin 파이프(rawvideo)로 넘기고, 시계 분 단위 mp4로 자른다. 파일 이름은 순번이고
(서버 시간대와 ffmpeg의 strftime 시간대에 기대지 않으려는 것) 완성된 파일은 segment list(CSV)에 올라온다.
"""

import re
from dataclasses import dataclass
from pathlib import Path

SEGMENT_SEC = 60
SEGMENT_NAME = re.compile(r"_(\d{5,})\.mp4$")  # %05d는 99999를 넘으면 자리가 늘어난다


@dataclass(frozen=True)
class RecorderSettings:
    width: int = 1280
    height: int = 720
    fps: int = 10
    # 좌표 파일의 fps 값(분석 프레임 수). 영상 fps와 다르다
    analyze_fps: int = 5
    tmp_dir: str = "/var/lib/analyzer/tmp"
    ffmpeg_bin: str = "ffmpeg"


def segment_pattern(tmp_dir: Path, camera_id: int, token: str) -> Path:
    return tmp_dir / f"cam{camera_id}_{token}_%05d.mp4"


def segment_list_path(tmp_dir: Path, camera_id: int, token: str) -> Path:
    return tmp_dir / f"cam{camera_id}_{token}.csv"


def build_command(
    settings: RecorderSettings,
    input_width: int,
    input_height: int,
    pattern: Path,
    segment_list: Path,
) -> list[str]:
    # 1초마다 키프레임: 분 경계에서 거의 정확히 자르고 구간 이동도 빠르다
    gop = str(settings.fps)
    return [
        settings.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        # 들어온 시각대로 시간을 매겨 분석이 느려도 영상 길이가 늘거나 줄지 않게 한다
        "-use_wallclock_as_timestamps",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{input_width}x{input_height}",
        "-i",
        "pipe:0",
        "-vf",
        f"scale={settings.width}:{settings.height},fps={settings.fps}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-g",
        gop,
        "-keyint_min",
        gop,
        "-f",
        "segment",
        "-segment_time",
        str(SEGMENT_SEC),
        "-segment_atclocktime",
        "1",
        "-reset_timestamps",
        "1",
        "-segment_format_options",
        "movflags=+faststart",
        "-segment_list",
        str(segment_list),
        "-segment_list_type",
        "csv",
        str(pattern),
    ]
