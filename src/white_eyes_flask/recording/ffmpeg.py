"""FNC-S-008 녹화용 ffmpeg 명령

프레임은 stdin 파이프(rawvideo)로 넘기고, 시계 분 단위 mp4로 자른다. 파일 이름은 순번이고
(서버 시간대와 ffmpeg의 strftime 시간대에 기대지 않으려는 것) 완성된 파일은 segment list(CSV)에 올라온다.
"""

import re
from dataclasses import dataclass
from pathlib import Path

# 조각 하나의 길이(초). 녹화 파일은 1분 단위다.
SEGMENT_SEC = 60
SEGMENT_NAME = re.compile(r"_(\d{5,})\.mp4$")  # %05d는 99999를 넘으면 자리가 늘어난다


@dataclass(frozen=True)
class RecorderSettings:
    """녹화 설정."""

    # 출력 영상 크기. 입력 프레임 크기가 달라도 이 크기로 맞춘다.
    width: int = 1280
    height: int = 720
    # 출력 영상의 초당 프레임 수(분석 fps와 별개다).
    fps: int = 10
    # 좌표 파일의 fps 값(분석 프레임 수). 영상 fps와 다르다
    analyze_fps: int = 5
    # 완성 전 조각 파일을 두는 로컬 임시 폴더(공유 폴더로 옮기기 전).
    tmp_dir: str = "/var/lib/analyzer/tmp"
    # ffmpeg 실행 파일 이름 또는 경로.
    ffmpeg_bin: str = "ffmpeg"


def segment_pattern(tmp_dir: Path, camera_id: int, token: str) -> Path:
    """ffmpeg가 조각을 쓸 파일 이름 패턴.

    %05d는 0부터 늘어나는 조각 순번이다. token(첫 프레임 시각)을 넣어 같은 캠의 다른 녹화와 이름이 겹치지 않게 한다.
    """
    return tmp_dir / f"cam{camera_id}_{token}_%05d.mp4"


def segment_list_path(tmp_dir: Path, camera_id: int, token: str) -> Path:
    """ffmpeg가 완성한 조각을 한 줄씩 적는 목록(CSV) 경로. 녹화기가 이 파일을 읽어 닫힌 조각을 안다."""
    return tmp_dir / f"cam{camera_id}_{token}.csv"


def build_command(
    settings: RecorderSettings,
    input_width: int,
    input_height: int,
    pattern: Path,
    segment_list: Path,
) -> list[str]:
    """녹화용 ffmpeg 명령(인자 목록)을 만든다.

    input_width·input_height는 stdin 파이프로 넘기는 원본 프레임의 크기다. 출력 크기와 fps는 settings를 따른다.
    """
    # 1초마다 키프레임: 분 경계에서 거의 정확히 자르고 구간 이동도 빠르다
    gop = str(settings.fps)
    return [
        settings.ffmpeg_bin,
        # 배너를 끄고 오류만 출력한다(오류는 파일로 받아 죽은 원인을 확인하는 데 쓴다).
        "-hide_banner",
        "-loglevel",
        "error",
        # 같은 이름의 파일이 있으면 묻지 않고 덮어쓴다.
        "-y",
        # 들어온 시각대로 시간을 매겨 분석이 느려도 영상 길이가 늘거나 줄지 않게 한다
        "-use_wallclock_as_timestamps",
        "1",
        # 입력은 가공 전 프레임 데이터(rawvideo)이고 OpenCV가 주는 BGR 순서(bgr24)다. 헤더가 없어 크기(-s)를 따로 알려 준다.
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{input_width}x{input_height}",
        # 입력은 표준 입력 파이프(pipe:0)다.
        "-i",
        "pipe:0",
        # 필터: 출력 크기로 맞추고 초당 프레임 수를 맞춘다.
        "-vf",
        f"scale={settings.width}:{settings.height},fps={settings.fps}",
        # H.264(libx264)로 인코딩한다. veryfast는 인코딩 속도를 우선한 설정이고, yuv420p는 브라우저 같은 일반 플레이어가 재생할 수 있는 화소 형식이다.
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        # 키프레임 간격을 fps와 같게 해(-keyint_min도 같은 값) 1초마다 키프레임을 둔다.
        "-g",
        gop,
        "-keyint_min",
        gop,
        # 출력은 segment 형식으로 여러 파일(조각)로 나눈다.
        "-f",
        "segment",
        # 조각 길이(초).
        "-segment_time",
        str(SEGMENT_SEC),
        # 1이면 조각 길이만큼 흐른 시점이 아니라 시계(벽시계)의 분 경계에 맞춰 자른다.
        "-segment_atclocktime",
        "1",
        # 각 조각의 시간을 0부터 다시 시작해 파일 하나만으로 재생되게 한다.
        "-reset_timestamps",
        "1",
        # faststart: 재생에 필요한 정보(moov)를 파일 앞으로 옮겨 웹에서 바로 재생·구간 이동이 되게 한다.
        "-segment_format_options",
        "movflags=+faststart",
        # 완성된 조각을 CSV 목록에 한 줄씩 올린다(녹화기가 읽는다).
        "-segment_list",
        str(segment_list),
        "-segment_list_type",
        "csv",
        # 마지막 인자: 조각 파일 이름 패턴.
        str(pattern),
    ]
