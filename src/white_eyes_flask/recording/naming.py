"""FNC-S-008 라이브 녹화 파일 이름·폴더 규칙(conventions '공유 폴더 구조')"""

from datetime import datetime

from white_eyes_flask.camera.kst import KST

LIVE_DIR = "recordings/live"
MINUTE_MS = 60_000


def minute_start_ms(epoch_ms: int) -> int:
    """그 시각이 속한 분의 시작(밀리초)."""
    return epoch_ms - epoch_ms % MINUTE_MS


def nearest_minute_ms(epoch_ms: int) -> int:
    """가장 가까운 분 경계(밀리초). 분 경계 근처에서 끊긴 파일의 이름 시각을 정할 때 쓴다."""
    return (epoch_ms + MINUTE_MS // 2) // MINUTE_MS * MINUTE_MS


def _moment(epoch_ms: int) -> datetime:
    return datetime.fromtimestamp(epoch_ms // 1000, KST)


def file_stem(camera_id: int, started_ms: int) -> str:
    """파일 이름 시각 = 그 분의 시작(KST). 예: cam7_20261012_1403"""
    return f"cam{camera_id}_{_moment(started_ms):%Y%m%d_%H%M}"


def relative_dir(camera_id: int, started_ms: int) -> str:
    """공유 폴더 기준 폴더. 예: recordings/live/cam7/20261012"""
    return f"{LIVE_DIR}/cam{camera_id}/{_moment(started_ms):%Y%m%d}"
