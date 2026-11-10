import math
from datetime import datetime, timedelta, timezone

# 서버 시간대와 상관없이 API 시각은 항상 한국 표준시(+09:00)로 쓴다
KST = timezone(timedelta(hours=9))


def format_at(epoch_seconds: float) -> str:
    """1초 결과의 at. 초 단위로 내림한다(예: 2026-10-12T14:03:05+09:00)."""
    moment = datetime.fromtimestamp(math.floor(epoch_seconds), KST)
    return moment.isoformat(timespec="seconds")


def format_started_at(epoch_ms: int) -> str:
    """녹화 startedAt. 밀리초까지 쓴다(예: 2026-10-12T14:03:00.000+09:00)."""
    moment = datetime.fromtimestamp(epoch_ms // 1000, KST)
    moment = moment.replace(microsecond=(epoch_ms % 1000) * 1000)
    return moment.isoformat(timespec="milliseconds")
