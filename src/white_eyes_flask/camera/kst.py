"""API에 쓰는 시각 문자열 변환(FNC-S-006·010).

분석 서버가 관리자 앱에 보내는 모든 시각은 서버 시간대와 무관하게 한국 표준시(+09:00)다.
"""

import math
from datetime import datetime, timedelta, timezone

# 서버 시간대와 상관없이 API 시각은 항상 한국 표준시(+09:00)로 쓴다
KST = timezone(timedelta(hours=9))


def format_at(epoch_seconds: float) -> str:
    """1초 결과의 at. 초 단위로 내림한다(예: 2026-10-12T14:03:05+09:00)."""
    # 소수 초는 올림·반올림하지 않고 버린다. 같은 초의 결과가 항상 같은 at을 갖게 하려는 것이다.
    moment = datetime.fromtimestamp(math.floor(epoch_seconds), KST)
    # 초 단위까지만 쓴다(밀리초 없음).
    return moment.isoformat(timespec="seconds")


def format_started_at(epoch_ms: int) -> str:
    """녹화 startedAt. 밀리초까지 쓴다(예: 2026-10-12T14:03:00.000+09:00)."""
    # 정수 초 부분을 먼저 시각으로 바꾼다.
    moment = datetime.fromtimestamp(epoch_ms // 1000, KST)
    # 남은 밀리초를 마이크로초로 환산해 붙인다. 실수 나눗셈을 쓰지 않아 오차가 없다.
    moment = moment.replace(microsecond=(epoch_ms % 1000) * 1000)
    return moment.isoformat(timespec="milliseconds")
