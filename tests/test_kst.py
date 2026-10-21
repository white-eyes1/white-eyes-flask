from datetime import datetime

from white_eyes_flask.camera.kst import KST, format_at, format_started_at


def epoch(year, month, day, hour, minute, second, micro=0) -> float:
    return datetime(
        year, month, day, hour, minute, second, micro, tzinfo=KST
    ).timestamp()


def test_at_is_whole_seconds_in_kst():
    assert format_at(epoch(2026, 10, 12, 14, 3, 5)) == "2026-10-12T14:03:05+09:00"


def test_at_drops_the_fraction_of_a_second():
    assert (
        format_at(epoch(2026, 10, 12, 14, 3, 5, 987000)) == "2026-10-12T14:03:05+09:00"
    )


def test_started_at_has_milliseconds_and_kst_offset():
    millis = int(epoch(2026, 10, 12, 14, 3, 0) * 1000)
    assert format_started_at(millis) == "2026-10-12T14:03:00.000+09:00"


def test_started_at_keeps_milliseconds():
    millis = int(epoch(2026, 10, 12, 14, 3, 0) * 1000) + 250
    assert format_started_at(millis) == "2026-10-12T14:03:00.250+09:00"


def test_kst_is_used_regardless_of_server_timezone():
    # UTC 05:03:05는 KST 14:03:05
    utc = datetime(2026, 10, 12, 5, 3, 5, tzinfo=__import__("datetime").timezone.utc)
    assert format_at(utc.timestamp()) == "2026-10-12T14:03:05+09:00"
