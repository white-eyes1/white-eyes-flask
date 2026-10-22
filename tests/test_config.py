"""관리자 앱 내부 주소 설정(ADMIN_INTERNAL_BASE_URL)"""

from pathlib import Path

from white_eyes_flask.config import Config

DEFAULT_URL = "http://localhost:8082/admin"
ENV_EXAMPLE = Path(__file__).resolve().parent.parent / ".env.example"


def test_admin_internal_base_url_default_includes_the_admin_context_path(monkeypatch):
    monkeypatch.delenv("ADMIN_INTERNAL_BASE_URL", raising=False)
    assert Config.from_env().admin_internal_base_url == DEFAULT_URL


def test_admin_internal_base_url_dataclass_default_matches_the_env_default():
    field = Config.__dataclass_fields__["admin_internal_base_url"]
    assert field.default == DEFAULT_URL


def test_admin_internal_base_url_can_be_set_from_the_environment(monkeypatch):
    monkeypatch.setenv("ADMIN_INTERNAL_BASE_URL", "http://admin-vm:8082/admin")
    assert Config.from_env().admin_internal_base_url == "http://admin-vm:8082/admin"


def test_env_example_shows_the_admin_context_path():
    lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    assert f"ADMIN_INTERNAL_BASE_URL={DEFAULT_URL}" in lines


def test_record_settings_default_to_the_spec_values(monkeypatch):
    for name in (
        "RECORD_WIDTH",
        "RECORD_HEIGHT",
        "RECORD_FPS",
        "RECORD_TMP_DIR",
        "FFMPEG_BIN",
    ):
        monkeypatch.delenv(name, raising=False)
    config = Config.from_env()
    assert (config.record_width, config.record_height, config.record_fps) == (
        1280,
        720,
        10,
    )
    assert config.record_tmp_dir == "/var/lib/analyzer/tmp"
    assert config.ffmpeg_bin == "ffmpeg"


def test_record_settings_can_be_set_from_the_environment(monkeypatch):
    monkeypatch.setenv("RECORD_WIDTH", "640")
    monkeypatch.setenv("RECORD_HEIGHT", "360")
    monkeypatch.setenv("RECORD_FPS", "5")
    monkeypatch.setenv("RECORD_TMP_DIR", "/tmp/rec")
    monkeypatch.setenv("FFMPEG_BIN", "/opt/ffmpeg")
    config = Config.from_env()
    assert (config.record_width, config.record_height, config.record_fps) == (
        640,
        360,
        5,
    )
    assert config.record_tmp_dir == "/tmp/rec"
    assert config.ffmpeg_bin == "/opt/ffmpeg"


def test_live_reconnect_fail_sec_defaults_to_30_and_can_be_set(monkeypatch):
    monkeypatch.delenv("LIVE_RECONNECT_FAIL_SEC", raising=False)
    assert Config.from_env().live_reconnect_fail_sec == 30
    monkeypatch.setenv("LIVE_RECONNECT_FAIL_SEC", "12")
    assert Config.from_env().live_reconnect_fail_sec == 12


def test_env_example_lists_the_live_reconnect_setting():
    lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    assert "LIVE_RECONNECT_FAIL_SEC=30" in lines


def test_env_example_lists_the_record_settings():
    lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    for line in (
        "RECORD_WIDTH=1280",
        "RECORD_HEIGHT=720",
        "RECORD_FPS=10",
        "RECORD_TMP_DIR=/var/lib/analyzer/tmp",
        "FFMPEG_BIN=ffmpeg",
    ):
        assert line in lines
