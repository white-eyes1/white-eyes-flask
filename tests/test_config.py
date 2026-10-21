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
