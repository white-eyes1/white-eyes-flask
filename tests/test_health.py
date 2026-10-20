import pytest

from white_eyes_flask import create_app
from white_eyes_flask.config import Config

TOKEN = "test-token"


@pytest.fixture
def client():
    app = create_app(
        Config(internal_token=TOKEN, yolo_weights=None, host="127.0.0.1", port=5000)
    )
    return app.test_client()


def test_health_returns_up(client):
    res = client.get("/analyzer/v1/health", headers={"X-Internal-Token": TOKEN})
    assert res.status_code == 200
    assert res.get_json() == {"status": "UP", "cameras": [], "bufferSize": 0}


def test_health_without_token_is_401(client):
    assert client.get("/analyzer/v1/health").status_code == 401


def test_health_with_wrong_token_is_401(client):
    res = client.get("/analyzer/v1/health", headers={"X-Internal-Token": "wrong"})
    assert res.status_code == 401


def test_empty_configured_token_rejects_everything():
    app = create_app(
        Config(internal_token="", yolo_weights=None, host="127.0.0.1", port=5000)
    )
    res = app.test_client().get("/analyzer/v1/health", headers={"X-Internal-Token": ""})
    assert res.status_code == 401
