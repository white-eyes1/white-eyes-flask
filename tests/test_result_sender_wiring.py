"""FNC-S-006 전송기를 앱에 연결 (create_app)"""

import pytest

from white_eyes_flask import app as app_module
from white_eyes_flask.app import create_app
from white_eyes_flask.config import Config


class FakePreparer:
    def prepare(self, *args, **kwargs):  # pragma: no cover - 쓰이지 않는다
        raise AssertionError


class FakePlayer:
    def play(self, *args, **kwargs):  # pragma: no cover - 쓰이지 않는다
        raise AssertionError


class FakeSender:
    def __init__(self):
        self.started = 0
        self.stopped = 0

    def start(self):
        self.started += 1

    def stop(self, timeout=5.0):
        self.stopped += 1


def config(**kwargs):
    return Config(
        internal_token="t0ken",
        yolo_weights=None,
        host="127.0.0.1",
        port=5000,
        **kwargs,
    )


def test_config_reads_the_retry_buffer_seconds_from_env(monkeypatch):
    monkeypatch.setenv("RESULT_RETRY_BUFFER_SEC", "120")
    assert Config.from_env().result_retry_buffer_sec == 120
    monkeypatch.delenv("RESULT_RETRY_BUFFER_SEC")
    assert Config.from_env().result_retry_buffer_sec == 300


def test_an_injected_sender_is_started_and_kept_on_the_app():
    sender = FakeSender()
    app = create_app(
        config(), preparer=FakePreparer(), player=FakePlayer(), result_sender=sender
    )
    assert sender.started == 1
    assert app.extensions["result_sender"] is sender


def test_no_sender_is_started_when_a_player_is_injected():
    # 가짜 재생기는 전송 큐를 쓰지 않으므로 보낼 것이 없다(시험에서 쓸데없는 스레드가 돌지 않게)
    app = create_app(config(), preparer=FakePreparer(), player=FakePlayer())
    assert "result_sender" not in app.extensions


def test_main_stops_the_sender_when_the_server_ends(monkeypatch):
    import white_eyes_flask

    sender = FakeSender()
    events = []

    class FakeRecovery:
        def start(self):
            events.append("recovery start")

        def stop(self):
            events.append("recovery stop")

    class FakeService:
        def shutdown(self):
            events.append("service shutdown")

    class FakeApp:
        extensions = {"result_sender": sender, "camera_service": FakeService()}

        def run(self, host, port, use_reloader=True):
            events.append(("run", host, port, use_reloader))
            raise KeyboardInterrupt  # 서버가 Ctrl+C로 끝난다

    monkeypatch.setattr(white_eyes_flask, "load_dotenv", lambda: None)
    monkeypatch.setattr(white_eyes_flask, "create_app", lambda config: FakeApp())
    monkeypatch.setattr(
        white_eyes_flask,
        "build_restart_recovery",
        lambda config, service: FakeRecovery(),
    )
    served = Config(internal_token="t", yolo_weights=None, host="127.0.0.1", port=5001)
    monkeypatch.setattr(white_eyes_flask.Config, "from_env", lambda: served)
    with pytest.raises(KeyboardInterrupt):
        white_eyes_flask.main()
    # 서버를 켜기 전에 복구를 시작하고(FNC-S-012), 끝나면 멈춘다
    assert events == [
        "recovery start",
        ("run", "127.0.0.1", 5001, False),
        "recovery stop",
        "service shutdown",
    ]
    assert sender.stopped == 1


def test_the_real_run_builds_a_sender_from_the_config(monkeypatch):
    built = {}

    class RecordingSender(FakeSender):
        def __init__(self, queue, client, retry_buffer_sec):
            super().__init__()
            built.update(queue=queue, client=client, retry=retry_buffer_sec)

    class RecordingClient:
        def __init__(self, base_url, token):
            built.update(base_url=base_url, token=token)

    monkeypatch.setattr(app_module, "ResultSender", RecordingSender)
    monkeypatch.setattr(app_module, "ResultClient", RecordingClient)
    app = create_app(
        config(
            admin_internal_base_url="http://admin:8082/admin",
            result_retry_buffer_sec=240,
        ),
        preparer=FakePreparer(),
    )
    sender = app.extensions["result_sender"]
    assert sender.started == 1
    assert built["base_url"] == "http://admin:8082/admin"
    assert built["token"] == "t0ken"
    assert built["retry"] == 240
    # 재생기가 넣는 큐를 보내는 쪽이 비운다(같은 큐)
    assert built["queue"] is app.extensions["result_queue"]
