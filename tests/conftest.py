import pytest

from white_eyes_flask.camera.result_sender import ResultSender


@pytest.fixture(autouse=True)
def stop_started_result_senders(monkeypatch):
    """앱을 실제 재생기로 만들면 결과 전송 스레드가 시작된다. 시험이 끝나면 모두 멈춘다."""
    started = []
    original = ResultSender.start

    def start(self):
        started.append(self)
        original(self)

    monkeypatch.setattr(ResultSender, "start", start)
    yield
    for sender in started:
        sender.stop(timeout=1.0)
