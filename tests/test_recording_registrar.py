"""FNC-S-010 녹화 등록 (RecordingRegistrar)"""

import json
import logging
import socket
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from white_eyes_flask.camera.kst import KST
from white_eyes_flask.camera.recording_registrar import (
    RecordingRegistrar,
    RegisterResult,
)

TOKEN = "t0ken"
STARTED_MS = int(datetime(2026, 10, 12, 14, 3, 20, tzinfo=KST).timestamp() * 1000)


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.server.received.append(
            {
                "path": self.path,
                "token": self.headers.get("X-Internal-Token"),
                "content_type": self.headers.get("Content-Type"),
                "body": json.loads(self.rfile.read(length)),
            }
        )
        self.send_response(self.server.status)
        self.send_header("Content-Type", "application/json")
        for name, value in self.server.headers.items():
            self.send_header(name, value)
        self.end_headers()
        if self.server.status == 201:
            self.wfile.write(b'{"id": 900}')

    def do_GET(self):
        # 리다이렉트를 따라오면 여기로 온다
        self.server.received.append({"path": self.path, "method": "GET"})
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    httpd.received = []
    httpd.status = 201
    httpd.headers = {}
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def registrar(server) -> RecordingRegistrar:
    return RecordingRegistrar(f"http://127.0.0.1:{server.server_address[1]}", TOKEN)


def register(reg):
    return reg.register(7, STARTED_MS, "dummy/videos/a.mp4", "dummy/boxes/a.json.gz")


@pytest.mark.parametrize("suffix", ["/admin", "/admin/"])
def test_admin_context_path_in_the_base_url_reaches_the_internal_api(server, suffix):
    base_url = f"http://127.0.0.1:{server.server_address[1]}{suffix}"
    assert register(RecordingRegistrar(base_url, TOKEN)) == RegisterResult.OK
    assert [r["path"] for r in server.received] == ["/admin/internal/v1/recordings"]


def test_posts_the_recording_without_ended_at(server):
    assert register(registrar(server)) == RegisterResult.OK
    assert server.received == [
        {
            "path": "/internal/v1/recordings",
            "token": TOKEN,
            "content_type": "application/json",
            "body": {
                "cameraId": 7,
                "startedAt": "2026-10-12T14:03:20.000+09:00",
                "videoPath": "dummy/videos/a.mp4",
                "boxPath": "dummy/boxes/a.json.gz",
            },
        }
    ]
    assert "endedAt" not in server.received[0]["body"]


def test_bad_request_is_rejected_not_retried(server, caplog):
    server.status = 400
    assert register(registrar(server)) == RegisterResult.REJECTED
    assert "400" in caplog.text


@pytest.mark.parametrize("status", [401, 403, 404, 408, 413, 422, 429])
def test_every_other_client_error_is_worth_retrying(server, caplog, status):
    """400만 거절한다. 토큰·주소 설정 실수는 고친 뒤 등록할 수 있어야 한다(결과 전송과 같은 규칙)."""
    server.status = status
    assert register(registrar(server)) == RegisterResult.RETRY
    assert str(status) in caplog.text
    assert TOKEN not in caplog.text


@pytest.mark.parametrize("status", [401, 403])
def test_token_failures_tell_to_check_the_token_setting(server, caplog, status):
    server.status = status
    register(registrar(server))
    assert "토큰 설정을 확인하세요" in caplog.text


def test_repeated_failures_are_logged_only_now_and_then_and_recovery_once(
    server, caplog
):
    reg = registrar(server)
    server.status = 503
    with caplog.at_level(logging.INFO):
        for _ in range(61):
            assert register(reg) == RegisterResult.RETRY
        failures = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(failures) == 3  # 1번째, 30번째, 60번째
        server.status = 201
        assert register(reg) == RegisterResult.OK
        assert "복구" in caplog.records[-1].getMessage()


def test_redirect_is_not_followed_and_is_retried(server):
    # 토큰이 다른 주소로 따라가지 않게 한다. 리다이렉트는 예상 밖 응답이라 재시도 대상이다.
    server.status = 302
    server.headers = {"Location": f"http://127.0.0.1:{server.server_address[1]}/elsewhere"}
    assert register(registrar(server)) == RegisterResult.RETRY
    assert len(server.received) == 1


def test_a_broken_response_is_worth_retrying_not_an_unhandled_error():
    # HTTP 응답이 아닌 것을 돌려주는 서버(BadStatusLine)
    broken = socket.socket()
    broken.bind(("127.0.0.1", 0))
    broken.listen(1)

    def answer():
        connection, _ = broken.accept()
        connection.recv(65536)
        connection.sendall(b"not http at all\r\n\r\n")
        connection.close()

    threading.Thread(target=answer, daemon=True).start()
    try:
        port = broken.getsockname()[1]
        reg = RecordingRegistrar(f"http://127.0.0.1:{port}", TOKEN, timeout=2.0)
        result = register(reg)
    finally:
        broken.close()
    assert result == RegisterResult.RETRY


@pytest.mark.parametrize("status", [500, 502, 503])
def test_server_errors_are_worth_retrying(server, status):
    server.status = status
    assert register(registrar(server)) == RegisterResult.RETRY


def test_connection_failure_is_worth_retrying(caplog):
    reg = RecordingRegistrar("http://127.0.0.1:1", TOKEN, timeout=0.5)
    assert register(reg) == RegisterResult.RETRY
    assert "녹화 등록 실패" in caplog.text


def test_token_is_never_logged(caplog):
    register(RecordingRegistrar("http://127.0.0.1:1", TOKEN, timeout=0.5))
    assert TOKEN not in caplog.text
