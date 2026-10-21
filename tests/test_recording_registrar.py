"""FNC-S-010 녹화 등록 (RecordingRegistrar)"""

import json
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
        self.end_headers()
        if self.server.status == 201:
            self.wfile.write(b'{"id": 900}')

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    httpd.received = []
    httpd.status = 201
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


@pytest.mark.parametrize("status", [400, 401, 404])
def test_client_errors_are_rejected_not_retried(server, caplog, status):
    server.status = status
    assert register(registrar(server)) == RegisterResult.REJECTED
    assert str(status) in caplog.text


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
