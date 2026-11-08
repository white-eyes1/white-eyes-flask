"""관리자 앱 내부 API 호출(admin_client) — 결과 전송(FNC-S-006)·상태 보고·녹화 등록(FNC-S-008·010)·재시작 복구(FNC-S-012)가 같이 쓴다."""

import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from white_eyes_flask.camera.admin_client import TOKEN_HEADER, AdminClient


class _Handler(BaseHTTPRequestHandler):
    def _record(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        self.server.requests.append(
            {
                "method": self.command,
                "path": self.path,
                "token": self.headers.get(TOKEN_HEADER),
                "contentType": self.headers.get("Content-Type"),
                "body": self.rfile.read(length) if length else b"",
            }
        )
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header(
                "Location", f"http://127.0.0.1:{self.server.server_port}/elsewhere"
            )
            self.end_headers()
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"[]")

    do_GET = do_POST = do_PATCH = _record

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    httpd.requests = []
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def _client(server, base_suffix: str = "") -> AdminClient:
    return AdminClient(f"http://127.0.0.1:{server.server_port}{base_suffix}", "secret")


def test_sends_token_and_json_body(server):
    with _client(server).open(
        "POST", "/internal/v1/results", timeout=2, body=[{"a": 1}]
    ) as response:
        assert response.status == 200
    request = server.requests[0]
    assert request["method"] == "POST"
    assert request["path"] == "/internal/v1/results"
    assert request["token"] == "secret"
    assert request["contentType"] == "application/json"
    assert json.loads(request["body"]) == [{"a": 1}]


def test_get_without_body(server):
    with _client(server, "/").open(
        "GET", "/internal/v1/cameras/desired", timeout=2
    ) as response:
        assert response.read() == b"[]"
    request = server.requests[0]
    # 기준 주소 끝의 / 는 한 번만 쓴다
    assert request["path"] == "/internal/v1/cameras/desired"
    assert request["contentType"] is None
    assert request["body"] == b""


def test_redirect_is_not_followed(server):
    with pytest.raises(urllib.error.HTTPError) as raised:
        _client(server).open(
            "PATCH", "/redirect", timeout=2, body={"status": "RUNNING"}
        )
    raised.value.close()
    assert raised.value.code == 302
    # 다른 주소로 토큰을 들고 가지 않는다
    assert [r["path"] for r in server.requests] == ["/redirect"]
