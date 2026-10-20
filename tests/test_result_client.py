import json
import logging
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from white_eyes_flask.camera.result_client import ResultClient, SendResult

TOKEN = "t0ken"
ITEMS = [
    {
        "cameraId": 7,
        "at": "2026-10-12T14:03:05+09:00",
        "personCount": 42,
        "in": {"N": 1, "E": 0, "S": 0, "W": 2},
        "out": {"N": 0, "E": 0, "S": 0, "W": 0},
        "dwellSec": [305.2, 41.0],
    }
]


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
        for name, value in self.server.headers.items():
            self.send_header(name, value)
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    httpd.received = []
    httpd.status = 204
    httpd.headers = {}
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def client(server, timeout=3.0) -> ResultClient:
    return ResultClient(f"http://127.0.0.1:{server.server_address[1]}", TOKEN, timeout)


def test_posts_the_batch_as_a_json_array_with_the_token(server):
    assert client(server).send(ITEMS) == SendResult.OK
    assert server.received == [
        {
            "path": "/internal/v1/results",
            "token": TOKEN,
            "content_type": "application/json",
            "body": ITEMS,
        }
    ]


@pytest.mark.parametrize("status", [200, 201, 204])
def test_success_statuses(server, status):
    server.status = status
    assert client(server).send(ITEMS) == SendResult.OK


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_reject_the_batch(server, caplog, status):
    server.status = status
    assert client(server).send(ITEMS) == SendResult.REJECTED
    assert str(status) in caplog.text


@pytest.mark.parametrize("status", [408, 429])
def test_timeout_and_rate_limit_are_worth_retrying(server, status):
    server.status = status
    assert client(server).send(ITEMS) == SendResult.RETRY


@pytest.mark.parametrize("status", [500, 502, 503])
def test_server_errors_are_worth_retrying(server, status):
    server.status = status
    assert client(server).send(ITEMS) == SendResult.RETRY


def test_connection_failure_is_worth_retrying_and_hides_the_token(caplog):
    assert (
        ResultClient("http://127.0.0.1:1", TOKEN, timeout=0.5).send(ITEMS)
        == SendResult.RETRY
    )
    assert TOKEN not in caplog.text


def test_repeated_failures_are_logged_only_now_and_then_and_recovery_once(
    server, caplog
):
    c = client(server)
    server.status = 503
    with caplog.at_level(logging.INFO):
        for _ in range(61):
            assert c.send(ITEMS) == SendResult.RETRY
        failures = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(failures) == 3  # 1번째, 30번째, 60번째
        server.status = 204
        assert c.send(ITEMS) == SendResult.OK
        assert "복구" in caplog.records[-1].getMessage()
        assert c.send(ITEMS) == SendResult.OK
        assert sum("복구" in r.getMessage() for r in caplog.records) == 1  # 한 번만
        server.status = 503
        c.send(ITEMS)
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 4


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
        result = ResultClient(f"http://127.0.0.1:{port}", TOKEN, timeout=2.0).send(
            ITEMS
        )
    finally:
        broken.close()
    assert result == SendResult.RETRY


def test_redirect_is_not_followed_and_is_retried(server):
    # 토큰이 다른 주소로 따라가지 않게 한다. 리다이렉트는 예상 밖 응답이라 재시도 대상이다.
    server.status = 307
    server.headers = {"Location": "http://127.0.0.1:1/elsewhere"}
    assert client(server).send(ITEMS) == SendResult.RETRY
    assert len(server.received) == 1


def test_timeout_is_worth_retrying():
    # 연결은 되지만 응답하지 않는 서버(받아 주지 않아도 접속 대기열에는 들어간다)
    silent = socket.socket()
    silent.bind(("127.0.0.1", 0))
    silent.listen(1)
    try:
        port = silent.getsockname()[1]
        result = ResultClient(f"http://127.0.0.1:{port}", TOKEN, timeout=0.3).send(
            ITEMS
        )
    finally:
        silent.close()
    assert result == SendResult.RETRY
