import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from white_eyes_flask.camera.status_reporter import ReportResult, StatusReporter

TOKEN = "t0ken"


class _Handler(BaseHTTPRequestHandler):
    def do_PATCH(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        self.server.received.append(
            {
                "path": self.path,
                "token": self.headers.get("X-Internal-Token"),
                "content_type": self.headers.get("Content-Type"),
                "body": json.loads(body),
            }
        )
        self.send_response(self.server.status)
        self.end_headers()

    def log_message(self, *args):  # 시험 출력을 조용히
        pass


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    httpd.received = []
    httpd.status = 204
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def base_url(server) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}"


def test_report_sends_patch_with_token_and_json_body(server):
    reporter = StatusReporter(base_url(server), TOKEN)
    reporter.report(7, "PREPARING", progress=45)
    assert server.received == [
        {
            "path": "/internal/v1/cameras/7/status",
            "token": TOKEN,
            "content_type": "application/json",
            "body": {"status": "PREPARING", "progress": 45, "errorMessage": None},
        }
    ]


@pytest.mark.parametrize("suffix", ["/admin", "/admin/"])
def test_admin_context_path_in_the_base_url_reaches_the_internal_api(server, suffix):
    reporter = StatusReporter(base_url(server) + suffix, TOKEN)
    assert reporter.report(7, "RUNNING", progress=100) is True
    assert [r["path"] for r in server.received] == [
        "/admin/internal/v1/cameras/7/status"
    ]


def test_report_error_includes_message(server):
    reporter = StatusReporter(base_url(server), TOKEN)
    reporter.report(7, "ERROR", error_message="영상 파일을 읽을 수 없음")
    body = server.received[0]["body"]
    assert body == {
        "status": "ERROR",
        "progress": None,
        "errorMessage": "영상 파일을 읽을 수 없음",
    }


def test_trailing_slash_in_base_url_is_ignored(server):
    StatusReporter(base_url(server) + "/", TOKEN).report(1, "RUNNING", progress=100)
    assert server.received[0]["path"] == "/internal/v1/cameras/1/status"


def test_report_returns_true_on_204(server):
    assert StatusReporter(base_url(server), TOKEN).report(1, "RUNNING") is True


def test_server_error_is_logged_and_swallowed(server, caplog):
    server.status = 500
    assert StatusReporter(base_url(server), TOKEN).report(1, "RUNNING") is False
    assert "상태 보고 실패" in caplog.text


@pytest.mark.parametrize("status", [401, 404])
def test_client_error_is_logged_and_swallowed(server, caplog, status):
    server.status = status
    assert StatusReporter(base_url(server), TOKEN).report(1, "RUNNING") is False
    assert str(status) in caplog.text


def test_connection_failure_is_logged_and_swallowed(caplog):
    reporter = StatusReporter("http://127.0.0.1:1", TOKEN, timeout=0.5)
    assert reporter.report(1, "RUNNING") is False
    assert "상태 보고 실패" in caplog.text


def test_token_value_is_never_logged(caplog):
    StatusReporter("http://127.0.0.1:1", TOKEN, timeout=0.5).report(1, "RUNNING")
    assert TOKEN not in caplog.text


@pytest.mark.parametrize(
    "status, expected",
    [
        (204, ReportResult.OK),
        (400, ReportResult.REJECTED),
        (404, ReportResult.REJECTED),
        (401, ReportResult.RETRY),
        (403, ReportResult.RETRY),
        (500, ReportResult.RETRY),
        (503, ReportResult.RETRY),
    ],
)
def test_send_tells_retryable_failures_from_hopeless_ones(server, status, expected):
    server.status = status
    result, _ = StatusReporter(base_url(server), TOKEN).send(1, "RUNNING")
    assert result == expected


def test_send_does_not_log_and_never_includes_the_token(server, caplog):
    server.status = 401
    result, detail = StatusReporter(base_url(server), TOKEN).send(1, "RUNNING")
    assert result == ReportResult.RETRY
    assert "토큰 설정" in detail and TOKEN not in detail
    assert caplog.text == ""


def test_send_treats_connection_failure_as_retryable():
    reporter = StatusReporter("http://127.0.0.1:1", TOKEN, timeout=0.5)
    assert reporter.send(1, "RUNNING")[0] == ReportResult.RETRY
