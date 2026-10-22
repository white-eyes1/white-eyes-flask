"""로그 설정(code-conventions 7장): 형식, 요청 ID, 캠 작업 스레드 ID, LOG_LEVEL"""

import logging
import re
import threading
from pathlib import Path

import pytest

from white_eyes_flask import create_app
from white_eyes_flask.camera.preparer import NullPreparer
from white_eyes_flask.config import Config
from white_eyes_flask.logging_setup import accept_request_id, configure_logging

TOKEN = "test-token"
HEADERS = {"X-Internal-Token": TOKEN}
START_BODY = {"sourceType": "DUMMY", "sourceUri": "dummy/videos/a.mp4"}
# 2026-10-22 14:03:05.123 INFO  [MainThread] req-1 name - message
LINE = re.compile(
    r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3} (?P<level>\S+)\s+"
    r"\[(?P<thread>[^\]]+)\] (?P<request_id>\S+) (?P<name>\S+) - (?P<message>.*)$"
)


@pytest.fixture(autouse=True)
def restore_root_logger():
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:], root.level = handlers, level


def make_app(tmp_path: Path, **overrides):
    video = tmp_path / "dummy" / "videos" / "a.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"")
    config = Config(
        internal_token=TOKEN,
        yolo_weights=None,
        host="127.0.0.1",
        port=5000,
        media_root=str(tmp_path),
        **overrides,
    )
    return create_app(config, preparer=NullPreparer())


def parse(out: str) -> list[dict[str, str]]:
    return [m.groupdict() for m in map(LINE.match, out.splitlines()) if m]


def test_log_line_follows_the_convention_format(capsys):
    configure_logging("INFO")
    logging.getLogger("white_eyes_flask.sample").info("캠 가동 cameraId=%s", 7)
    out = capsys.readouterr().out
    line = parse(out)[0]
    assert line["level"] == "INFO"
    assert line["thread"] == "MainThread"
    assert line["request_id"] == "-"
    assert line["name"] == "white_eyes_flask.sample"
    assert line["message"] == "캠 가동 cameraId=7"


def test_log_time_is_korea_standard_time():
    record = logging.LogRecord("n", logging.INFO, __file__, 1, "m", (), None)
    record.created = 0  # 1970-01-01 00:00:00 UTC
    record.msecs = 0
    record.request_id = "-"
    configure_logging("INFO")
    formatter = logging.getLogger().handlers[0].formatter
    assert formatter.format(record).startswith("1970-01-01 09:00:00.000 ")


def test_output_goes_to_stdout_only(capsys):
    configure_logging("INFO")
    logging.getLogger("x").info("hello")
    captured = capsys.readouterr()
    assert "hello" in captured.out
    assert captured.err == ""


def test_configuring_again_keeps_one_handler():
    configure_logging("INFO")
    configure_logging("INFO")
    assert len(logging.getLogger().handlers) == 1


@pytest.mark.parametrize(
    "value, expected",
    [("DEBUG", logging.DEBUG), ("warning", logging.WARNING), ("nope", logging.INFO)],
)
def test_level_comes_from_the_argument_and_falls_back_to_info(value, expected):
    configure_logging(value)
    assert logging.getLogger().level == expected


def test_log_level_env_is_read_and_defaults_to_info(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    assert Config.from_env().log_level == "INFO"
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    assert Config.from_env().log_level == "DEBUG"


def test_env_example_lists_log_level():
    example = Path(__file__).resolve().parent.parent / ".env.example"
    assert "LOG_LEVEL=INFO" in example.read_text(encoding="utf-8").splitlines()


def test_request_id_from_header_is_logged_and_returned(tmp_path, capsys):
    app = make_app(tmp_path)

    @app.get("/probe")
    def probe():
        logging.getLogger("probe").info("in request")
        return "ok"

    res = app.test_client().get(
        "/probe", headers={**HEADERS, "X-Request-Id": "abc-123"}
    )
    assert res.headers["X-Request-Id"] == "abc-123"
    lines = [l for l in parse(capsys.readouterr().out) if l["name"] == "probe"]
    assert lines[0]["request_id"] == "abc-123"


def test_missing_request_id_is_generated(tmp_path):
    res = make_app(tmp_path).test_client().get("/analyzer/v1/health", headers=HEADERS)
    assert re.fullmatch(r"[0-9a-f]{12}", res.headers["X-Request-Id"])


@pytest.mark.parametrize("bad", ["a b", "x" * 65, "한글", "a;rm"])
def test_unsafe_request_id_is_replaced(tmp_path, bad):
    res = (
        make_app(tmp_path)
        .test_client()
        .get("/analyzer/v1/health", headers={**HEADERS, "X-Request-Id": bad})
    )
    assert re.fullmatch(r"[0-9a-f]{12}", res.headers["X-Request-Id"])


def test_unauthorized_response_still_carries_a_request_id(tmp_path):
    res = (
        make_app(tmp_path)
        .test_client()
        .get("/analyzer/v1/health", headers={"X-Request-Id": "req-401"})
    )
    assert res.status_code == 401
    assert res.headers["X-Request-Id"] == "req-401"


@pytest.mark.parametrize(
    "thread_name, expected",
    [
        ("cam-7", "cam-7"),
        ("cam-12-recording", "cam-12"),
        ("cam-5-preview", "cam-5"),  # 더미 미리보기 스레드(FNC-S-010 5번)
        ("result-sender", "-"),
    ],
)
def test_worker_thread_logs_the_camera_id(capsys, thread_name, expected):
    configure_logging("INFO")
    log = logging.getLogger("worker")
    thread = threading.Thread(target=lambda: log.info("working"), name=thread_name)
    thread.start()
    thread.join()
    line = [l for l in parse(capsys.readouterr().out) if l["name"] == "worker"][0]
    assert line["thread"] == thread_name
    assert line["request_id"] == expected


def test_camera_start_and_stop_print_info_lines(tmp_path, capsys):
    app = make_app(tmp_path)
    client = app.test_client()
    client.post("/analyzer/v1/cameras/7/start", json=START_BODY, headers=HEADERS)
    client.post("/analyzer/v1/cameras/7/stop", headers=HEADERS)
    app.extensions["camera_service"].shutdown()
    out = capsys.readouterr().out
    lines = parse(out)
    messages = [l["message"] for l in lines if l["level"] == "INFO"]
    assert any(m.startswith("캠 가동 cameraId=7") for m in messages)
    assert any(m.startswith("캠 정지 cameraId=7") for m in messages)
    assert TOKEN not in out
