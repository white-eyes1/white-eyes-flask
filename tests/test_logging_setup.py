"""로그 설정(code-conventions 7장): 형식, 요청 ID, 캠 작업 스레드 ID, LOG_LEVEL"""

import logging
import re
import threading
from pathlib import Path

import pytest

from white_eyes_flask import create_app
from white_eyes_flask.camera.preparer import NullPreparer
from white_eyes_flask.config import Config
from white_eyes_flask.logging_setup import (
    APP_LOGGER,
    LIBRARY_LOGGERS,
    accept_request_id,
    configure_logging,
    quiet_library_loggers,
    remember_request_id,
)

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
    app_logger = logging.getLogger(APP_LOGGER)
    handlers, level = root.handlers[:], root.level
    app_level = app_logger.level
    libraries = {n: logging.getLogger(n) for n in LIBRARY_LOGGERS}
    saved = {n: (l.level, l.handlers[:], l.propagate) for n, l in libraries.items()}
    remember_request_id(None)
    yield
    root.handlers[:], root.level = handlers, level
    app_logger.setLevel(app_level)
    for name, library in libraries.items():
        library.setLevel(saved[name][0])
        library.handlers[:] = saved[name][1]
        library.propagate = saved[name][2]
    remember_request_id(None)


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
    logging.getLogger("white_eyes_flask.x").info("hello")
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
    # LOG_LEVEL은 앱 패키지에만 적용하고, 루트(그 밖의 라이브러리)는 WARNING이다
    assert logging.getLogger(APP_LOGGER).level == expected
    assert logging.getLogger().level == logging.WARNING


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
        logging.getLogger("white_eyes_flask.probe").info("in request")
        return "ok"

    res = app.test_client().get(
        "/probe", headers={**HEADERS, "X-Request-Id": "abc-123"}
    )
    assert res.headers["X-Request-Id"] == "abc-123"
    lines = [
        l
        for l in parse(capsys.readouterr().out)
        if l["name"] == "white_eyes_flask.probe"
    ]
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
    log = logging.getLogger("white_eyes_flask.worker")
    thread = threading.Thread(target=lambda: log.info("working"), name=thread_name)
    thread.start()
    thread.join()
    line = [
        l
        for l in parse(capsys.readouterr().out)
        if l["name"] == "white_eyes_flask.worker"
    ][0]
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


# ---- 라이브러리 로그 레벨 분리(code-conventions 7-1·7-5) ----


def simulate_library_import(name: str) -> None:
    """라이브러리가 import하며 하는 설정을 흉내 낸다: INFO, 자체 핸들러, propagate 끔."""
    library = logging.getLogger(name)
    library.setLevel(logging.INFO)
    library.addHandler(logging.StreamHandler())
    library.propagate = False


@pytest.mark.parametrize("level", ["INFO", "DEBUG"])
def test_real_ultralytics_info_and_debug_logs_are_not_shown(capsys, level):
    import ultralytics  # noqa: F401  (import하면 자기 로거를 INFO·자체 핸들러로 설정한다)

    configure_logging(level)
    library = logging.getLogger("ultralytics")
    assert library.level == logging.WARNING
    assert library.handlers == [] and library.propagate is True
    library.info("lib info")
    library.debug("lib debug")
    assert capsys.readouterr().out == ""


def test_library_warnings_and_errors_are_still_shown_in_our_format(capsys):
    configure_logging("INFO")
    logging.getLogger("ultralytics").warning("lib warning")
    logging.getLogger("torch").error("lib error")
    lines = parse(capsys.readouterr().out)
    assert [(l["level"], l["name"], l["message"]) for l in lines] == [
        ("WARNING", "ultralytics", "lib warning"),
        ("ERROR", "torch", "lib error"),
    ]


@pytest.mark.parametrize("name", LIBRARY_LOGGERS)
def test_a_library_that_resets_its_logger_after_startup_is_quieted_again(capsys, name):
    configure_logging("INFO")
    simulate_library_import(name)  # 앱이 켜진 뒤 라이브러리가 자기 설정을 다시 한다
    quiet_library_loggers()
    library = logging.getLogger(name)
    assert library.level == logging.WARNING
    assert library.handlers == [] and library.propagate is True
    library.info("noisy")
    assert capsys.readouterr().out == ""


def test_the_detector_provider_quiets_libraries_after_loading_the_model(tmp_path):
    from white_eyes_flask.analysis.detector_provider import LazyDetectorProvider

    weights = tmp_path / "w.pt"
    weights.write_bytes(b"x")
    provider = LazyDetectorProvider(str(weights), 640, 0.3)
    provider._load = lambda path: (simulate_library_import("ultralytics"), object())[1]
    provider()  # 모델을 불러오며 라이브러리가 로거를 다시 설정한다
    library = logging.getLogger("ultralytics")
    assert library.level == logging.WARNING
    assert library.handlers == [] and library.propagate is True


def test_a_logging_setup_failure_is_not_reported_as_a_model_load_failure(
    tmp_path, monkeypatch
):
    from white_eyes_flask.analysis import detector_provider
    from white_eyes_flask.analysis.detector_provider import (
        DetectorUnavailableError,
        LazyDetectorProvider,
    )

    def broken():
        raise RuntimeError("logging broke")

    monkeypatch.setattr(detector_provider, "quiet_library_loggers", broken)
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"x")
    provider = LazyDetectorProvider(str(weights), 640, 0.3)
    model = object()
    provider._load = lambda path: model
    with pytest.raises(RuntimeError, match="logging broke"):
        provider()
    assert not issubclass(RuntimeError, DetectorUnavailableError)
    monkeypatch.setattr(detector_provider, "quiet_library_loggers", lambda: None)
    assert provider() is model  # 모델은 이미 불러왔으니 다시 불러오지 않는다


def test_app_package_follows_log_level_including_sub_loggers(capsys):
    configure_logging("DEBUG")
    logging.getLogger("white_eyes_flask.camera.camera_service").debug("app debug")
    assert "app debug" in capsys.readouterr().out
    configure_logging("INFO")
    logging.getLogger("white_eyes_flask.camera.camera_service").debug("hidden")
    logging.getLogger("white_eyes_flask.camera.camera_service").info("shown")
    out = capsys.readouterr().out
    assert "hidden" not in out and "shown" in out


def test_app_logs_still_use_the_single_console_handler(capsys):
    configure_logging("INFO")
    configure_logging("INFO")
    logging.getLogger("white_eyes_flask.x").info("once")
    assert (
        capsys.readouterr().out.count("once") == 1
    )  # 핸들러가 겹쳐 두 번 나오지 않는다


# ---- werkzeug 접속 로그의 요청 ID ----


def test_the_access_log_line_after_a_request_carries_its_request_id(tmp_path, capsys):
    app = make_app(tmp_path)
    app.test_client().get(
        "/analyzer/v1/health", headers={**HEADERS, "X-Request-Id": "req-777"}
    )
    capsys.readouterr()
    # 접속 로그는 Flask 요청 컨텍스트가 닫힌 뒤에 찍힌다
    logging.getLogger("werkzeug").warning("127.0.0.1 - GET /analyzer/v1/health 200")
    line = [l for l in parse(capsys.readouterr().out) if l["name"] == "werkzeug"][0]
    assert line["request_id"] == "req-777"


def test_the_next_request_replaces_the_remembered_id(tmp_path, capsys):
    app = make_app(tmp_path)
    client = app.test_client()
    client.get("/analyzer/v1/health", headers={**HEADERS, "X-Request-Id": "first"})
    client.get("/analyzer/v1/health", headers={**HEADERS, "X-Request-Id": "second"})
    capsys.readouterr()
    logging.getLogger("werkzeug").warning("access")
    line = [l for l in parse(capsys.readouterr().out) if l["name"] == "werkzeug"][0]
    assert line["request_id"] == "second"


def test_other_loggers_do_not_pick_up_the_remembered_id(tmp_path, capsys):
    app = make_app(tmp_path)
    app.test_client().get(
        "/analyzer/v1/health", headers={**HEADERS, "X-Request-Id": "req-777"}
    )
    capsys.readouterr()
    logging.getLogger("white_eyes_flask.later").info("after the request")
    line = [
        l
        for l in parse(capsys.readouterr().out)
        if l["name"] == "white_eyes_flask.later"
    ][0]
    assert line["request_id"] == "-"  # 요청이 끝난 뒤의 다른 로그에 옛 ID가 붙지 않는다


def test_the_access_log_before_any_request_has_no_id(capsys):
    configure_logging("INFO")
    logging.getLogger("werkzeug").warning("boot")
    line = [l for l in parse(capsys.readouterr().out) if l["name"] == "werkzeug"][0]
    assert line["request_id"] == "-"


def test_the_remembered_id_is_per_thread(tmp_path, capsys):
    app = make_app(tmp_path)
    app.test_client().get(
        "/analyzer/v1/health", headers={**HEADERS, "X-Request-Id": "main-thread"}
    )
    capsys.readouterr()
    seen = []

    def other_thread():
        logging.getLogger("werkzeug").warning("from another thread")

    worker = threading.Thread(target=other_thread)
    worker.start()
    worker.join()
    line = [l for l in parse(capsys.readouterr().out) if l["name"] == "werkzeug"][0]
    assert line["request_id"] == "-"  # 다른 스레드가 처리한 요청의 ID를 쓰지 않는다
