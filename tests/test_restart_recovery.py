"""FNC-S-012 분석 서버 재시작 복구"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from white_eyes_flask.camera.camera_service import CameraService, CameraStoppingError
from white_eyes_flask.camera.restart_recovery import DesiredClient, RestartRecovery
from white_eyes_flask.camera.status_reporter import ReportResult

TOKEN = "test-token"


class FakeReporter:
    """send를 부를 때마다 results에서 결과를 하나씩 꺼낸다(다 쓰면 OK)."""

    def __init__(self, *results) -> None:
        self.calls: list[tuple] = []
        self.results = list(results)

    def send(self, camera_id, status, progress=None, error_message=None):
        self.calls.append((camera_id, status, error_message))
        result = self.results.pop(0) if self.results else ReportResult.OK
        if isinstance(result, Exception):
            raise result
        return result, "" if result == ReportResult.OK else "HTTP Error 500"


class FakeClient:
    """fetch를 부를 때마다 준비된 결과를 하나씩 돌려준다(None이면 연결 실패)."""

    def __init__(self, *results) -> None:
        self.results = list(results)
        self.calls = 0

    def fetch(self):
        self.calls += 1
        return self.results.pop(0) if self.results else None


@pytest.fixture
def service(tmp_path: Path):
    for name in ("a.mp4", "b.mp4"):
        video = tmp_path / "dummy" / "videos" / name
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"")
    service = CameraService(str(tmp_path))
    yield service
    service.shutdown()


def dummy(camera_id: int, name: str = "a.mp4") -> dict:
    return {
        "cameraId": camera_id,
        "sourceType": "DUMMY",
        "sourceUri": f"dummy/videos/{name}",
    }


def make(client, service, reporter=None, **kwargs):
    return RestartRecovery(client, service, reporter or FakeReporter(), **kwargs)


def wait_until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


# ---------- 복구 ----------


def test_starts_each_desired_camera_and_reports_status(service):
    reporter = FakeReporter()
    make(FakeClient(), service, reporter).recover([dummy(1), dummy(2, "b.mp4")])
    assert service.job_count() == 2
    # 캐시가 없으니 PREPARING으로 시작한다(FNC-S-001과 같은 방식)
    assert sorted(reporter.calls) == [(1, "PREPARING", None), (2, "PREPARING", None)]


def test_empty_list_starts_nothing(service):
    make(FakeClient(), service).recover([])
    assert service.job_count() == 0


def test_camera_already_running_is_not_started_twice(service):
    service.start(1, "DUMMY", "dummy/videos/a.mp4")
    first_job = service._get_job(1)
    make(FakeClient(), service).recover([dummy(1)])
    assert service.job_count() == 1
    assert service._get_job(1) is first_job


def test_live_camera_is_restarted_and_next_camera_still_starts(service):
    live = {"cameraId": 5, "sourceType": "LIVE", "sourceUri": "rtsp://x/y"}
    reporter = FakeReporter()
    make(FakeClient(), service, reporter).recover([live, dummy(1)])
    assert service.job_count() == 2
    assert (5, "RUNNING", None) in reporter.calls
    assert (1, "PREPARING", None) in reporter.calls


def test_live_camera_with_a_bad_address_reports_error_and_continues(service):
    bad = {"cameraId": 5, "sourceType": "LIVE", "sourceUri": "file:///etc/passwd"}
    reporter = FakeReporter()
    make(FakeClient(), service, reporter).recover([bad, dummy(1)])
    assert service.job_count() == 1
    assert (5, "ERROR", "영상 파일을 찾을 수 없음") in reporter.calls


def test_missing_video_reports_error_and_continues(service):
    reporter = FakeReporter()
    missing = dummy(3, "nope.mp4")
    make(FakeClient(), service, reporter).recover([missing, dummy(1)])
    assert service.job_count() == 1
    assert (3, "ERROR", "영상 파일을 찾을 수 없음") in reporter.calls
    assert (1, "PREPARING", None) in reporter.calls


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "x",
        {},
        {"cameraId": "1", "sourceType": "DUMMY", "sourceUri": "dummy/videos/a.mp4"},
        {"cameraId": True, "sourceType": "DUMMY", "sourceUri": "dummy/videos/a.mp4"},
        {"cameraId": 1, "sourceType": "DUMMY"},
    ],
)
def test_malformed_item_is_skipped_and_next_still_starts(service, bad):
    make(FakeClient(), service).recover([bad, dummy(2)])
    assert service.job_count() == 1


def test_exception_from_one_camera_does_not_stop_others():
    class Boom:
        def __init__(self) -> None:
            self.started: list[int] = []

        def start(self, camera_id, source_type, source_uri):
            if camera_id == 1:
                raise RuntimeError("boom")
            self.started.append(camera_id)
            return "RUNNING"

        def current_status(self, camera_id):
            return "RUNNING"

    boom = Boom()
    make(FakeClient(), boom).recover([dummy(1), dummy(2)])
    assert boom.started == [2]


# ---------- 재시도 ----------


def test_retries_until_admin_app_answers(service):
    client = FakeClient(None, None, [dummy(1)])
    recovery = make(client, service, retry_interval=0.01)
    recovery.run()
    assert client.calls == 3
    assert service.job_count() == 1


def test_background_thread_does_not_block_and_stops_on_request(service):
    client = FakeClient()  # 계속 실패
    recovery = make(client, service, retry_interval=0.01)
    recovery.start()
    assert wait_until(lambda: client.calls >= 3)
    recovery.stop()
    assert not recovery._thread.is_alive()
    calls = client.calls
    time.sleep(0.05)
    assert client.calls == calls


def test_does_not_fetch_again_after_success(service):
    client = FakeClient([dummy(1)], [dummy(2, "b.mp4")])
    make(client, service, retry_interval=0.01).run()
    assert client.calls == 1


# ---------- HTTP 클라이언트 ----------


class AdminStub:
    """관리자 앱 흉내. responses 순서대로 (status, body)를 돌려주고 받은 헤더를 기록한다."""

    def __init__(self, *responses) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[str, str | None]] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                stub.requests.append((self.path, self.headers.get("X-Internal-Token")))
                status, body = stub.responses.pop(0) if stub.responses else (500, b"")
                self.send_response(status)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/admin"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stub():
    created = []

    def make_stub(*responses):
        created.append(AdminStub(*responses))
        return created[-1]

    yield make_stub
    for s in created:
        s.close()


def test_client_gets_list_with_token_at_spec_path(stub):
    body = json.dumps([dummy(1)]).encode()
    admin = stub((200, body))
    assert DesiredClient(admin.url, TOKEN).fetch() == [dummy(1)]
    assert admin.requests == [("/admin/internal/v1/cameras/desired", TOKEN)]


@pytest.mark.parametrize(
    "response",
    [(401, b""), (500, b""), (200, b"not json"), (200, b'{"a": 1}'), (302, b"")],
)
def test_client_returns_none_on_bad_response(stub, response):
    admin = stub(response)
    assert DesiredClient(admin.url, TOKEN).fetch() is None


def test_client_returns_none_when_admin_app_is_down():
    assert DesiredClient("http://127.0.0.1:1/admin", TOKEN, timeout=0.5).fetch() is None


def test_client_log_never_contains_token(stub, caplog):
    admin = stub((401, b""))
    with caplog.at_level("WARNING"):
        DesiredClient(admin.url, TOKEN).fetch()
    assert TOKEN not in caplog.text
    assert "토큰 설정" in caplog.text


class StubService:
    """CameraService 흉내. start가 돌려줄 값과 지금 상태(current)를 시험이 바꾼다."""

    def __init__(self, start_results=("PREPARING",)) -> None:
        self.start_results = list(start_results)
        self.calls = 0
        self.current = "PREPARING"

    def start(self, camera_id, source_type, source_uri):
        self.calls += 1
        result = self.start_results.pop(0) if self.start_results else "PREPARING"
        if isinstance(result, Exception):
            raise result
        return result

    def current_status(self, camera_id):
        return self.current


def test_camera_still_stopping_is_retried_until_it_can_start():
    stopping = CameraStoppingError(1)
    service = StubService([stopping, stopping, "PREPARING"])
    reporter = FakeReporter()
    make(FakeClient([dummy(1)]), service, reporter, retry_interval=0.01).run()
    assert service.calls == 3
    assert reporter.calls == [(1, "PREPARING", None)]


def test_failed_status_report_is_sent_again_without_restarting_the_camera():
    service = StubService()
    reporter = FakeReporter(ReportResult.RETRY, ReportResult.RETRY, ReportResult.OK)
    make(FakeClient([dummy(1)]), service, reporter, retry_interval=0.01).run()
    assert reporter.calls == [(1, "PREPARING", None)] * 3
    assert service.calls == 1  # 시작은 한 번뿐


def test_report_exception_is_retried_too():
    reporter = FakeReporter(RuntimeError("boom"), ReportResult.OK)
    make(FakeClient([dummy(1)]), StubService(), reporter, retry_interval=0.01).run()
    assert len(reporter.calls) == 2


def test_rejected_report_is_dropped_once_and_never_repeated(caplog):
    # 관리자가 그 사이 캠을 삭제하면 404로 거절된다. 다시 보내도 소용없다.
    reporter = FakeReporter(ReportResult.REJECTED)
    recovery = make(
        FakeClient([dummy(1)]), StubService(), reporter, retry_interval=0.01
    )
    with caplog.at_level("ERROR"):
        recovery.run()
    assert len(reporter.calls) == 1
    assert "거절돼 버림" in caplog.text
    assert "ADMIN_INTERNAL_BASE_URL" in caplog.text


def test_rejected_error_report_is_dropped_too(service):
    reporter = FakeReporter(ReportResult.REJECTED)
    missing = dummy(3, "nope.mp4")
    make(FakeClient([missing]), service, reporter, retry_interval=0.01).run()
    assert reporter.calls == [(3, "ERROR", "영상 파일을 찾을 수 없음")]


def test_retry_sends_the_current_status_not_the_stored_one():
    service = StubService()
    reporter = FakeReporter(ReportResult.RETRY, ReportResult.OK)

    class Flip(FakeReporter):
        def send(self, *args, **kwargs):
            out = super().send(*args, **kwargs)
            service.current = "RUNNING"  # 첫 보고 실패 뒤 준비가 끝났다
            return out

    flip = Flip(ReportResult.RETRY, ReportResult.OK)
    make(FakeClient([dummy(1)]), service, flip, retry_interval=0.01).run()
    assert flip.calls == [(1, "PREPARING", None), (1, "RUNNING", None)]
    assert reporter.calls == []


def test_retry_is_dropped_if_the_camera_was_stopped_meanwhile():
    service = StubService()
    reporter = FakeReporter(ReportResult.RETRY, ReportResult.OK)

    class StopAfterFirst(FakeReporter):
        def send(self, *args, **kwargs):
            out = super().send(*args, **kwargs)
            service.current = None  # 관리자가 정지해 캠 작업이 사라졌다
            return out

    stopper = StopAfterFirst(ReportResult.RETRY, ReportResult.OK)
    make(FakeClient([dummy(1)]), service, stopper, retry_interval=0.01).run()
    assert stopper.calls == [
        (1, "PREPARING", None)
    ]  # 정지된 캠의 PREPARING을 다시 보내지 않는다
    assert reporter.calls == []


def test_repeated_report_failures_are_logged_only_now_and_then(caplog):
    results = [ReportResult.RETRY] * 40 + [ReportResult.OK]
    reporter = FakeReporter(*results)
    with caplog.at_level("WARNING"):
        make(FakeClient([dummy(1)]), StubService(), reporter, retry_interval=0).run()
    assert len(reporter.calls) == 41
    warnings = [r for r in caplog.records if "상태 보고 실패" in r.getMessage()]
    assert len(warnings) == 2  # 처음과 30번째


def test_real_camera_service_current_status(service):
    assert service.current_status(1) is None
    service.start(1, "DUMMY", "dummy/videos/a.mp4")
    assert str(service.current_status(1)) in ("PREPARING", "RUNNING")
    service.stop(1)
    assert service.current_status(1) is None
