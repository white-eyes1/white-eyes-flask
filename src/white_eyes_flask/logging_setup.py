import contextvars
import logging
import logging.config
import re
import threading
from datetime import datetime

from flask import g, has_request_context

from white_eyes_flask.camera.kst import KST

LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d %(levelname)-5s [%(threadName)s] "
    "%(request_id)s %(name)s - %(message)s"
)
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
NO_REQUEST_ID = "-"
# LOG_LEVEL을 적용하는 앱 패키지. 그 밖의 라이브러리(ultralytics 등)는 WARNING 이상만 남긴다(code-conventions 7-1·7-5).
APP_LOGGER = "white_eyes_flask"
LIBRARY_LEVEL = "WARNING"
# 자기 로거에 핸들러를 달고 propagate를 끄는 라이브러리. import할 때 그렇게 설정해서 루트 설정만으로는 줄어들지 않는다.
LIBRARY_LOGGERS = ("ultralytics", "torch")
# werkzeug 접속 로그는 Flask 요청 컨텍스트가 닫힌 뒤에 찍혀 g를 못 읽는다. 그 로그 줄에만 쓰려고
# 이 스레드가 마지막으로 처리한 요청의 ID를 기억해 둔다(다른 로거에 옛 ID가 붙지 않게 werkzeug만 읽는다).
_last_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "last_request_id", default=None
)
ACCESS_LOGGER = "werkzeug"
# 캠 작업 스레드 이름: cam-{cameraId}, 녹화 등록 스레드 cam-{cameraId}-recording, 더미 미리보기 스레드 cam-{cameraId}-preview
CAMERA_THREAD = re.compile(r"cam-(\d+)(?:-(?:recording|preview))?")
# 밖에서 받은 X-Request-Id는 이 모양만 믿는다(로그 줄을 속이는 문자를 막는다).
REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


class KstFormatter(logging.Formatter):
    """서버 시간대와 상관없이 로그 시각을 한국 표준시로 쓴다."""

    def converter(self, timestamp):
        return datetime.fromtimestamp(timestamp, KST).timetuple()


class RequestIdFilter(logging.Filter):
    """요청 처리 중이면 요청 ID, 캠 작업 스레드면 cam-{cameraId}를 record에 넣는다."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = self._request_id(record)
        return True

    @staticmethod
    def _request_id(record: logging.LogRecord) -> str:
        try:
            if has_request_context():
                return g.get("request_id", NO_REQUEST_ID)
            if record.name == ACCESS_LOGGER:
                return _last_request_id.get() or NO_REQUEST_ID
            match = CAMERA_THREAD.fullmatch(threading.current_thread().name)
            return f"cam-{match[1]}" if match else NO_REQUEST_ID
        except Exception:
            # 로그를 남기려다 예외가 나서 로그가 끊기면 안 된다.
            return NO_REQUEST_ID


def accept_request_id(value: str | None) -> str | None:
    """받은 X-Request-Id가 안전한 모양이면 그대로, 아니면 None."""
    return value if value and REQUEST_ID.fullmatch(value) else None


def quiet_library_loggers() -> None:
    """라이브러리 로거를 WARNING 이상만 내고 우리 콘솔 핸들러로 보내게 한다.

    ultralytics·torch는 import할 때 자기 로거를 INFO·자체 핸들러·propagate=False로 다시 설정한다.
    그래서 라이브러리를 불러온 뒤에 눌러 줘야 한다(여러 번 불러도 같다).
    """
    for name in LIBRARY_LOGGERS:
        library = logging.getLogger(name)
        library.setLevel(LIBRARY_LEVEL)
        for handler in library.handlers[:]:
            library.removeHandler(handler)
        library.propagate = True


def remember_request_id(request_id: str | None) -> None:
    """이 스레드가 처리 중인 요청의 ID를 기억한다(요청이 끝난 뒤 찍히는 접속 로그용)."""
    _last_request_id.set(request_id)


def configure_logging(level: str = "INFO") -> None:
    """루트 로거에 표준 출력 핸들러 하나를 건다. 여러 번 불러도 핸들러는 하나다.

    level(LOG_LEVEL)은 앱 패키지(white_eyes_flask)에만 적용하고, 그 밖의 라이브러리는 WARNING 이상만 낸다.
    """
    level = level.upper()
    if not isinstance(logging.getLevelName(level), int):
        level = "INFO"  # 알 수 없는 레벨 이름 때문에 서버가 못 뜨면 안 된다.
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {"request_id": {"()": RequestIdFilter}},
            "formatters": {
                "default": {
                    "()": KstFormatter,
                    "format": LOG_FORMAT,
                    "datefmt": LOG_DATE_FORMAT,
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                    "formatter": "default",
                    "filters": ["request_id"],
                }
            },
            # 앱 로거는 루트의 핸들러로 올려 보낸다(propagate). 루트의 레벨은 라이브러리 기준이다.
            "loggers": {APP_LOGGER: {"level": level, "propagate": True}},
            "root": {"level": LIBRARY_LEVEL, "handlers": ["console"]},
        }
    )
    quiet_library_loggers()  # 이미 불러온 라이브러리(ultralytics 등)의 자체 설정을 누른다
