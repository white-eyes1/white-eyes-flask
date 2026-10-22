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
# 캠 작업 스레드 이름: cam-{cameraId}, 녹화 등록 스레드 cam-{cameraId}-recording
CAMERA_THREAD = re.compile(r"cam-(\d+)(?:-recording)?")
# 밖에서 받은 X-Request-Id는 이 모양만 믿는다(로그 줄을 속이는 문자를 막는다).
REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


class KstFormatter(logging.Formatter):
    """서버 시간대와 상관없이 로그 시각을 한국 표준시로 쓴다."""

    def converter(self, timestamp):
        return datetime.fromtimestamp(timestamp, KST).timetuple()


class RequestIdFilter(logging.Filter):
    """요청 처리 중이면 요청 ID, 캠 작업 스레드면 cam-{cameraId}를 record에 넣는다."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = self._request_id()
        return True

    @staticmethod
    def _request_id() -> str:
        try:
            if has_request_context():
                return g.get("request_id", NO_REQUEST_ID)
            match = CAMERA_THREAD.fullmatch(threading.current_thread().name)
            return f"cam-{match[1]}" if match else NO_REQUEST_ID
        except Exception:
            # 로그를 남기려다 예외가 나서 로그가 끊기면 안 된다.
            return NO_REQUEST_ID


def accept_request_id(value: str | None) -> str | None:
    """받은 X-Request-Id가 안전한 모양이면 그대로, 아니면 None."""
    return value if value and REQUEST_ID.fullmatch(value) else None


def configure_logging(level: str = "INFO") -> None:
    """루트 로거에 표준 출력 핸들러 하나를 건다. 여러 번 불러도 핸들러는 하나다."""
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
            "root": {"level": level, "handlers": ["console"]},
        }
    )
