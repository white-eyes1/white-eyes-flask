"""FNC-S-010 녹화 등록 (POST /internal/v1/recordings)"""

import http.client
import json
import logging
import threading
import urllib.error
import urllib.request
from enum import StrEnum

from white_eyes_flask.camera.kst import format_started_at

logger = logging.getLogger(__name__)

TOKEN_HEADER = "X-Internal-Token"
# 결과 전송(ResultClient)과 같은 규칙: 400만 버리고 나머지는 다시 보낸다
_REJECTED_STATUSES = {400}
_AUTH_FAILURES = {401, 403}
# 같은 실패가 이어질 때 첫 번째와 이 횟수마다 한 번만 로그를 남긴다
_LOG_EVERY = 30


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        # 따라가지 않는다. 토큰이 다른 주소로 가지 않게 한다(3xx는 재시도).
        return None


class RegisterResult(StrEnum):
    OK = "OK"
    RETRY = "RETRY"  # 400 외의 실패(401·404·5xx·연결 실패 등): 고치면 성공할 수 있음
    REJECTED = "REJECTED"  # 400: 같은 요청을 다시 보내도 소용없음


class RecordingRegistrar:
    """분석 서버가 관리자 앱에 녹화를 등록한다(POST /internal/v1/recordings).

    더미 캠은 endedAt이 없다. 관리자 앱이 다음 반복이 등록될 때 이전 반복을 닫는다.
    """

    def __init__(self, base_url: str, token: str, timeout: float = 2.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout
        self._opener = urllib.request.build_opener(_NoRedirect)
        # 한 인스턴스를 여러 캠 스레드가 같이 쓴다
        self._lock = threading.Lock()
        self._failures = 0

    def register(
        self, camera_id: int, started_ms: int, video_path: str, box_path: str
    ) -> RegisterResult:
        body = json.dumps(
            {
                "cameraId": camera_id,
                "startedAt": format_started_at(started_ms),
                "videoPath": video_path,
                "boxPath": box_path,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}/internal/v1/recordings",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", TOKEN_HEADER: self._token},
        )
        try:
            with self._opener.open(request, timeout=self._timeout):
                self._recovered()
                return RegisterResult.OK
        except urllib.error.HTTPError as error:
            error.close()  # 응답 본문·연결 자원을 바로 놓는다
            if error.code in _REJECTED_STATUSES:
                logger.error(
                    "녹화 등록 거절 cameraId=%s status=%s", camera_id, error.code
                )
                return RegisterResult.REJECTED
            hint = (
                " (인증 실패: 토큰 설정을 확인하세요)"
                if error.code in _AUTH_FAILURES
                else ""
            )
            self._warn(
                "녹화 등록 실패 cameraId=%s status=%s%s", camera_id, error.code, hint
            )
            return RegisterResult.RETRY
        except (OSError, ValueError, http.client.HTTPException) as error:
            # URLError·타임아웃, 응답 형식이 깨진 경우(BadStatusLine 등) 포함
            # 토큰은 로그에 남기지 않는다
            self._warn("녹화 등록 실패 cameraId=%s: %s", camera_id, error)
            return RegisterResult.RETRY

    def _warn(self, message: str, *args) -> None:
        """연속 실패는 첫 번째와 _LOG_EVERY번째마다만 남긴다."""
        with self._lock:
            self._failures += 1
            count = self._failures
        if count == 1 or count % _LOG_EVERY == 0:
            logger.warning(message + " (연속 %s번째)", *args, count)

    def _recovered(self) -> None:
        with self._lock:
            count, self._failures = self._failures, 0
        if count:
            logger.info("녹화 등록 복구 (그동안 %s번 실패)", count)
