"""FNC-S-010 녹화 등록 (POST /internal/v1/recordings)"""

import json
import logging
import urllib.error
import urllib.request
from enum import StrEnum

from white_eyes_flask.camera.kst import format_started_at

logger = logging.getLogger(__name__)

TOKEN_HEADER = "X-Internal-Token"
# 다시 보내면 성공할 수 있는 4xx
_RETRYABLE_CLIENT_ERRORS = {408, 429}


class RegisterResult(StrEnum):
    OK = "OK"
    RETRY = "RETRY"  # 연결 실패·5xx 같은 일시 오류: 다시 시도할 만함
    REJECTED = "REJECTED"  # 400·401·404 등: 같은 요청을 다시 보내도 소용없음


class RecordingRegistrar:
    """분석 서버가 관리자 앱에 녹화를 등록한다(POST /internal/v1/recordings).

    더미 캠은 endedAt이 없다. 관리자 앱이 다음 반복이 등록될 때 이전 반복을 닫는다.
    """

    def __init__(self, base_url: str, token: str, timeout: float = 2.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout

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
            with urllib.request.urlopen(request, timeout=self._timeout):
                return RegisterResult.OK
        except urllib.error.HTTPError as error:
            error.close()  # 응답 본문·연결 자원을 바로 놓는다
            if 400 <= error.code < 500 and error.code not in _RETRYABLE_CLIENT_ERRORS:
                logger.error(
                    "녹화 등록 거절 cameraId=%s status=%s", camera_id, error.code
                )
                return RegisterResult.REJECTED
            logger.warning(
                "녹화 등록 실패 cameraId=%s status=%s", camera_id, error.code
            )
            return RegisterResult.RETRY
        except (OSError, ValueError) as error:  # URLError·타임아웃 포함
            # 토큰은 로그에 남기지 않는다
            logger.warning("녹화 등록 실패 cameraId=%s: %s", camera_id, error)
            return RegisterResult.RETRY
