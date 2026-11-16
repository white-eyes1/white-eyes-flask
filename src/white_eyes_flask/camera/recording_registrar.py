"""FNC-S-010 녹화 등록 (POST /internal/v1/recordings)"""

import http.client
import logging
import threading
import urllib.error
from enum import StrEnum

from white_eyes_flask.camera.admin_client import AdminClient
from white_eyes_flask.camera.kst import format_started_at

logger = logging.getLogger(__name__)

# 결과 전송(ResultClient)과 같은 규칙: 400만 버리고 나머지는 다시 보낸다
_REJECTED_STATUSES = {400}
_AUTH_FAILURES = {401, 403}
# 같은 실패가 이어질 때 첫 번째와 이 횟수마다 한 번만 로그를 남긴다
_LOG_EVERY = 30


class RegisterResult(StrEnum):
    """녹화 등록 결과. 호출한 쪽이 이 값으로 다시 보낼지 정한다."""

    OK = "OK"
    RETRY = "RETRY"  # 400 외의 실패(401·404·5xx·연결 실패 등): 고치면 성공할 수 있음
    REJECTED = "REJECTED"  # 400: 같은 요청을 다시 보내도 소용없음


class RecordingRegistrar:
    """분석 서버가 관리자 앱에 녹화를 등록한다(POST /internal/v1/recordings).

    더미 캠은 endedAt이 없다. 관리자 앱이 다음 반복이 등록될 때 이전 반복을 닫는다.
    """

    def __init__(self, base_url: str, token: str, timeout: float = 2.0) -> None:
        """등록기를 만든다.

        :param base_url: 관리자 앱 내부 API의 기준 주소
        :param token: 내부 API 인증 토큰
        :param timeout: 요청 한 번의 제한 시간(초). 재생 루프를 오래 붙잡지 않게 짧다
        """
        self._admin = AdminClient(base_url, token)
        self._timeout = timeout
        # 한 인스턴스를 여러 캠 스레드가 같이 쓴다
        self._lock = threading.Lock()
        # 연속 실패 횟수(로그를 줄이는 데만 쓴다). 성공하면 0으로 돌아간다.
        self._failures = 0

    def register(
        self,
        camera_id: int,
        started_ms: int,
        video_path: str,
        box_path: str,
        ended_ms: int | None = None,
    ) -> RegisterResult:
        """녹화 1건을 관리자 앱에 등록한다.

        :param started_ms: 녹화(또는 더미 반복)의 시작 시각, epoch 밀리초
        :param video_path: 공유 폴더 기준 영상 경로
        :param box_path: 공유 폴더 기준 좌표 파일 경로
        :param ended_ms: 라이브 1분 파일의 끝 시각. 더미는 None(endedAt을 보내지 않는다)
        :return: OK, RETRY(다시 보낼 만함), REJECTED(400, 다시 보내도 소용없음)
        """
        # 요청 본문. 시각은 API 규칙대로 KST 문자열로 바꿔 보낸다.
        fields = {
            "cameraId": camera_id,
            "startedAt": format_started_at(started_ms),
            "videoPath": video_path,
            "boxPath": box_path,
        }
        if ended_ms is not None:  # 라이브 1분 파일. 더미는 endedAt이 없다
            fields["endedAt"] = format_started_at(ended_ms)
        try:
            # 2xx면 예외 없이 지나간다. with 문이 응답을 닫는다.
            with self._admin.open(
                "POST", "/internal/v1/recordings", timeout=self._timeout, body=fields
            ):
                # 성공했으니 연속 실패 기록을 지우고, 그동안 실패가 있었으면 복구 로그를 남긴다.
                self._recovered()
                return RegisterResult.OK
        # 서버가 응답은 했지만 2xx가 아니다.
        except urllib.error.HTTPError as error:
            error.close()  # 응답 본문·연결 자원을 바로 놓는다
            # 400은 같은 요청을 다시 보내도 소용없다. 거절로 돌려 재시도하지 않게 한다.
            if error.code in _REJECTED_STATUSES:
                logger.error(
                    "녹화 등록 거절 cameraId=%s status=%s", camera_id, error.code
                )
                return RegisterResult.REJECTED
            # 401·403이면 토큰 설정 실수일 가능성이 커서 로그에 안내를 붙인다.
            hint = (
                " (인증 실패: 토큰 설정을 확인하세요)"
                if error.code in _AUTH_FAILURES
                else ""
            )
            self._warn(
                "녹화 등록 실패 cameraId=%s status=%s%s", camera_id, error.code, hint
            )
            # 그 밖의 코드(401·404·5xx 등)는 원인을 고치면 성공할 수 있으니 다시 보낼 대상이다.
            return RegisterResult.RETRY
        except (OSError, ValueError, http.client.HTTPException) as error:
            # URLError·타임아웃, 응답 형식이 깨진 경우(BadStatusLine 등) 포함
            # 토큰은 로그에 남기지 않는다
            self._warn("녹화 등록 실패 cameraId=%s: %s", camera_id, error)
            return RegisterResult.RETRY

    def _warn(self, message: str, *args) -> None:
        """연속 실패는 첫 번째와 _LOG_EVERY번째마다만 남긴다."""
        # 카운터는 여러 캠 스레드가 같이 올리므로 잠금 안에서 바꾼다. 로그 출력은 잠금 밖에서 한다.
        with self._lock:
            self._failures += 1
            count = self._failures
        if count == 1 or count % _LOG_EVERY == 0:
            logger.warning(message + " (연속 %s번째)", *args, count)

    def _recovered(self) -> None:
        """성공했을 때 연속 실패 카운터를 0으로 되돌리고, 그동안 실패가 있었으면 복구 로그를 남긴다."""
        with self._lock:
            # 읽기와 0으로 되돌리기를 한 번에 해서, 그 사이 다른 스레드가 올린 실패 횟수가 사라지지 않게 한다.
            count, self._failures = self._failures, 0
        if count:
            logger.info("녹화 등록 복구 (그동안 %s번 실패)", count)
