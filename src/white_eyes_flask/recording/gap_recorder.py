"""FNC-S-008 스트림이 끊겼다 복구되면 녹화 파일을 새로 시작하는 녹화기

LiveRecorder는 ffmpeg 하나로 이어서 녹화하므로, 프레임이 오래 안 오면 ffmpeg가 그 빈 시간을 마지막
화면으로 채우고 분 경계에서도 자르지 못한다(실제 RTSP로 74초 끊김 뒤 120초짜리 파일이 나왔다).
이 래퍼는 프레임이 GAP_SEC 이상 끊겼다가 다시 오면 그 LiveRecorder를 마무리(마지막 프레임까지의
파일을 게시기에 넘김)하고 새 LiveRecorder로 복구한 프레임부터 새 파일을 시작한다. 끊긴 구간에는 녹화가 없다.

호출 규칙은 LiveRecorder와 같다: write_frame·close는 캠 작업 스레드 하나, add_boxes는 분석 스레드에서도 된다.
"""

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from white_eyes_flask.analysis.result import TrackedBox
from white_eyes_flask.recording.recorder import LiveRecorder

logger = logging.getLogger(__name__)

# 재연결 간격(2초)과 연결 시도 시간보다 약간 큰 값. 이보다 짧은 끊김은 파일이 이어진다(멈춘 화면으로 채워짐)
GAP_SEC = 5.0
# 닫는 중인 이전 녹화기를 close()에서 기다리는 전체 시간(초). 캠 정지 대기(10초) 안에서 현재 녹화기를 닫고 남는 몫이다
CLOSE_JOIN_SEC = 2.0
# 이전 녹화기가 이만큼 넘게 아직 닫히는 중이면 경고한다(끊김이 아주 잦은 스트림)
_CLOSING_WARN = 3


class GapSplittingRecorder:
    def __init__(
        self,
        camera_id: int,
        make_recorder: Callable[[], LiveRecorder],
        gap_sec: float = GAP_SEC,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.camera_id = camera_id
        self._make = make_recorder
        self._gap_sec = gap_sec
        self._clock = (
            clock  # 끊김 판정에는 시스템 시계 보정에 흔들리지 않는 시계를 쓴다
        )
        self._lock = threading.Lock()
        self._current: LiveRecorder | None = None
        self._last_frame_at: float | None = None
        self._closers: list[threading.Thread] = []
        self._closed = False

    # ---- 캠 작업 스레드가 부른다 ----

    def write_frame(self, frame: Any, at: float) -> None:
        """LiveRecorder.write_frame과 같다. 끊겼다 복구된 첫 프레임이면 새 파일을 시작한다."""
        now = self._clock()
        with self._lock:
            if self._closed:
                return
            current = self._current
        if (
            current is not None
            and self._last_frame_at is not None
            and now - self._last_frame_at >= self._gap_sec
        ):
            logger.info(
                "스트림이 %.1f초 끊겼다 복구되어 녹화 파일을 새로 시작함 cameraId=%s",
                now - self._last_frame_at,
                self.camera_id,
            )
            self._retire(current)
            current = None
        if current is None:
            current = self._make()
            with self._lock:
                self._current = current
        self._last_frame_at = now
        current.write_frame(frame, at)

    def add_boxes(self, at: float, boxes: list[TrackedBox]) -> None:
        """분석한 프레임의 박스. 끊김 직전 프레임의 박스가 복구 뒤에 도착하면 새 파일의 시작보다 이르다고 버려진다."""
        with self._lock:
            current = self._current
        if current is not None:
            current.add_boxes(at, boxes)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            current, self._current = self._current, None
        try:
            if current is not None:
                current.close()
        finally:
            self._join_closers()

    # ---- 이전 녹화기 마무리 ----

    def _retire(self, recorder: LiveRecorder) -> None:
        """이전 녹화기를 백그라운드에서 닫는다. 복구한 프레임을 읽는 스레드를 오래 막지 않기 위해서다."""
        with self._lock:
            self._current = None
        thread = threading.Thread(
            target=self._close_quietly,
            args=(recorder,),
            name=f"cam-{self.camera_id}-recording",
            daemon=True,
        )
        thread.start()
        self._closers = [t for t in self._closers if t.is_alive()] + [thread]
        if len(self._closers) > _CLOSING_WARN:
            logger.warning(
                "이전 녹화 파일을 닫는 중인 것이 %s개로 쌓임 cameraId=%s",
                len(self._closers),
                self.camera_id,
            )

    def _close_quietly(self, recorder: LiveRecorder) -> None:
        try:
            recorder.close()
        except Exception:
            logger.exception(
                "끊김 앞 녹화 파일을 닫다 예외 cameraId=%s", self.camera_id
            )

    def _join_closers(self) -> None:
        deadline = self._clock() + CLOSE_JOIN_SEC
        for thread in self._closers:
            thread.join(max(0.0, deadline - self._clock()))
        if any(t.is_alive() for t in self._closers):
            logger.warning(
                "끊김 앞 녹화 파일을 다 닫지 못한 채 정지함 cameraId=%s", self.camera_id
            )
