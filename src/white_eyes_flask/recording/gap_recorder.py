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
    """끊김이 있으면 녹화 파일을 새로 시작하는 래퍼.

    안쪽의 LiveRecorder 하나를 쓰다가, 프레임이 gap_sec 이상 끊겼다 돌아오면 그 녹화기를
    백그라운드에서 마무리하고 새 녹화기로 이어 간다. LiveRecorder와 같은 write_frame·add_boxes·close를 가진다.
    """

    def __init__(
        self,
        camera_id: int,
        make_recorder: Callable[[], LiveRecorder],
        gap_sec: float = GAP_SEC,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """래퍼를 만든다.

        :param camera_id: 캠 번호
        :param make_recorder: 새 LiveRecorder를 만들어 돌려주는 함수(복구할 때마다 부른다)
        :param gap_sec: 이 시간 이상 프레임이 없으면 끊긴 것으로 본다
        :param clock: 현재 시각(초). 시험에서 바꿔 끼운다
        """
        self.camera_id = camera_id
        self._make = make_recorder
        self._gap_sec = gap_sec
        self._clock = (
            clock  # 끊김 판정에는 시스템 시계 보정에 흔들리지 않는 시계를 쓴다
        )
        # _current와 _closed를 지킨다(캠 작업 스레드와 분석 스레드가 함께 쓴다).
        self._lock = threading.Lock()
        # 지금 쓰는 녹화기. 첫 프레임이 오기 전이나 닫은 뒤에는 None.
        self._current: LiveRecorder | None = None
        # 마지막 프레임이 온 시각(clock 기준). 끊김 판정에 쓴다.
        self._last_frame_at: float | None = None
        # 이전 녹화기를 닫고 있는 백그라운드 스레드들.
        self._closers: list[threading.Thread] = []
        self._closed = False

    # ---- 캠 작업 스레드가 부른다 ----

    def write_frame(self, frame: Any, at: float) -> None:
        """LiveRecorder.write_frame과 같다. 끊겼다 복구된 첫 프레임이면 새 파일을 시작한다."""
        # 끊김을 판정할 현재 시각을 먼저 읽는다.
        now = self._clock()
        # 잠금 안에서는 상태만 읽는다. 녹화기를 닫거나 만드는 느린 일은 잠금 밖에서 한다.
        with self._lock:
            # close가 끝난 뒤 온 프레임은 버린다.
            if self._closed:
                return
            current = self._current
        # 직전 프레임 이후 gap_sec 이상 지났으면 스트림이 끊겼다 돌아온 것이다.
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
            # 이전 녹화기는 백그라운드에서 마무리하고, 이 프레임부터는 새 녹화기를 쓴다.
            self._retire(current)
            current = None
        # 첫 프레임이거나 방금 끊김으로 이전 녹화기를 보냈다.
        if current is None:
            current = self._make()
            with self._lock:
                self._current = current
        # 이 값은 캠 작업 스레드(write_frame을 부르는 스레드)만 읽고 쓴다.
        self._last_frame_at = now
        current.write_frame(frame, at)

    def add_boxes(self, at: float, boxes: list[TrackedBox]) -> None:
        """분석한 프레임의 박스. 끊김 직전 프레임의 박스가 복구 뒤에 도착하면 새 파일의 시작보다 이르다고 버려진다."""
        # 지금 쓰는 녹화기에게 넘긴다. 끊긴 동안(녹화기가 없을 때)의 박스는 버려진다.
        with self._lock:
            current = self._current
        if current is not None:
            current.add_boxes(at, boxes)

    def close(self) -> None:
        """녹화를 마무리한다.

        현재 녹화기를 닫고, 이전에 닫는 중이던 녹화기들도 CLOSE_JOIN_SEC까지 기다린다.
        두 번 불러도 한 번만 실행한다.
        """
        with self._lock:
            # 두 번째 호출은 아무것도 하지 않는다.
            if self._closed:
                return
            self._closed = True
            # 현재 녹화기를 잠금 안에서 떼어 내고, 닫는 일은 잠금 밖에서 한다.
            current, self._current = self._current, None
        try:
            if current is not None:
                current.close()
        # 현재 녹화기를 닫다 예외가 나도 이전 녹화기들이 끝나길 기다린다.
        finally:
            self._join_closers()

    # ---- 이전 녹화기 마무리 ----

    def _retire(self, recorder: LiveRecorder) -> None:
        """이전 녹화기를 백그라운드에서 닫는다. 복구한 프레임을 읽는 스레드를 오래 막지 않기 위해서다."""
        # 닫는 동안 add_boxes가 이 녹화기로 박스를 보내지 않게 먼저 떼어 낸다.
        with self._lock:
            self._current = None
        # 스레드 이름은 로그에 cam-{번호}-recording으로 나온다.
        thread = threading.Thread(
            target=self._close_quietly,
            args=(recorder,),
            name=f"cam-{self.camera_id}-recording",
            daemon=True,
        )
        thread.start()
        # 이미 끝난 스레드는 목록에서 빼고 새 스레드를 더한다.
        self._closers = [t for t in self._closers if t.is_alive()] + [thread]
        if len(self._closers) > _CLOSING_WARN:
            logger.warning(
                "이전 녹화 파일을 닫는 중인 것이 %s개로 쌓임 cameraId=%s",
                len(self._closers),
                self.camera_id,
            )

    def _close_quietly(self, recorder: LiveRecorder) -> None:
        """이전 녹화기를 닫는다. 예외는 로그만 남기고 삼킨다(백그라운드 스레드가 조용히 죽지 않게)."""
        try:
            recorder.close()
        except Exception:
            logger.exception(
                "끊김 앞 녹화 파일을 닫다 예외 cameraId=%s", self.camera_id
            )

    def _join_closers(self) -> None:
        """이전 녹화기를 닫는 스레드들이 끝나길 기다린다. 전체 대기는 CLOSE_JOIN_SEC를 넘지 않는다."""
        deadline = self._clock() + CLOSE_JOIN_SEC
        for thread in self._closers:
            # 스레드마다 남은 시간만 기다려 전체 대기 시간이 CLOSE_JOIN_SEC를 넘지 않게 한다.
            thread.join(max(0.0, deadline - self._clock()))
        if any(t.is_alive() for t in self._closers):
            logger.warning(
                "끊김 앞 녹화 파일을 다 닫지 못한 채 정지함 cameraId=%s", self.camera_id
            )
