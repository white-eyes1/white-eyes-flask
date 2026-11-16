"""FNC-S-011 MJPEG·스냅샷용 최신 프레임 보관과 시청자 수 (StreamService)"""

import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from white_eyes_flask.stream.frame_renderer import (
    LIVE_QUALITY,
    LIVE_WIDTH,
    SNAPSHOT_QUALITY,
    SNAPSHOT_WIDTH,
    no_signal_jpeg,
    render_jpeg,
)

SNAPSHOT_REUSE_SEC = 1.0  # 같은 캠의 스냅샷을 이 시간 동안 다시 쓴다
SNAPSHOT_WANTED_SEC = 10.0  # 마지막 스냅샷 요청 뒤 이 시간 동안은 프레임을 만들어 둔다
FRAME_WAIT_SEC = 1.0  # 새 프레임을 기다리다 정지·끊김을 확인하는 간격
IDLE_RESEND_SEC = 2.0  # 새 프레임이 없을 때 이 시간마다 이전 이미지를 다시 보낸다


@dataclass(frozen=True)
class LiveFrame:
    """분석 쪽이 넘겨준 최신 프레임. 넘긴 뒤에는 고치지 않는다."""

    # 원본 프레임(BGR)
    image: np.ndarray
    # 원본 픽셀 좌표의 박스 (N, 4) [x1, y1, x2, y2]
    boxes: np.ndarray
    # 화면에 표시할 인원수
    person_count: int


class CameraStream:
    """캠 하나의 최신 프레임·시청자 수·스냅샷 재사용.

    프레임을 넣는 쪽(분석)과 읽는 쪽(요청 스레드)이 달라 모든 상태를 _cond로 지킨다.
    그리기·JPEG 변환은 요청이 있을 때만 하고, 그 결과는 프레임 번호(또는 1초)로 재사용한다.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        # 시간 측정 함수(시험에서 가짜 시계로 바꾸려고 주입받는다)
        self._clock = clock
        # 모든 상태를 지키는 잠금 + 새 프레임·종료를 알리는 조건 변수
        self._cond = threading.Condition()
        # 가장 최근 프레임과 그 번호(프레임이 들어올 때마다 1 증가. 시청자가 새 프레임인지 판단하는 기준)
        self._frame: LiveFrame | None = None
        self._seq = 0
        # 지금 MJPEG을 보고 있는 연결 수
        self._viewers = 0
        # 캠 작업이 끝났는지
        self._closed = False
        # 마지막 스냅샷 요청 시각(wants_frames가 "최근에 요청이 있었나" 판단하는 데 쓴다)
        self._last_snapshot_request: float | None = None
        self._snapshot: tuple[float, bytes] | None = None  # (만든 시각, JPEG)
        self._live_jpeg: tuple[int, bytes] | None = None  # (프레임 번호, JPEG)
        # 같은 캠의 변환이 겹치지 않게 한다. _cond를 쥔 채 변환하면 프레임 넣기가 밀린다.
        self._encode_lock = threading.Lock()

    def publish(self, frame: LiveFrame) -> None:
        """분석 쪽이 최신 프레임을 넣는다. 가장 최근 것 하나만 남는다."""
        with self._cond:
            # 닫힌 뒤에 늦게 도착한 프레임은 무시한다
            if self._closed:
                return
            self._frame = frame
            self._seq += 1
            # 기다리는 시청자 연결들을 모두 깨운다
            self._cond.notify_all()

    def wants_frames(self) -> bool:
        """MJPEG 시청자가 있거나 최근 10초 안에 스냅샷 요청이 있으면 True(분석 쪽이 확인)."""
        with self._cond:
            if self._viewers > 0:
                return True
            last = self._last_snapshot_request
            return last is not None and self._clock() - last <= SNAPSHOT_WANTED_SEC

    @property
    def viewer_count(self) -> int:
        """지금 MJPEG을 보는 연결 수."""
        with self._cond:
            return self._viewers

    def close(self) -> None:
        """캠 작업이 끝났다. 기다리던 MJPEG 연결이 끝나도록 깨운다."""
        with self._cond:
            self._closed = True
            self._frame = None
            self._snapshot = None
            self._live_jpeg = None
            self._cond.notify_all()

    def snapshot_jpeg(self) -> bytes | None:
        """카드 미리보기 JPEG. 1초 안의 요청은 같은 이미지를 돌려준다. 프레임이 아직 없으면 None."""
        # 같은 캠의 변환이 겹치지 않게 변환 잠금을 먼저 잡는다
        with self._encode_lock:
            now = self._clock()
            # 요청 시각을 기록해 두면 wants_frames가 True가 되어 분석 쪽이 프레임을 넣어 준다.
            # 상태는 짧게만 잠그고(복사해 나옴), 오래 걸리는 그리기·인코딩은 잠금 밖에서 한다
            with self._cond:
                self._last_snapshot_request = now
                cached = self._snapshot
                frame = self._frame
            # 만든 지 1초 안이면 다시 그리지 않고 재사용한다(카드 여러 장이 동시에 요청해도 부담을 줄인다)
            if cached is not None and now - cached[0] < SNAPSHOT_REUSE_SEC:
                return cached[1]
            if frame is None:
                return None
            jpeg = _render(frame, SNAPSHOT_WIDTH, SNAPSHOT_QUALITY)
            with self._cond:
                if not self._closed:
                    self._snapshot = (self._clock(), jpeg)  # 변환이 끝난 때부터 1초
            return jpeg

    def live_jpegs(self, min_interval: float = 0.0) -> Iterator[bytes]:
        """MJPEG용 JPEG를 새 프레임이 나올 때마다 내보낸다. 캠 작업이 끝나면 멈춘다.

        이 생성기가 처음 실행될 때 시청자가 늘고, 닫히거나 끝나면(클라이언트가 끊을 때
        포함) 줄어든다. 시청자가 없으면 아무도 그리지 않는다.
        min_interval은 연속 두 프레임 사이의 최소 간격(초)이다.
        """
        with self._cond:
            if self._closed:
                return
            self._viewers += 1
        try:
            # 이미 보낸 가장 최근 프레임 번호와 마지막으로 보낸 시각
            seen = 0
            sent_at = self._clock()
            while True:
                with self._cond:
                    # 새 프레임·종료가 오면 즉시 깨고, 아무 일 없어도 FRAME_WAIT_SEC마다 깨어 아래 조건들을 점검한다
                    self._cond.wait_for(
                        lambda: self._closed or self._seq != seen,
                        timeout=FRAME_WAIT_SEC,
                    )
                    if self._closed:
                        return
                    fresh = self._seq != seen and self._frame is not None
                    if fresh:
                        seen = self._seq
                if fresh:
                    jpeg = self._live_jpeg_for(seen)
                elif self._clock() - sent_at >= IDLE_RESEND_SEC:
                    # 프레임이 안 들어와도(RTSP 끊김) 가끔 보내야, 창을 닫았을 때 쓰기 오류로 알아챈다
                    jpeg = self._idle_jpeg()
                else:
                    # 새 프레임도 없고 재전송 시간도 안 됐으면 다시 기다린다
                    continue
                if jpeg is None:
                    continue
                yield jpeg
                sent_at = self._clock()
                # 너무 빠르게 보내지 않도록 최소 간격까지 쉰다(그 사이 닫히면 바로 깬다)
                if min_interval > 0:
                    self._pause_until(sent_at + min_interval)
        finally:
            # 정상 종료·예외·클라이언트가 연결을 끊어 제너레이터가 닫힌 경우 모두 시청자 수를 되돌린다
            with self._cond:
                self._viewers -= 1

    def _idle_jpeg(self) -> bytes:
        """새 프레임이 없을 때 다시 보낼 이미지: 마지막으로 보낸 프레임, 없으면 '신호 없음' 이미지."""
        with self._cond:
            cached = self._live_jpeg
        return cached[1] if cached is not None else no_signal_jpeg()

    def _live_jpeg_for(self, seq: int) -> bytes | None:
        """프레임 번호 seq의 JPEG. 시청자가 여럿이어도 한 번만 변환한다."""
        with self._encode_lock:
            with self._cond:
                cached = self._live_jpeg
                frame = self._frame if self._seq == seq else None
            if cached is not None and cached[0] == seq:
                return cached[1]
            if frame is None:
                return None  # 그 사이 더 새 프레임이 들어왔다. 다음 반복에서 읽는다.
            jpeg = _render(frame, LIVE_WIDTH, LIVE_QUALITY)
            with self._cond:
                if not self._closed:
                    self._live_jpeg = (seq, jpeg)
            return jpeg

    def _pause_until(self, deadline: float) -> None:
        """deadline(clock 기준 시각)까지 쉰다. 단 캠 작업이 끝나면 즉시 깨어난다."""
        with self._cond:
            self._cond.wait_for(
                lambda: self._closed,
                timeout=max(0.0, deadline - self._clock()),
            )


def _render(frame: LiveFrame, width: int, quality: int) -> bytes:
    """LiveFrame을 지정한 크기·품질의 JPEG로 그린다."""
    return render_jpeg(
        frame.image,
        frame.boxes,
        frame.person_count,
        width=width,
        quality=quality,
    )


class StreamService:
    """캠 작업이 있는 동안만 CameraStream을 가진다. 캠 작업 시작·정지(CameraService)가 연다·닫는다."""

    def __init__(self) -> None:
        # _streams 사전을 여러 스레드(캠 작업, 요청 처리)가 동시에 고치지 않게 하는 잠금
        self._lock = threading.Lock()
        # 캠 번호 -> 스트림
        self._streams: dict[int, CameraStream] = {}

    def open(self, camera_id: int) -> CameraStream:
        """캠 작업이 시작될 때 새 스트림을 만든다. 같은 캠의 옛 스트림이 남아 있으면 닫는다."""
        with self._lock:
            old = self._streams.get(camera_id)
            stream = CameraStream()
            self._streams[camera_id] = stream
        # 잠금 밖에서 닫아, 닫기(기다리는 시청자 깨우기)가 사전 잠금을 오래 잡지 않게 한다
        if old is not None:
            old.close()
        return stream

    def close(self, camera_id: int) -> None:
        """캠 작업이 끝날 때 스트림을 지우고 닫는다. 없으면 아무것도 하지 않는다."""
        with self._lock:
            stream = self._streams.pop(camera_id, None)
        if stream is not None:
            stream.close()

    def get(self, camera_id: int) -> CameraStream | None:
        """캠 작업이 없으면 None(→ 404)."""
        with self._lock:
            return self._streams.get(camera_id)
