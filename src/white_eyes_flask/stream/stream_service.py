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
    render_jpeg,
)

SNAPSHOT_REUSE_SEC = 1.0  # 같은 캠의 스냅샷을 이 시간 동안 다시 쓴다
SNAPSHOT_WANTED_SEC = 10.0  # 마지막 스냅샷 요청 뒤 이 시간 동안은 프레임을 만들어 둔다
FRAME_WAIT_SEC = 1.0  # 새 프레임을 기다리다 정지·끊김을 확인하는 간격


@dataclass(frozen=True)
class LiveFrame:
    """분석 쪽이 넘겨준 최신 프레임. 넘긴 뒤에는 고치지 않는다."""

    image: np.ndarray
    boxes: np.ndarray
    person_count: int


class CameraStream:
    """캠 하나의 최신 프레임·시청자 수·스냅샷 재사용.

    프레임을 넣는 쪽(분석)과 읽는 쪽(요청 스레드)이 달라 모든 상태를 _cond로 지킨다.
    그리기·JPEG 변환은 요청이 있을 때만 하고, 그 결과는 프레임 번호(또는 1초)로 재사용한다.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._cond = threading.Condition()
        self._frame: LiveFrame | None = None
        self._seq = 0
        self._viewers = 0
        self._closed = False
        self._last_snapshot_request: float | None = None
        self._snapshot: tuple[float, bytes] | None = None  # (만든 시각, JPEG)
        self._live_jpeg: tuple[int, bytes] | None = None  # (프레임 번호, JPEG)
        # 같은 캠의 변환이 겹치지 않게 한다. _cond를 쥔 채 변환하면 프레임 넣기가 밀린다.
        self._encode_lock = threading.Lock()

    def publish(self, frame: LiveFrame) -> None:
        """분석 쪽이 최신 프레임을 넣는다. 가장 최근 것 하나만 남는다."""
        with self._cond:
            if self._closed:
                return
            self._frame = frame
            self._seq += 1
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
        with self._encode_lock:
            now = self._clock()
            with self._cond:
                self._last_snapshot_request = now
                cached = self._snapshot
                frame = self._frame
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
            seen = 0
            sent_at: float | None = None
            while True:
                with self._cond:
                    self._cond.wait_for(
                        lambda: self._closed or self._seq != seen,
                        timeout=FRAME_WAIT_SEC,
                    )
                    if self._closed:
                        return
                    if self._seq == seen or self._frame is None:
                        continue
                    seen = self._seq
                jpeg = self._live_jpeg_for(seen)
                if jpeg is None:
                    continue
                yield jpeg
                sent_at = self._clock()
                if min_interval > 0:
                    self._pause_until(sent_at + min_interval)
        finally:
            with self._cond:
                self._viewers -= 1

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
        with self._cond:
            self._cond.wait_for(
                lambda: self._closed,
                timeout=max(0.0, deadline - self._clock()),
            )


def _render(frame: LiveFrame, width: int, quality: int) -> bytes:
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
        self._lock = threading.Lock()
        self._streams: dict[int, CameraStream] = {}

    def open(self, camera_id: int) -> CameraStream:
        with self._lock:
            old = self._streams.get(camera_id)
            stream = CameraStream()
            self._streams[camera_id] = stream
        if old is not None:
            old.close()
        return stream

    def close(self, camera_id: int) -> None:
        with self._lock:
            stream = self._streams.pop(camera_id, None)
        if stream is not None:
            stream.close()

    def get(self, camera_id: int) -> CameraStream | None:
        """캠 작업이 없으면 None(→ 404)."""
        with self._lock:
            return self._streams.get(camera_id)
