"""FNC-S-010 5번: 더미 재생의 실시간 화면·카드 미리보기 프레임 공급

영상을 재생 루프와 같은 위치(반복 시작 시각 S부터 흐른 초)로 읽어 박스를 붙여
CameraStream.publish로 넘긴다. MJPEG 시청자가 있거나 최근 10초 안에 스냅샷 요청이 있을 때(
wants_frames)만 만든다. 재생 루프와 따로 도는 스레드라 1초 결과 전송에 영향을 주지 않는다.
"""

import bisect
import gzip
import json
import logging
import math
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from white_eyes_flask.stream.stream_service import CameraStream, LiveFrame

logger = logging.getLogger(__name__)

IDLE_POLL_SEC = 0.2  # 아무도 안 볼 때 wants_frames를 확인하는 간격
FAILURE_COOLDOWN_SEC = 5.0  # 연속으로 실패하면 이만큼 쉬었다가 다시 시도한다
MAX_FAILURES = 3
# 이 프레임 수 이내의 앞쪽 이동은 시크하지 않고 건너뛰며 읽는다(긴 GOP 영상에서 시크가 느리고 부정확하다)
MAX_FORWARD_SKIP = 90
JOIN_SEC = 2.0
IDLE_CLOSE_SEC = (
    30.0  # 이만큼 아무도 안 보면 영상 파일을 닫는다(다시 보면 열고 시크한다)
)
END_SLACK_FRAMES = 3  # 영상 끝에서 이 프레임 수 이내의 읽기 실패는 끝으로 보고 마지막 프레임을 다시 쓴다


class PreviewUnavailable(Exception):
    """영상을 열거나 읽을 수 없어 이번 프레임을 만들지 못함."""


class VideoFrames:
    """영상 하나를 열어 두고 위치(초)에 맞는 프레임을 돌려준다. 하나의 스레드만 쓴다."""

    def __init__(self, path: Path, opener: Callable[[str], Any] = cv2.VideoCapture):
        self._path = path
        self._opener = opener
        self._capture: Any = None
        self._fps = 0.0
        self._total = 0
        self._last_index: int | None = None
        self._last_frame: Any = None

    def frame_at(self, position_sec: float):
        capture = self._ensure_open()
        index = max(0, int(position_sec * self._fps))
        if self._total > 0:
            index = min(index, self._total - 1)
        if index == self._last_index and self._last_frame is not None:
            return self._last_frame  # 영상 fps보다 자주 요청하면 같은 프레임이다
        ok, frame = self._read(capture, index)
        if not ok or frame is None:
            if self._last_frame is not None and self._near_end(index):
                # 메타 프레임 수가 실제보다 크면 마지막 프레임이 안 읽힐 수 있다. 끝으로 보고 직전 프레임을 쓴다
                return self._last_frame
            self._last_index = None
            self._last_frame = None
            raise PreviewUnavailable(f"영상 프레임을 읽을 수 없음: {self._path.name}")
        # OpenCV는 시크한 뒤 프레임을 정확히 맞춰 읽는다(실제 드론 영상 6개의 무작위 30위치에서 모두 일치함을 확인).
        # 읽은 뒤 CAP_PROP_POS_FRAMES는 "요청한 번호 + 읽은 수"라 실제 디코드 위치를 알려 주지 못하므로 믿지 않는다.
        self._last_index = index
        self._last_frame = frame
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            self._last_index = None
            self._last_frame = None

    def _near_end(self, index: int) -> bool:
        return self._total <= 0 or index >= self._total - END_SLACK_FRAMES

    def _ensure_open(self):
        if self._capture is not None:
            return self._capture
        capture = self._opener(str(self._path))
        try:
            if not capture.isOpened():
                raise PreviewUnavailable(f"영상을 열 수 없음: {self._path.name}")
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps <= 0:
                raise PreviewUnavailable(f"영상 fps를 알 수 없음: {self._path.name}")
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        except Exception as error:
            capture.release()  # 열다가 실패해도 자원을 놓는다
            if isinstance(error, PreviewUnavailable):
                raise
            raise PreviewUnavailable(
                f"영상 정보를 읽을 수 없음: {self._path.name}"
            ) from error
        self._fps = fps
        self._total = total
        self._capture = capture
        self._last_index = None
        self._last_frame = None
        return capture

    def _read(self, capture, index: int):
        last = self._last_index
        if last is not None and 0 < index - last <= MAX_FORWARD_SKIP:
            for _ in range(index - last - 1):
                if not capture.grab():
                    return False, None
            return capture.read()
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        return capture.read()


class BoxTimeline:
    """좌표 파일(.json.gz)에서 위치(초)에 맞는 박스를 찾는다. 규칙: t ≤ 위치인 마지막 프레임."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._times: list[float] = []
        self._rows: list[list[list[float]]] = []
        self._loaded = False

    def rows_at(self, position_sec: float) -> list[list[float]]:
        """[추적 번호, x, y, w, h] 목록(0~1 비율). 읽을 수 없으면 박스 없음으로 본다."""
        if not self._loaded:
            self._load()
        index = bisect.bisect_right(self._times, position_sec) - 1
        return self._rows[index] if index >= 0 else []

    def _load(self) -> None:
        self._loaded = True
        try:
            with gzip.open(self._path, "rt", encoding="utf-8") as stream:
                frames = json.load(stream)["frames"]
            pairs = sorted(
                (float(frame["t"]), list(frame["b"]))
                for frame in frames
                if math.isfinite(float(frame["t"]))
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            # 박스가 없어도 영상은 보여 준다
            logger.warning(
                "좌표 파일을 읽지 못해 박스 없이 보여 줌: %s (%s)",
                self._path.name,
                error,
            )
            return
        self._times = [t for t, _ in pairs]
        self._rows = [rows for _, rows in pairs]


def to_pixels(rows: list[list[float]], width: int, height: int) -> np.ndarray:
    """[번호, x, y, w, h](0~1 비율, 왼쪽 위 기준)를 [x1, y1, x2, y2](픽셀)로 바꾼다. 영상 밖은 잘라 낸다."""
    boxes: list[list[float]] = []
    for row in rows:
        if not isinstance(row, (list, tuple)):
            continue  # 딕셔너리 등은 슬라이스로 읽으면 KeyError가 난다
        try:
            # 추적 번호(row[0])는 None일 수 있고 그리는 데 쓰지 않으므로 좌표만 읽는다
            x, y, w, h = (float(value) for value in row[1:5])
        except (ValueError, TypeError):
            continue
        if not all(math.isfinite(value) for value in (x, y, w, h)):
            continue
        x1, y1 = min(max(x, 0.0), 1.0) * width, min(max(y, 0.0), 1.0) * height
        x2 = min(max(x + w, 0.0), 1.0) * width
        y2 = min(max(y + h, 0.0), 1.0) * height
        if x2 <= x1 or y2 <= y1:
            continue  # 크기가 없거나 거꾸로인 박스는 그리지 않는다
        boxes.append([x1, y1, x2, y2])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


class DummyPreview:
    """더미 캠 하나의 미리보기 프레임 공급. step()은 스레드 없이 한 번 만든다(시험용)."""

    def __init__(
        self,
        stream: CameraStream,
        video_path: Path,
        boxes_path: Path,
        seconds: list[dict[str, Any]],
        position: Callable[[], float | None],
        *,
        fps: int,
        stop: threading.Event,
        clock: Callable[[], float] = time.monotonic,
        opener: Callable[[str], Any] = cv2.VideoCapture,
        idle_close_sec: float = IDLE_CLOSE_SEC,
    ) -> None:
        self._stream = stream
        self._idle_close_sec = idle_close_sec
        self._seconds = seconds
        self._position = position
        self._interval = 1.0 / max(1, fps)
        self._stop = stop
        self._clock = clock
        self._video = VideoFrames(video_path, opener)
        self._boxes = BoxTimeline(boxes_path)
        self._failures = 0
        self._rest_until = 0.0
        # 재생 루프가 예외로 끝나 job의 stop이 서지 않아도 이 스레드를 끌 수 있게 따로 둔다
        self._halt = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, name: str) -> None:
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        """미리보기를 끝내고 스레드가 끝나길 잠시 기다린다."""
        self._halt.set()
        self.join()

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join(JOIN_SEC)
            if self._thread.is_alive():
                logger.warning(
                    "미리보기 스레드가 %s초 안에 끝나지 않음 name=%s",
                    JOIN_SEC,
                    self._thread.name,
                )

    def step(self) -> bool:
        """지금 영상 위치의 프레임을 만들어 넘긴다. 넘겼으면 True."""
        position = self._position()
        if position is None or not self._seconds:
            return False  # 아직 재생이 시작되지 않았다
        if self._clock() < self._rest_until:
            return False
        try:
            image = self._video.frame_at(position)
        except PreviewUnavailable as error:
            self._failed(error)
            return False
        self._failures = 0
        height, width = image.shape[:2]
        boxes = to_pixels(self._boxes.rows_at(position), width, height)
        # 인원수는 캐시의 그 위치 초 결과다(전송하는 k초 결과와 같은 값)
        second = self._seconds[min(int(position), len(self._seconds) - 1)]
        self._stream.publish(LiveFrame(image, boxes, int(second["personCount"])))
        return True

    def _halted(self) -> bool:
        return self._stop.is_set() or self._halt.is_set()

    def _run(self) -> None:
        next_at = time.monotonic()
        idle_since: float | None = None
        try:
            while not self._halted():
                delay = IDLE_POLL_SEC
                try:
                    if self._stream.wants_frames():
                        idle_since = None
                        self.step()
                        # 만드는 데 걸린 시간만큼 줄여서 초당 fps번을 지킨다(밀렸으면 지금부터 다시 센다)
                        next_at = max(next_at + self._interval, time.monotonic())
                        delay = max(0.0, next_at - time.monotonic())
                    else:
                        next_at = time.monotonic()
                        now = time.monotonic()
                        if idle_since is None:
                            idle_since = now
                        elif now - idle_since >= self._idle_close_sec:
                            self._video.close()  # 오래 안 보면 파일·디코더를 놓는다. 다시 보면 새로 연다
                            idle_since = now
                except Exception:
                    logger.exception("미리보기 프레임을 만드는 중 예외")
                    delay = FAILURE_COOLDOWN_SEC
                    next_at = time.monotonic() + delay
                self._halt.wait(delay)
        finally:
            self._video.close()

    def _failed(self, error: Exception) -> None:
        self._failures += 1
        if self._failures == 1:
            logger.warning("미리보기 프레임을 만들지 못함: %s", error)
        if self._failures >= MAX_FAILURES:
            # 영구히 끄지 않고 잠시 쉰 뒤 다시 시도한다(파일이 잠깐 안 읽히는 경우 대비).
            # 캡처가 고장 났을 수 있어 닫아 두고 다음에 새로 연다.
            self._video.close()
            self._rest_until = self._clock() + FAILURE_COOLDOWN_SEC
            logger.warning(
                "미리보기를 %s초 쉼(연속 %s번 실패): %s",
                FAILURE_COOLDOWN_SEC,
                self._failures,
                error,
            )
