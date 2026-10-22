"""FNC-S-008 라이브 녹화(1분 파일)

원본 프레임을 ffmpeg에 넘겨 시계 분 단위 mp4로 녹화하고, 분석 프레임의 박스를 같은 분 단위
좌표 파일로 저장한다. 분이 끝나 파일이 완성되면 게시기(RecordingPublisher)에 넘긴다.

호출 규칙: write_frame은 캠 작업 스레드 하나에서만, close는 그 스레드가 끝날 때 부른다.
add_boxes는 분석 스레드에서 불러도 된다(박스 잠금으로 지킨다). close 전에 분석 스레드를 먼저 끝낸다.
"""

import contextlib
import csv
import logging
import os
import queue
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from white_eyes_flask.analysis.result import TrackedBox
from white_eyes_flask.recording.box_file import write_box_file
from white_eyes_flask.recording.ffmpeg import (
    SEGMENT_NAME,
    RecorderSettings,
    build_command,
    segment_list_path,
    segment_pattern,
)
from white_eyes_flask.recording.naming import (
    MINUTE_MS,
    file_stem,
    minute_start_ms,
    nearest_minute_ms,
)
from white_eyes_flask.recording.publisher import FinishedRecording, RecordingPublisher

logger = logging.getLogger(__name__)

# ffmpeg가 느려 쓰기가 막혀도 분석 스레드가 멈추지 않게, 가득 차면 프레임을 버린다
QUEUE_FRAMES = 60
POLL_SEC = 1.0
# close()가 쓰는 시간의 합이 캠 정지 대기(10초)를 넘지 않게 한다: 2 + 3 + 1 + 1 + 1 = 최대 8초
FLUSH_WAIT_SEC = 2.0
EXIT_WAIT_SEC = 3.0
KILL_WAIT_SEC = 1.0
WATCH_JOIN_SEC = 1.0
# 파일이 닫힌 뒤 분석이 그 분의 박스를 다 넘길 때까지 기다리는 최대 시간(초)
BOX_GRACE_SEC = 10.0
# 아직 파일에 넣지 못한 박스를 쌓아 두는 상한(5fps로 20분). 넘으면 오래된 것부터 버린다
BOX_BACKLOG_LIMIT = 6000
# 프레임을 버린 로그는 첫 번째와 이 횟수마다 한 번만 남긴다
_DROP_LOG_EVERY = 100
# 좌표 파일을 쓰지 못한 파일을 다시 시도하는 최대 횟수(넘으면 영상은 임시 폴더에 남긴다)
MAX_BOX_WRITE_ATTEMPTS = 3


class Recorder(Protocol):
    def close(self) -> None: ...


class RecorderUnavailable(RuntimeError):
    """녹화를 시작할 수 없다(ffmpeg 없음 등). 캠 작업은 ERROR로 처리한다."""


class RecorderFailed(RuntimeError):
    """녹화 중에 ffmpeg가 죽었다. 이후 녹화는 남지 않으므로 캠 작업은 ERROR로 처리한다."""


class LiveRecorder:
    def __init__(
        self,
        camera_id: int,
        settings: RecorderSettings,
        publisher: RecordingPublisher,
        popen: Callable[..., Any] = subprocess.Popen,
        box_grace_sec: float = BOX_GRACE_SEC,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.camera_id = camera_id
        self._settings = settings
        self._publisher = publisher
        self._popen = popen
        self._box_grace_sec = box_grace_sec
        self._clock = clock
        self._tmp_dir = Path(settings.tmp_dir)
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=QUEUE_FRAMES)
        self._closing = threading.Event()
        self._proc: Any = None
        self._error_log: Path | None = None
        self._list_path: Path | None = None
        self._pattern: Path | None = None
        self._list_offset = 0
        # 닫혔다고 목록에 오른 파일 순번 → (이름, 본 시각, 목록의 시작·끝 초). 박스가 모이면 게시기로 넘긴다
        self._waiting: dict[int, tuple[str, float, float | None, float | None]] = {}
        # 이미 게시기에 넘긴 파일 순번. segment list가 처음부터 다시 쓰여도 두 번 넘기지 않는다
        self._finished: set[int] = set()
        self._listed: set[int] = set()  # 목록에 오른 적 있는 파일 순번
        self._box_write_failures: dict[int, int] = {}
        self._unpublished: set[int] = (
            set()
        )  # 좌표 파일을 쓰지 못해 게시하지 못한 파일 순번
        self._collect_lock = threading.Lock()
        self._box_lock = threading.Lock()
        self._boxes: list[tuple[int, list[TrackedBox]]] = []
        self._box_watermark_ms = 0  # 지금까지 받은 박스 중 가장 늦은 프레임 시각
        self._backlog_warned = False
        self._first_ms: int | None = None
        self._last_ms = 0
        self._size: tuple[int, int] | None = None  # (높이, 너비) 첫 프레임 기준
        self._resize_warned = False
        self._dropped = 0
        self._died = False
        self._pump: threading.Thread | None = None
        self._watch: threading.Thread | None = None
        self._closed = False

    # ---- 캠 작업 스레드가 부른다 ----

    def write_frame(self, frame, at: float) -> None:
        """원본 프레임 한 장(BGR, 높이×너비×3)과 그 프레임을 읽은 시각(epoch 초).

        ffmpeg가 죽었으면 RecorderFailed를 던진다(이후 녹화가 남지 않는다).
        """
        at_ms = int(at * 1000)
        if self._proc is None:
            self._open(frame, at_ms)
        if self._died:
            raise RecorderFailed(f"녹화 ffmpeg가 죽었음 cameraId={self.camera_id}")
        frame = self._fit(frame)
        self._last_ms = max(self._last_ms, at_ms)
        try:
            self._queue.put_nowait(frame.tobytes())
        except queue.Full:
            self._dropped += 1
            if self._dropped == 1 or self._dropped % _DROP_LOG_EVERY == 0:
                logger.warning(
                    "녹화가 따라가지 못해 프레임을 버림 cameraId=%s (누적 %s장)",
                    self.camera_id,
                    self._dropped,
                )

    def add_boxes(self, at: float, boxes: list[TrackedBox]) -> None:
        """분석한 프레임의 박스. at은 그 프레임을 읽은 시각(epoch 초)."""
        if self._died:
            return  # 파일이 나오지 않으니 쌓아 두지 않는다
        at_ms = int(at * 1000)
        with self._box_lock:
            self._boxes.append((at_ms, boxes))
            self._box_watermark_ms = max(self._box_watermark_ms, at_ms)
            if len(self._boxes) > BOX_BACKLOG_LIMIT:
                del self._boxes[: BOX_BACKLOG_LIMIT // 10]
                if not self._backlog_warned:
                    self._backlog_warned = True
                    logger.warning(
                        "녹화 파일이 나오지 않아 오래된 박스를 버림 cameraId=%s",
                        self.camera_id,
                    )

    def close(self) -> None:
        """녹화를 마무리한다. 마지막 파일(1분 미만 포함)까지 게시기에 넘긴 뒤 돌아온다.

        걸리는 시간은 최대 약 8초다(캠 정지 대기 10초 안).
        """
        if self._proc is None or self._closed:
            return
        self._closed = True
        self._closing.set()
        stuck = False
        if self._pump is not None:
            self._pump.join(FLUSH_WAIT_SEC)
            stuck = self._pump.is_alive()
            if stuck:
                logger.warning(
                    "녹화 프레임을 다 넘기지 못함 cameraId=%s", self.camera_id
                )
        # 쓰기가 막혀 있으면 stdin이 닫히지 않아 ffmpeg가 끝나길 기다려도 소용없다
        self._stop_process(wait=not stuck)
        if self._watch is not None:
            self._watch.join(WATCH_JOIN_SEC)
        self.poll_segments()
        self._warn_leftovers()
        self._report_exit()

    # ---- 시작·감시 ----

    def _open(self, frame, first_ms: int) -> None:
        height, width = frame.shape[:2]
        self._size = (height, width)
        self._first_ms = first_ms
        token = str(first_ms)
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        self._list_path = segment_list_path(self._tmp_dir, self.camera_id, token)
        self._pattern = segment_pattern(self._tmp_dir, self.camera_id, token)
        command = build_command(
            self._settings, width, height, self._pattern, self._list_path
        )
        self._error_log = self._tmp_dir / f"cam{self.camera_id}_{token}.log"
        try:
            with open(self._error_log, "wb") as errors:
                self._proc = self._popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=errors,
                )
        except OSError as error:
            raise RecorderUnavailable(
                f"녹화용 ffmpeg를 실행할 수 없음({self._settings.ffmpeg_bin}): {error}"
            ) from error
        # 스레드 이름은 로그의 요청 ID 칸에 cam-{cameraId}로 나오는 규칙을 따른다
        name = f"cam-{self.camera_id}-recording"
        self._pump = threading.Thread(target=self._pump_frames, name=name, daemon=True)
        self._watch = threading.Thread(
            target=self._watch_segments, name=name, daemon=True
        )
        self._pump.start()
        self._watch.start()
        logger.info("라이브 녹화 시작 cameraId=%s", self.camera_id)

    def _fit(self, frame):
        """첫 프레임과 크기가 다르면 맞춘다. 그대로 쓰면 이후 영상이 전부 깨진다."""
        assert self._size is not None
        if frame.shape[:2] == self._size:
            return frame
        if not self._resize_warned:
            self._resize_warned = True
            logger.warning(
                "프레임 크기가 바뀌어 처음 크기로 맞춤 cameraId=%s %s -> %s",
                self.camera_id,
                frame.shape[:2],
                self._size,
            )
        import cv2  # 크기가 바뀔 때만 쓴다

        height, width = self._size
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    def _pump_frames(self) -> None:
        """큐의 프레임을 ffmpeg stdin에 쓴다. 쓰기가 막혀도 분석 스레드는 영향을 받지 않는다."""
        stdin = self._proc.stdin
        broken = False
        try:
            while True:
                try:
                    data = self._queue.get(timeout=0.5)
                except queue.Empty:
                    if self._closing.is_set():
                        break
                    continue
                if broken:
                    continue
                try:
                    stdin.write(data)
                except (BrokenPipeError, OSError):
                    broken = True
                    logger.error("녹화 ffmpeg에 쓸 수 없음 cameraId=%s", self.camera_id)
        finally:
            with contextlib.suppress(OSError):
                stdin.close()

    def _watch_segments(self) -> None:
        while not self._closing.wait(POLL_SEC):
            try:
                self._check_alive()
                self.poll_segments()
            except Exception:
                logger.exception("녹화 파일 확인 중 예외 cameraId=%s", self.camera_id)

    def _check_alive(self) -> None:
        """녹화 중에 ffmpeg가 죽었으면 알린다. 캠 작업은 다음 write_frame에서 RecorderFailed를 받는다."""
        if self._died or self._closing.is_set():
            return
        code = self._proc.poll()
        if code is not None:
            self._died = True
            tail = self._error_tail()
            logger.error(
                "녹화 ffmpeg가 녹화 중에 죽음 cameraId=%s code=%s: %s",
                self.camera_id,
                code,
                tail,
            )

    # ---- 파일 완성 처리 ----

    def poll_segments(self) -> None:
        """segment list에 새로 올라온 파일(= 닫힌 파일)을 읽고, 박스가 모인 파일을 게시기에 넘긴다."""
        if self._list_path is None:
            return
        with self._collect_lock:
            self._read_list()
            self._flush_waiting()

    def _read_list(self) -> None:
        try:
            with open(self._list_path, "rb") as stream:
                if os.fstat(stream.fileno()).st_size < self._list_offset:
                    self._list_offset = 0  # 목록이 처음부터 다시 쓰였다
                stream.seek(self._list_offset)
                data = stream.read()
        except FileNotFoundError:
            return
        # 줄 끝까지 쓰이지 않은 마지막 줄은 다음에 읽는다
        complete = data[: data.rfind(b"\n") + 1]
        self._list_offset += len(complete)
        text = complete.decode("utf-8", errors="replace")
        for row in csv.reader(text.splitlines()):
            if not row:
                continue
            name = row[0].strip()
            match = SEGMENT_NAME.search(name)
            if not match:
                continue
            index = int(match.group(1))
            self._listed.add(index)
            if (
                index not in self._finished
                and index not in self._waiting
                and index not in self._unpublished
            ):
                start_s, end_s = _times(row)
                self._waiting[index] = (name, self._clock(), start_s, end_s)

    def _flush_waiting(self) -> None:
        """닫힌 파일을 순서대로 넘긴다. 분석이 그 분의 박스를 다 넘기지 않았으면 조금 기다린다."""
        for index in sorted(self._waiting):
            name, seen_at, start_s, end_s = self._waiting[index]
            _, end_ms = self._window(index, start_s, end_s)
            if not self._ready(end_ms, seen_at):
                return  # 뒤 파일은 앞 파일 다음에 넘긴다
            del self._waiting[index]
            self._finish_segment(index, name, start_s, end_s)

    def _ready(self, end_ms: int, seen_at: float) -> bool:
        if self._closing.is_set():
            return True
        if self._box_watermark_ms >= end_ms:
            return True  # 분석이 이 파일의 끝을 지났으니 박스가 다 왔다
        return self._clock() - seen_at >= self._box_grace_sec

    def _finish_segment(
        self, index: int, name: str, start_s: float | None, end_s: float | None
    ) -> None:
        video = self._tmp_dir / Path(name).name
        started_ms, window_end_ms = self._window(index, start_s, end_s)
        # 정지로 닫힌 마지막 파일이면 그 뒤의 박스도 모두 이 파일 몫이다
        is_last = self._closing.is_set() and not any(i > index for i in self._waiting)
        with self._box_lock:
            # 이 파일의 시간대 박스만 쓴다. 시작보다 이른 박스는 앞 파일이 이미 나간 뒤 늦게
            # 도착한 것이라 버린다. 박스는 좌표 파일을 쓴 뒤에 지운다(쓰기에 실패하면 다시 시도해야 한다).
            inside = [
                i
                for i in self._boxes
                if i[0] >= started_ms and (is_last or i[0] < window_end_ms)
            ]
        ended_ms = window_end_ms
        if self._closing.is_set():
            # 목록의 끝 시각은 마지막 프레임의 길이와 마무리만큼 더 길다
            frame_ms = 1000 // max(1, self._settings.fps)
            ended_ms = min(ended_ms, self._last_ms + frame_ms)
        # 관리자 앱은 endedAt이 startedAt보다 늦어야 받는다(프레임이 한 장뿐인 파일 대비)
        ended_ms = max(ended_ms, started_ms + 1)
        # 첫 파일은 가동 시각이 속한 분, 그 뒤 파일은 ffmpeg가 끊은 시각에서 가장 가까운 분이 이름이다
        name_ms = started_ms if index == 0 else nearest_minute_ms(started_ms)
        stem = file_stem(self.camera_id, name_ms)
        box_path = self._tmp_dir / f"{stem}_{index:05d}.json.gz"
        frames = [((at - started_ms) / 1000, boxes) for at, boxes in inside]
        try:
            write_box_file(box_path, self._settings.analyze_fps, frames)
        except OSError:
            self._box_write_failed(index, name, start_s, end_s)
            return
        with self._box_lock:
            # 끝보다 늦은 박스는 다음 파일 몫으로 남긴다
            self._boxes = (
                [] if is_last else [i for i in self._boxes if i[0] >= window_end_ms]
            )
        self._publisher.submit(
            FinishedRecording(
                self.camera_id,
                started_ms,
                ended_ms,
                video,
                box_path,
                name_ms=name_ms,
            )
        )
        self._finished.add(index)
        logger.info("녹화 파일 완성 cameraId=%s file=%s", self.camera_id, video.name)

    def _box_write_failed(
        self, index: int, name: str, start_s: float | None, end_s: float | None
    ) -> None:
        """좌표 파일을 쓰지 못했다. 영상이 임시 폴더에 묻히지 않게 다음 확인 때 다시 시도한다."""
        failures = self._box_write_failures.get(index, 0) + 1
        self._box_write_failures[index] = failures
        if failures < MAX_BOX_WRITE_ATTEMPTS:
            logger.warning(
                "좌표 파일을 쓰지 못해 다시 시도함 cameraId=%s file=%s (%s번째)",
                self.camera_id,
                name,
                failures,
            )
            self._waiting[index] = (name, self._clock(), start_s, end_s)
            return
        self._unpublished.add(index)
        logger.exception(
            "좌표 파일을 %s번 쓰지 못해 게시하지 못함(영상은 임시 폴더에 남음) cameraId=%s file=%s",
            failures,
            self.camera_id,
            name,
        )

    def _window(
        self, index: int, start_s: float | None, end_s: float | None
    ) -> tuple[int, int]:
        """파일이 담은 구간(시작, 끝) 밀리초.

        ffmpeg가 목록에 적은 시작·끝 시각(첫 프레임부터의 초)으로 계산한다. ffmpeg는 분 경계가 아니라
        자기가 시작한 시점을 기준으로 첫 파일을 끊기 때문에(실제 ffmpeg로 확인) 분 경계를 가정하면
        어긋난다. 목록에서 시각을 읽지 못한 경우에만 분 경계를 가정한다.
        """
        assert self._first_ms is not None
        if start_s is not None and end_s is not None:
            return (
                self._first_ms + round(start_s * 1000),
                self._first_ms + round(end_s * 1000),
            )
        return self._bounds(index)

    def _bounds(self, index: int) -> tuple[int, int]:
        """index번째 파일의 (시작, 끝 분 경계) 밀리초. 목록에 시각이 없을 때만 쓰는 가정이다."""
        assert self._first_ms is not None
        first_minute = minute_start_ms(self._first_ms)
        started = self._first_ms if index == 0 else first_minute + index * MINUTE_MS
        return started, first_minute + (index + 1) * MINUTE_MS

    # ---- 종료 ----

    def _stop_process(self, wait: bool = True) -> None:
        proc = self._proc
        if wait:
            try:
                proc.wait(timeout=EXIT_WAIT_SEC)
                return
            except subprocess.TimeoutExpired:
                logger.warning(
                    "녹화 ffmpeg가 끝나지 않아 종료시킴 cameraId=%s", self.camera_id
                )
        proc.terminate()
        try:
            proc.wait(timeout=KILL_WAIT_SEC)
        except subprocess.TimeoutExpired:
            proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=KILL_WAIT_SEC)

    def _warn_leftovers(self) -> None:
        """게시하지 못한 파일이 남아 있으면 이유를 구분해 알린다.

        목록에 오르지 않은 파일은 닫히지 못한 것이라 게시하지 않는다(강제 종료 등). 목록에는 올랐는데
        게시하지 못한 파일은 좌표 파일을 쓰지 못한 것이라 영상은 온전하다.
        """
        if self._pattern is None:
            return
        pending = self._unpublished | set(self._waiting)
        for path in sorted(self._tmp_dir.glob(self._pattern.name.replace("%05d", "*"))):
            match = SEGMENT_NAME.search(path.name)
            if not match:
                continue
            index = int(match.group(1))
            if index in self._finished:
                continue
            if index in pending:
                logger.warning(
                    "좌표 파일을 쓰지 못해 게시하지 못한 녹화 파일이 남음(영상은 온전함) cameraId=%s file=%s",
                    self.camera_id,
                    path.name,
                )
            elif index not in self._listed:
                logger.warning(
                    "닫히지 못한 녹화 파일이 남아 게시하지 않음 cameraId=%s file=%s",
                    self.camera_id,
                    path.name,
                )

    def _cleanup(self) -> None:
        """정상으로 끝났으면 쓰고 남은 목록·오류 로그를 지운다."""
        for path in (self._list_path, self._error_log):
            if path is not None:
                with contextlib.suppress(OSError):
                    path.unlink()

    def _error_tail(self) -> str:
        if self._error_log is None:
            return ""
        with contextlib.suppress(OSError):
            return self._error_log.read_text(errors="replace")[-500:]
        return ""

    def _report_exit(self) -> None:
        code = self._proc.poll()
        if code in (0, None):
            self._cleanup()
            return
        # 오류로 끝났으면 원인을 볼 수 있게 목록과 로그를 남긴다
        logger.error(
            "녹화 ffmpeg가 오류로 끝남 cameraId=%s code=%s: %s",
            self.camera_id,
            code,
            self._error_tail(),
        )


def _times(row: list[str]) -> tuple[float | None, float | None]:
    """segment list 한 줄(파일,시작,끝)의 시작·끝 초. 읽을 수 없으면 (None, None)."""
    try:
        start_s, end_s = float(row[1]), float(row[2])
    except (IndexError, ValueError):
        return None, None
    if not (0 <= start_s < end_s):  # 음수·역전·NaN은 믿지 않는다
        return None, None
    return start_s, end_s
