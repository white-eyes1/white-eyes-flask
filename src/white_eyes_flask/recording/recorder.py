"""FNC-S-008 라이브 녹화(1분 파일)

원본 프레임을 ffmpeg에 넘겨 시계 분 단위 mp4로 녹화하고, 분석 프레임의 박스를 같은 분 단위
좌표 파일로 저장한다. 분이 끝나 파일이 완성되면 게시기(RecordingPublisher)에 넘긴다.

호출 규칙: write_frame·add_boxes는 캠 작업 스레드 하나에서만, close는 그 스레드가 끝날 때 부른다.
"""

import contextlib
import csv
import logging
import os
import queue
import subprocess
import threading
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
from white_eyes_flask.recording.naming import MINUTE_MS, file_stem, minute_start_ms
from white_eyes_flask.recording.publisher import FinishedRecording, RecordingPublisher

logger = logging.getLogger(__name__)

# ffmpeg가 느려 쓰기가 막혀도 분석 스레드가 멈추지 않게, 가득 차면 프레임을 버린다
QUEUE_FRAMES = 60
POLL_SEC = 1.0
FLUSH_WAIT_SEC = 3.0
EXIT_WAIT_SEC = 4.0
KILL_WAIT_SEC = 1.0
# 프레임을 버린 로그는 첫 번째와 이 횟수마다 한 번만 남긴다
_DROP_LOG_EVERY = 100


class Recorder(Protocol):
    def close(self) -> None: ...


class RecorderUnavailable(RuntimeError):
    """녹화를 시작할 수 없다(ffmpeg 없음 등). 캠 작업은 ERROR로 처리한다."""


class LiveRecorder:
    def __init__(
        self,
        camera_id: int,
        settings: RecorderSettings,
        publisher: RecordingPublisher,
        popen: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        self.camera_id = camera_id
        self._settings = settings
        self._publisher = publisher
        self._popen = popen
        self._tmp_dir = Path(settings.tmp_dir)
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=QUEUE_FRAMES)
        self._closing = threading.Event()
        self._proc: Any = None
        self._error_log: Path | None = None
        self._list_path: Path | None = None
        self._list_offset = 0
        # 이미 게시기에 넘긴 파일 순번. segment list가 처음부터 다시 쓰여도 두 번 넘기지 않는다
        self._finished: set[int] = set()
        self._collect_lock = threading.Lock()
        self._box_lock = threading.Lock()
        self._boxes: list[tuple[int, list[TrackedBox]]] = []
        self._first_ms: int | None = None
        self._last_ms = 0
        self._dropped = 0
        self._pump: threading.Thread | None = None
        self._watch: threading.Thread | None = None
        self._closed = False

    # ---- 캠 작업 스레드가 부른다 ----

    def write_frame(self, frame, at: float) -> None:
        """원본 프레임 한 장(BGR, 높이×너비×3)과 그 프레임을 읽은 시각(epoch 초)."""
        at_ms = int(at * 1000)
        if self._proc is None:
            self._open(frame, at_ms)
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
        with self._box_lock:
            self._boxes.append((int(at * 1000), boxes))

    def close(self) -> None:
        """녹화를 마무리한다. 마지막 파일(1분 미만 포함)까지 게시기에 넘긴 뒤 돌아온다."""
        if self._proc is None or self._closed:
            return
        self._closed = True
        self._closing.set()
        if self._pump is not None:
            self._pump.join(FLUSH_WAIT_SEC)
            if self._pump.is_alive():
                logger.warning(
                    "녹화 프레임을 다 넘기지 못함 cameraId=%s", self.camera_id
                )
        self._stop_process()
        if self._watch is not None:
            self._watch.join(POLL_SEC + 1)
        self.poll_segments()
        self._report_exit()

    # ---- 시작·감시 ----

    def _open(self, frame, first_ms: int) -> None:
        height, width = frame.shape[:2]
        self._first_ms = first_ms
        token = str(first_ms)
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        self._list_path = segment_list_path(self._tmp_dir, self.camera_id, token)
        command = build_command(
            self._settings,
            width,
            height,
            segment_pattern(self._tmp_dir, self.camera_id, token),
            self._list_path,
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
                self.poll_segments()
            except Exception:
                logger.exception("녹화 파일 확인 중 예외 cameraId=%s", self.camera_id)

    # ---- 파일 완성 처리 ----

    def poll_segments(self) -> None:
        """segment list에 새로 올라온 파일(= 닫힌 파일)을 게시기에 넘긴다."""
        if self._list_path is None:
            return
        with self._collect_lock:
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
                if row:
                    self._finish_segment(row[0].strip())

    def _finish_segment(self, name: str) -> None:
        match = SEGMENT_NAME.search(name)
        if not match or self._first_ms is None:
            return
        index = int(match.group(1))
        if index in self._finished:
            return
        video = self._tmp_dir / Path(name).name
        started_ms, boundary_ms = self._bounds(index)
        with self._box_lock:
            # 이 파일의 시간대 박스만 쓴다. 시작보다 이른 박스는 앞 파일이 이미 나간 뒤 늦게
            # 도착한 것이라 버리고, 끝보다 늦은 박스는 다음 파일 몫으로 남긴다.
            inside = [i for i in self._boxes if started_ms <= i[0] < boundary_ms]
            self._boxes = [i for i in self._boxes if i[0] >= boundary_ms]
        # 녹화 중에 닫힌 파일은 분이 다 찬 것이다. 정지로 닫힌 마지막 파일은 마지막 프레임에서 끝난다.
        if not self._closing.is_set() or self._last_ms >= boundary_ms:
            ended_ms = boundary_ms
        else:
            ended_ms = max(self._last_ms, started_ms)
        stem = file_stem(self.camera_id, started_ms)
        box_path = self._tmp_dir / f"{stem}_{index:05d}.json.gz"
        frames = [((at - started_ms) / 1000, boxes) for at, boxes in inside]
        try:
            write_box_file(box_path, self._settings.analyze_fps, frames)
        except OSError:
            logger.exception(
                "좌표 파일을 쓰지 못함 cameraId=%s file=%s", self.camera_id, name
            )
            return
        self._publisher.submit(
            FinishedRecording(self.camera_id, started_ms, ended_ms, video, box_path)
        )
        self._finished.add(index)
        logger.info("녹화 파일 완성 cameraId=%s file=%s", self.camera_id, video.name)

    def _bounds(self, index: int) -> tuple[int, int]:
        """index번째 파일의 (시작, 끝 분 경계) 밀리초. 첫 파일은 가동 시각에서 시작한다."""
        assert self._first_ms is not None
        first_minute = minute_start_ms(self._first_ms)
        started = self._first_ms if index == 0 else first_minute + index * MINUTE_MS
        return started, first_minute + (index + 1) * MINUTE_MS

    # ---- 종료 ----

    def _stop_process(self) -> None:
        proc = self._proc
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

    def _cleanup(self) -> None:
        """정상으로 끝났으면 쓰고 남은 목록·오류 로그를 지운다."""
        for path in (self._list_path, self._error_log):
            if path is not None:
                with contextlib.suppress(OSError):
                    path.unlink()

    def _report_exit(self) -> None:
        code = self._proc.poll()
        if code in (0, None):
            self._cleanup()
            return
        # 오류로 끝났으면 원인을 볼 수 있게 목록과 로그를 남긴다
        tail = ""
        if self._error_log is not None:
            with contextlib.suppress(OSError):
                tail = self._error_log.read_text(errors="replace")[-500:]
        logger.error(
            "녹화 ffmpeg가 오류로 끝남 cameraId=%s code=%s: %s",
            self.camera_id,
            code,
            tail,
        )
