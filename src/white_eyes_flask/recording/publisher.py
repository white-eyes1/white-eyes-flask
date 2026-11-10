"""FNC-S-008 4~6번: 완성된 녹화 파일을 공유 폴더로 옮기고 관리자 앱에 등록한다.

캠 작업 스레드나 정지 요청을 막지 않게 별도 스레드 하나가 맡는다. 실패하면 1분 뒤에 다시 한다.
공유 폴더로 옮기지 못하면 로컬에 남기고 계속 다시 옮기고(명세: 다음 분에 다시 옮김), 등록은
MAX_ATTEMPTS번 넘게 실패하면 로그만 남긴다(파일은 공유 폴더에 남는다).
"""

import contextlib
import logging
import os
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from white_eyes_flask.camera.recording_registrar import (
    RecordingRegistrar,
    RegisterResult,
)
from white_eyes_flask.recording.naming import file_stem, relative_dir

logger = logging.getLogger(__name__)

RETRY_SEC = 60
MAX_ATTEMPTS = 10


@dataclass(frozen=True)
class FinishedRecording:
    """로컬 임시 폴더에 완성된 1분 파일 하나(영상 + 좌표)."""

    camera_id: int
    started_ms: int
    ended_ms: int
    video: Path
    boxes: Path
    # 공유 폴더의 파일 이름·날짜 폴더에 쓰는 시각(가까운 분, 첫 파일은 가동 시각이 속한 분). 없으면 started_ms
    name_ms: int | None = None


@dataclass
class _Job:
    recording: FinishedRecording
    due: float = 0.0
    move_failures: int = 0
    register_failures: int = 0
    # 옮긴 뒤 등록에 쓰는 공유 폴더 기준 경로. None이면 아직 못 옮김
    video_path: str | None = None
    box_path: str | None = None


class RecordingPublisher:
    def __init__(
        self,
        media_root: str,
        registrar: RecordingRegistrar,
        retry_sec: float = RETRY_SEC,
        max_attempts: int = MAX_ATTEMPTS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._media_root = Path(media_root)
        self._registrar = registrar
        self._retry_sec = retry_sec
        self._max_attempts = max_attempts
        self._clock = clock
        self._lock = threading.Lock()
        self._jobs: list[_Job] = []
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="recording-publisher", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """남은 일은 하지 않고 끝낸다. 옮기지 못한 파일은 로컬에 남는다."""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
            # 시간 안에 안 끝났으면 참조를 지우지 않는다(지우면 start()가 스레드를 하나 더 만든다)
            if not self._thread.is_alive():
                self._thread = None

    def submit(self, recording: FinishedRecording) -> None:
        with self._lock:
            self._jobs.append(_Job(recording))
        self._wake.set()

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._jobs)

    def run_due(self) -> float | None:
        """때가 된 일을 한 번씩 처리하고, 다음 일까지 남은 초(없으면 None)를 돌려준다."""
        now = self._clock()
        with self._lock:
            due = [job for job in self._jobs if job.due <= now]
        for job in due:
            # 한 작업의 예외가 뒤 작업을 막지 않게 작업별로 감싼다
            try:
                finished = self._step(job)
            except Exception:
                logger.exception(
                    "녹화 게시 중 예외 cameraId=%s", job.recording.camera_id
                )
                finished = self._give_up_or_retry(job, "register_failures", "녹화 게시")
            if finished:
                with self._lock:
                    self._jobs.remove(job)
        with self._lock:
            if not self._jobs:
                return None
            return max(0.0, min(job.due for job in self._jobs) - self._clock())

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                wait = self.run_due()
            except Exception:
                logger.exception("녹화 게시 중 예외")
                wait = self._retry_sec
            self._wake.wait(wait)
            self._wake.clear()

    def _step(self, job: _Job) -> bool:
        """일을 한 단계 진행한다. 끝났으면(성공이든 포기든) True."""
        recording = job.recording
        if job.video_path is None:
            if not self._move(job):
                # 공유 폴더가 돌아올 때까지 로컬에 두고 계속 다시 옮긴다(횟수 제한 없음)
                job.move_failures += 1
                job.due = self._clock() + self._retry_sec
                return False
        result = self._registrar.register(
            recording.camera_id,
            recording.started_ms,
            job.video_path,
            job.box_path,
            ended_ms=recording.ended_ms,
        )
        if result == RegisterResult.OK:
            logger.info(
                "녹화 등록 cameraId=%s file=%s",
                recording.camera_id,
                job.video_path,
            )
            return True
        if result == RegisterResult.REJECTED:
            logger.error(
                "녹화 등록 거절로 포기 cameraId=%s file=%s",
                recording.camera_id,
                job.video_path,
            )
            return True
        return self._give_up_or_retry(job, "register_failures", "녹화 등록")

    def _give_up_or_retry(self, job: _Job, counter: str, what: str) -> bool:
        failures = getattr(job, counter) + 1
        setattr(job, counter, failures)
        if failures >= self._max_attempts:
            logger.error(
                "%s을(를) %s번 실패해 포기 cameraId=%s file=%s",
                what,
                failures,
                job.recording.camera_id,
                job.video_path or job.recording.video.name,
            )
            return True
        job.due = self._clock() + self._retry_sec
        return False

    def _move(self, job: _Job) -> bool:
        """영상과 좌표를 공유 폴더로 옮긴다. 둘 다 올라간 뒤에만 로컬 파일을 지운다."""
        recording = job.recording
        name_ms = (
            recording.started_ms if recording.name_ms is None else recording.name_ms
        )
        folder = relative_dir(recording.camera_id, name_ms)
        stem = file_stem(recording.camera_id, name_ms)
        try:
            target = self._media_root / folder
            target.mkdir(parents=True, exist_ok=True)
            stem = self._free_stem(target, stem)
            video_dest = target / f"{stem}.mp4"
            box_dest = target / f"{stem}.json.gz"
            self._copy_atomic(recording.video, video_dest)
            try:
                self._copy_atomic(recording.boxes, box_dest)
            except OSError:
                # 영상만 올라간 채로 두지 않는다
                with contextlib.suppress(OSError):
                    video_dest.unlink()
                raise
        except OSError as error:
            logger.warning(
                "녹화 파일을 공유 폴더로 옮기지 못함 cameraId=%s: %s",
                recording.camera_id,
                error,
            )
            return False
        job.video_path = f"{folder}/{stem}.mp4"
        job.box_path = f"{folder}/{stem}.json.gz"
        for local in (recording.video, recording.boxes):
            with contextlib.suppress(OSError):
                local.unlink()
        return True

    @staticmethod
    def _free_stem(folder: Path, stem: str) -> str:
        """같은 분에 캠을 다시 시작하면 이름이 겹친다. 있던 파일을 덮지 않게 번호를 붙인다."""
        candidate, number = stem, 1
        while (folder / f"{candidate}.mp4").exists() or (
            folder / f"{candidate}.json.gz"
        ).exists():
            number += 1
            candidate = f"{stem}_{number}"
        return candidate

    @staticmethod
    def _copy_atomic(source: Path, dest: Path) -> None:
        """공유 폴더 안에서 임시 이름으로 다 복사한 뒤 이름을 바꿔, 읽는 쪽이 반쪽 파일을 보지 않게 한다."""
        temp = dest.with_name(dest.name + ".part")
        try:
            shutil.copyfile(source, temp)
            os.replace(temp, dest)
        except OSError:
            with contextlib.suppress(OSError):
                temp.unlink()
            raise
