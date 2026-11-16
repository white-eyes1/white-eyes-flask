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

# 일이 실패했을 때 다음 시도까지 기다리는 시간(초).
RETRY_SEC = 60
# 등록 실패를 허용하는 횟수. 이 횟수에 이르면 포기한다(공유 폴더로 옮기기는 횟수 제한 없이 계속 시도한다).
MAX_ATTEMPTS = 10


@dataclass(frozen=True)
class FinishedRecording:
    """로컬 임시 폴더에 완성된 1분 파일 하나(영상 + 좌표)."""

    camera_id: int
    # 영상의 첫 프레임 시각(epoch 밀리초). 관리자 앱에 등록하는 startedAt이 된다.
    started_ms: int
    # 영상의 끝 시각(epoch 밀리초). 등록하는 endedAt이 된다.
    ended_ms: int
    # 로컬 임시 폴더에 있는 mp4 파일.
    video: Path
    # 로컬 임시 폴더에 있는 좌표 파일(.json.gz).
    boxes: Path
    # 공유 폴더의 파일 이름·날짜 폴더에 쓰는 시각(가까운 분, 첫 파일은 가동 시각이 속한 분). 없으면 started_ms
    name_ms: int | None = None


@dataclass
class _Job:
    """게시 대기 중인 일 한 건과 그 진행 상태(다음 시도 시각, 실패 횟수, 옮긴 뒤의 경로)."""

    recording: FinishedRecording
    # 이 시각(clock 기준) 이후에 처리한다. 0이면 바로 처리한다.
    due: float = 0.0
    # 공유 폴더로 옮기기에 실패한 횟수(기록용. 포기 기준이 아니다).
    move_failures: int = 0
    # 등록(또는 게시 중 예외)에 실패한 횟수. max_attempts에 이르면 포기한다.
    register_failures: int = 0
    # 옮긴 뒤 등록에 쓰는 공유 폴더 기준 경로. None이면 아직 못 옮김
    video_path: str | None = None
    box_path: str | None = None


class RecordingPublisher:
    """완성된 녹화 파일을 공유 폴더로 옮기고 관리자 앱에 등록하는 게시기.

    스레드 하나가 일(_Job) 목록을 돌며 때가 된 것을 처리한다. 일은 submit으로 받는다.
    """

    def __init__(
        self,
        media_root: str,
        registrar: RecordingRegistrar,
        retry_sec: float = RETRY_SEC,
        max_attempts: int = MAX_ATTEMPTS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """게시기를 만든다(스레드는 start()에서 시작).

        :param media_root: 공유 폴더 루트
        :param registrar: 관리자 앱에 녹화를 등록하는 객체
        :param retry_sec: 실패한 일을 다시 시도하기까지의 간격(초)
        :param max_attempts: 등록 실패를 포기하기까지의 횟수
        :param clock: 현재 시각(초). 시험에서 바꿔 끼운다
        """
        self._media_root = Path(media_root)
        self._registrar = registrar
        self._retry_sec = retry_sec
        self._max_attempts = max_attempts
        self._clock = clock
        self._lock = threading.Lock()
        # 아직 끝나지 않은 일 목록. 락으로 지킨다.
        self._jobs: list[_Job] = []
        # 새 일이 오거나 정지할 때 쉬고 있는 스레드를 깨우는 신호.
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """게시 스레드를 시작한다. 이미 시작했으면 아무것도 하지 않는다."""
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
        """완성된 파일 하나를 게시 대기열에 올리고 스레드를 깨운다. 기다리지 않는다."""
        with self._lock:
            self._jobs.append(_Job(recording))
        self._wake.set()

    @property
    def pending(self) -> int:
        """아직 끝나지 않은 일의 수."""
        with self._lock:
            return len(self._jobs)

    def run_due(self) -> float | None:
        """때가 된 일을 한 번씩 처리하고, 다음 일까지 남은 초(없으면 None)를 돌려준다."""
        now = self._clock()
        with self._lock:
            # 락 안에서는 처리할 대상만 고르고, 느린 일(파일 복사·HTTP)은 락 밖에서 한다.
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
            # 끝난 일(성공이든 포기든)은 목록에서 뺀다. 끝나지 않은 일은 due가 미뤄진 채 남는다.
            if finished:
                with self._lock:
                    self._jobs.remove(job)
        with self._lock:
            if not self._jobs:
                return None
            return max(0.0, min(job.due for job in self._jobs) - self._clock())

    def _run(self) -> None:
        """게시 스레드 본체. 때가 된 일을 처리하고 다음 일까지 쉬는 것을 정지할 때까지 반복한다."""
        while not self._stop.is_set():
            try:
                wait = self.run_due()
            except Exception:
                logger.exception("녹화 게시 중 예외")
                # 예상 밖 예외에도 스레드가 죽지 않게 한다. 한 번 쉰 뒤 다시 돈다.
                wait = self._retry_sec
            # 다음 일 때까지 쉬다가, submit이나 stop이 깨우면 바로 일어난다.
            self._wake.wait(wait)
            self._wake.clear()

    def _step(self, job: _Job) -> bool:
        """일을 한 단계 진행한다. 끝났으면(성공이든 포기든) True."""
        recording = job.recording
        # 아직 공유 폴더로 옮기지 못한 일이면 먼저 옮긴다. 옮긴 일은 다음 시도에서 이 단계를 건너뛴다.
        if job.video_path is None:
            if not self._move(job):
                # 공유 폴더가 돌아올 때까지 로컬에 두고 계속 다시 옮긴다(횟수 제한 없음)
                job.move_failures += 1
                job.due = self._clock() + self._retry_sec
                return False
        # 옮겼으면 관리자 앱에 등록한다. 공유 폴더 기준 경로를 보낸다.
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
        """실패를 기록하고 포기할지 정한다.

        counter는 _Job의 실패 횟수 필드 이름이다. 최대 시도에 이르면 True(포기), 아니면 retry_sec 뒤로 미루고 False를 돌려준다.
        """
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
        # 공유 폴더의 폴더·파일 이름에 쓰는 시각. 따로 정해 두지 않았으면 영상 시작 시각을 쓴다.
        name_ms = (
            recording.started_ms if recording.name_ms is None else recording.name_ms
        )
        folder = relative_dir(recording.camera_id, name_ms)
        stem = file_stem(recording.camera_id, name_ms)
        try:
            target = self._media_root / folder
            target.mkdir(parents=True, exist_ok=True)
            # 같은 이름이 이미 있으면 번호를 붙여 덮어쓰지 않는다.
            stem = self._free_stem(target, stem)
            video_dest = target / f"{stem}.mp4"
            box_dest = target / f"{stem}.json.gz"
            # 영상 먼저, 좌표 파일은 그다음에 올린다. 좌표 파일이 실패하면 올라간 영상도 지운다(아래).
            self._copy_atomic(recording.video, video_dest)
            try:
                self._copy_atomic(recording.boxes, box_dest)
            except OSError:
                # 영상만 올라간 채로 두지 않는다
                with contextlib.suppress(OSError):
                    video_dest.unlink()
                raise
        # 공유 폴더가 안 열리는 등의 오류다. 로컬 파일은 그대로 두고 나중에 다시 옮긴다.
        except OSError as error:
            logger.warning(
                "녹화 파일을 공유 폴더로 옮기지 못함 cameraId=%s: %s",
                recording.camera_id,
                error,
            )
            return False
        # 둘 다 올라간 뒤에만 경로를 기록하고 로컬 파일을 지운다. 경로가 기록되면 이후 시도는 옮기기를 건너뛰고 등록만 한다.
        job.video_path = f"{folder}/{stem}.mp4"
        job.box_path = f"{folder}/{stem}.json.gz"
        for local in (recording.video, recording.boxes):
            with contextlib.suppress(OSError):
                local.unlink()
        return True

    @staticmethod
    def _free_stem(folder: Path, stem: str) -> str:
        """같은 분에 캠을 다시 시작하면 이름이 겹친다. 있던 파일을 덮지 않게 번호를 붙인다."""
        # 처음엔 원래 이름, 겹치면 _2, _3 … 순으로 비어 있는 이름을 찾는다.
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
        # .part 임시 이름으로 복사해, 읽는 쪽이 아직 복사 중인 파일을 집지 않게 한다.
        temp = dest.with_name(dest.name + ".part")
        try:
            shutil.copyfile(source, temp)
            # 다 복사한 뒤 이름을 바꿔 한 번에 나타나게 한다.
            os.replace(temp, dest)
        # 실패하면 반쯤 복사된 임시 파일을 지우고 예외를 다시 올린다.
        except OSError:
            with contextlib.suppress(OSError):
                temp.unlink()
            raise
