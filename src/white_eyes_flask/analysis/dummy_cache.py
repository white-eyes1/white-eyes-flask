import contextlib
import gzip
import json
import os
import threading
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from white_eyes_flask.analysis.result import TrackedBox

CACHE_VERSION = 1
LOCK_POLL_SEC = 0.2
BOXES_DIR = "dummy/boxes"
_COORD_DIGITS = 4


class DummyCache:
    """더미 분석 결과(FNC-S-009)를 두는 곳.

    - 초별 결과 캐시: {cache_dir}/{파일명}.json (분석 서버 로컬)
    - 좌표 파일: {media_root}/dummy/boxes/{파일명}.json.gz (공유 폴더)
    파일명은 영상 파일 이름(확장자 제외)이라 같은 영상을 쓰는 캠끼리 공유한다.
    두 파일이 모두 있어야 캐시가 있는 것이고, 캐시 JSON은 좌표 파일을 먼저 올린
    다음 마지막에 올려 "준비 끝" 표식으로 쓴다.
    """

    def __init__(self, cache_dir: str | Path, media_root: str | Path) -> None:
        self._cache_dir = Path(cache_dir)
        self._media_root = Path(media_root)
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    @staticmethod
    def name(source_uri: str) -> str:
        return PurePosixPath(source_uri).stem

    def result_path(self, source_uri: str) -> Path:
        return self._cache_dir / f"{self.name(source_uri)}.json"

    def boxes_relative_path(self, source_uri: str) -> str:
        return f"{BOXES_DIR}/{self.name(source_uri)}.json.gz"

    def boxes_path(self, source_uri: str) -> Path:
        return self._media_root / self.boxes_relative_path(source_uri)

    def has(self, source_uri: str) -> bool:
        return (
            self.result_path(source_uri).is_file()
            and self.boxes_path(source_uri).is_file()
        )

    @contextlib.contextmanager
    def lock(self, source_uri: str, stop: threading.Event) -> Iterator[bool]:
        """영상 하나를 한 번에 한 캠만 준비하도록 잠근다.

        다른 캠이 준비 중이면 끝날 때까지 기다린다. 기다리는 동안 stop이 서면
        기다림을 포기하고 False를 돌려준다(잠금을 얻으면 True).
        WSGI 워커 1개 전제라 프로세스 안의 잠금으로 충분하다.
        """
        name = self.name(source_uri)
        with self._guard:
            lock = self._locks.setdefault(name, threading.Lock())
        acquired = False
        while not stop.is_set():
            if lock.acquire(timeout=LOCK_POLL_SEC):
                acquired = True
                break
        try:
            yield acquired
        finally:
            if acquired:
                lock.release()

    def writer(self, source_uri: str, fps: int) -> "DummyResultWriter":
        return DummyResultWriter(self, source_uri, fps)


class DummyResultWriter:
    """임시 파일에 쓰다가 commit()에서 한꺼번에 올린다. commit하지 않고 나가면 지운다.

    좌표 파일은 프레임마다 바로 gzip으로 흘려 써서 긴 영상도 메모리에 쌓지 않는다.
    """

    def __init__(self, cache: DummyCache, source_uri: str, fps: int) -> None:
        self._cache = cache
        self._source_uri = source_uri
        self._fps = fps
        token = uuid.uuid4().hex[:8]
        self._boxes_final = cache.boxes_path(source_uri)
        self._result_final = cache.result_path(source_uri)
        self._boxes_tmp = self._boxes_final.with_name(
            f"{self._boxes_final.name}.{token}.tmp"
        )
        self._result_tmp = self._result_final.with_name(
            f"{self._result_final.name}.{token}.tmp"
        )
        self._stream: Any = None
        self._first_frame = True
        self._committed = False

    def __enter__(self) -> "DummyResultWriter":
        self._boxes_final.parent.mkdir(parents=True, exist_ok=True)
        self._result_final.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._stream = gzip.open(
                self._boxes_tmp, "wt", encoding="utf-8", compresslevel=6
            )
            self._stream.write(
                f'{{"version":{CACHE_VERSION},"fps":{self._fps},"frames":['
            )
        except BaseException:
            self._abort()  # __enter__가 실패하면 __exit__가 불리지 않는다
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if not self._committed:
            self._abort()
        return False

    def add_frame(self, t: float, boxes: list[TrackedBox]) -> None:
        """t는 영상 시작부터의 초. 박스는 [추적 번호, x, y, w, h]로 적는다."""
        rows = [
            [
                box.track_id,
                round(box.x, _COORD_DIGITS),
                round(box.y, _COORD_DIGITS),
                round(box.w, _COORD_DIGITS),
                round(box.h, _COORD_DIGITS),
            ]
            for box in boxes
        ]
        frame = json.dumps(
            {"t": round(t, _COORD_DIGITS), "b": rows}, separators=(",", ":")
        )
        self._stream.write(frame if self._first_frame else "," + frame)
        self._first_frame = False

    def commit(self, seconds: list[dict[str, Any]]) -> None:
        self._stream.write("]}")
        self._stream.close()
        self._stream = None
        result = {"version": CACHE_VERSION, "fps": self._fps, "seconds": seconds}
        self._result_tmp.write_text(
            json.dumps(result, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        # 옛 캐시 JSON이 새 좌표 파일과 짝으로 보이지 않게 먼저 치운다.
        with contextlib.suppress(FileNotFoundError):
            self._result_final.unlink()
        # 좌표 파일을 먼저 올리고 캐시 JSON을 마지막에 올린다(캐시 JSON이 완료 표식).
        os.replace(self._boxes_tmp, self._boxes_final)
        try:
            os.replace(self._result_tmp, self._result_final)
        except BaseException:
            # 반쪽만 올라간 채로 두지 않는다
            with contextlib.suppress(OSError):
                self._boxes_final.unlink()
            raise
        self._committed = True

    def _abort(self) -> None:
        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.close()
            self._stream = None
        for path in (self._boxes_tmp, self._result_tmp):
            with contextlib.suppress(OSError):
                path.unlink()
