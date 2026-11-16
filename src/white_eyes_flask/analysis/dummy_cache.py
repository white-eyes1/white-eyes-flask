"""더미(업로드 영상) 분석 결과 캐시.

준비 단계(video_analysis)가 영상을 끝까지 분석해 두 파일로 저장하고, 재생 단계가 이를 읽어 1초씩 내보낸다.
저장은 임시 파일에 쓴 뒤 한꺼번에 교체하는 방식이라, 중간에 실패해도 반쪽짜리 캐시가 남지 않는다.
"""

import contextlib
import gzip
import json
import math
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from white_eyes_flask.analysis.result import TrackedBox

# 캐시 파일 형식 버전(형식을 바꿀 때 올려서 구분한다)
CACHE_VERSION = 1
# 영상별 잠금을 기다리는 동안 stop 신호를 확인하는 간격(초)
LOCK_POLL_SEC = 0.2
# 좌표 파일이 놓이는 공유 폴더 안의 상대 경로
BOXES_DIR = "dummy/boxes"
# 좌표를 소수 몇 자리까지 저장할지(파일 크기를 줄이려고 반올림한다)
_COORD_DIGITS = 4
# 방향 이름. 캐시 검증 때 in/out에 이 네 방향 값이 다 있어야 한다
_DIRECTIONS = ("N", "E", "S", "W")


class CacheInvalidError(Exception):
    """캐시 파일이 없거나 깨져서 쓸 수 없음."""


@dataclass(frozen=True)
class CachedResult:
    """읽어서 검증한 더미 분석 캐시. 재생(FNC-S-010)이 쓴다."""

    seconds: list[dict[str, Any]]
    duration_sec: float  # 영상 길이 L초


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
        # _locks 사전 자체를 여러 스레드가 동시에 고치지 않게 하는 잠금
        self._guard = threading.Lock()
        # 영상 이름별 준비 잠금
        self._locks: dict[str, threading.Lock] = {}

    @staticmethod
    def name(source_uri: str) -> str:
        """영상 경로에서 폴더와 확장자를 뺀 파일 이름. 캐시 파일 이름의 기준이 된다."""
        return PurePosixPath(source_uri).stem

    def result_path(self, source_uri: str) -> Path:
        """초별 결과 캐시 JSON의 경로(분석 서버 로컬)."""
        return self._cache_dir / f"{self.name(source_uri)}.json"

    def video_path(self, source_uri: str) -> Path:
        """원본 영상 파일(공유 폴더 기준 source_uri). 미리보기가 프레임을 읽는다."""
        return self._media_root / source_uri

    def boxes_relative_path(self, source_uri: str) -> str:
        """좌표 파일의 공유 폴더 기준 상대 경로. 다른 서버가 같은 파일을 찾을 때 쓰는 표기다."""
        return f"{BOXES_DIR}/{self.name(source_uri)}.json.gz"

    def boxes_path(self, source_uri: str) -> Path:
        """좌표 파일(gzip JSON)의 실제 경로."""
        return self._media_root / self.boxes_relative_path(source_uri)

    def has(self, source_uri: str) -> bool:
        """캐시가 준비되어 있는지. 결과 JSON과 좌표 파일이 둘 다 있어야 한다(내용 검증은 load가 한다)."""
        return (
            self.result_path(source_uri).is_file()
            and self.boxes_path(source_uri).is_file()
        )

    def load(self, source_uri: str) -> CachedResult:
        """캐시 JSON을 읽고 검증한다. 없거나 깨졌으면 CacheInvalidError."""
        path = self.result_path(source_uri)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            # 파일이 없거나(OSError) JSON이 깨진 경우(ValueError)를 한 가지 오류로 묶어 호출자가 다루기 쉽게 한다
            raise CacheInvalidError(f"캐시를 읽을 수 없음: {path.name}") from error
        # 읽기만 성공했을 뿐이라 재생이 믿고 쓸 모양인지 한 번 더 검증한다
        return _validate(data, path.name)

    def invalidate(self, source_uri: str) -> None:
        """깨진 캐시를 지워서 다음 시작 때 다시 준비하게 한다."""
        # 둘 중 하나만 있어도 has()가 어긋나므로 둘 다 지운다. 이미 없으면 무시한다
        for path in (self.result_path(source_uri), self.boxes_path(source_uri)):
            with contextlib.suppress(FileNotFoundError):
                path.unlink()

    @contextlib.contextmanager
    def lock(self, source_uri: str, stop: threading.Event) -> Iterator[bool]:
        """영상 하나를 한 번에 한 캠만 준비하도록 잠근다.

        다른 캠이 준비 중이면 끝날 때까지 기다린다. 기다리는 동안 stop이 서면
        기다림을 포기하고 False를 돌려준다(잠금을 얻으면 True).
        WSGI 워커 1개 전제라 프로세스 안의 잠금으로 충분하다.
        """
        name = self.name(source_uri)
        # 같은 영상 이름이면 항상 같은 잠금 객체를 받도록, 사전 접근만 짧게 보호한다
        with self._guard:
            lock = self._locks.setdefault(name, threading.Lock())
        acquired = False
        # 오래 막히지 않도록 짧게 기다리고 돌아와 stop을 확인한다(캠을 끄면 곧바로 빠져나온다)
        while not stop.is_set():
            if lock.acquire(timeout=LOCK_POLL_SEC):
                acquired = True
                break
        try:
            yield acquired
        finally:
            # 잠금을 얻은 경우에만 풀어야 한다(얻지 못했는데 풀면 오류)
            if acquired:
                lock.release()

    def writer(self, source_uri: str, fps: int) -> "DummyResultWriter":
        """캐시를 새로 쓰는 쓰기 객체를 만든다. with 문으로 쓰고 끝에 commit()을 불러야 반영된다."""
        return DummyResultWriter(self, source_uri, fps)


class DummyResultWriter:
    """임시 파일에 쓰다가 commit()에서 한꺼번에 올린다. commit하지 않고 나가면 지운다.

    좌표 파일은 프레임마다 바로 gzip으로 흘려 써서 긴 영상도 메모리에 쌓지 않는다.
    """

    def __init__(self, cache: DummyCache, source_uri: str, fps: int) -> None:
        self._cache = cache
        self._source_uri = source_uri
        self._fps = fps
        # 임시 파일 이름에 붙이는 무작위 값. 같은 영상을 동시에 쓰더라도 임시 파일끼리 부딪치지 않는다
        token = uuid.uuid4().hex[:8]
        # 최종 위치(commit 때 여기로 교체)와 같은 폴더의 임시 위치.
        # 같은 폴더여야 os.replace가 한 번에(원자적으로) 바꿀 수 있다
        self._boxes_final = cache.boxes_path(source_uri)
        self._result_final = cache.result_path(source_uri)
        self._boxes_tmp = self._boxes_final.with_name(
            f"{self._boxes_final.name}.{token}.tmp"
        )
        self._result_tmp = self._result_final.with_name(
            f"{self._result_final.name}.{token}.tmp"
        )
        # 좌표 임시 파일에 열어 둔 gzip 스트림(열기 전·닫은 뒤에는 None)
        self._stream: Any = None
        # 첫 프레임 앞에는 쉼표를 붙이지 않으려는 표시(JSON 배열 문법)
        self._first_frame = True
        # commit이 끝났는지. 끝나지 않고 with를 나가면 임시 파일을 지운다
        self._committed = False

    def __enter__(self) -> "DummyResultWriter":
        """폴더를 만들고 좌표 임시 파일을 열어 JSON 머리부분({"version":..,"frames":[)을 먼저 쓴다."""
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
        """commit 없이 나가면(오류·중단) 임시 파일을 지운다. False를 돌려줘 원래 예외는 그대로 던져진다."""
        if not self._committed:
            self._abort()
        return False

    def add_frame(self, t: float, boxes: list[TrackedBox]) -> None:
        """t는 영상 시작부터의 초. 박스는 [추적 번호, x, y, w, h]로 적는다."""
        # 박스마다 한 줄. 좌표는 반올림해 파일 크기를 줄인다
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
        # 공백 없는 압축 JSON으로 한 프레임을 만든다
        frame = json.dumps(
            {"t": round(t, _COORD_DIGITS), "b": rows}, separators=(",", ":")
        )
        # frames 배열의 원소 사이에만 쉼표를 넣는다
        self._stream.write(frame if self._first_frame else "," + frame)
        self._first_frame = False

    def commit(
        self, seconds: list[dict[str, Any]], duration_sec: float | None = None
    ) -> None:
        """좌표 파일을 닫고 초별 결과를 써서, 두 파일을 최종 위치로 올린다.

        seconds는 초별 결과 목록, duration_sec는 영상 길이(생략하면 초 개수로 본다).
        """
        # 열어 둔 JSON 배열·객체를 닫고 gzip 스트림을 마무리한다
        self._stream.write("]}")
        self._stream.close()
        self._stream = None
        result: dict[str, Any] = {
            "version": CACHE_VERSION,
            "fps": self._fps,
            "seconds": seconds,
        }
        # 영상 길이는 알 때만 적는다. 없으면 읽을 때 초 개수를 길이로 쓴다(_validate 참고)
        if duration_sec is not None:
            result["durationSec"] = duration_sec
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
        # 여기까지 오면 두 파일이 모두 올라간 것이다
        self._committed = True

    def _abort(self) -> None:
        """열린 스트림을 닫고 임시 파일을 지운다. 정리 중의 오류는 무시한다(원래 오류를 가리지 않으려고)."""
        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.close()
            self._stream = None
        for path in (self._boxes_tmp, self._result_tmp):
            with contextlib.suppress(OSError):
                path.unlink()


def _is_number(value: Any) -> bool:
    """숫자(int·float)인지. bool은 int의 하위 클래스라 True/False를 숫자로 잘못 받지 않게 따로 뺀다."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate(data: Any, name: str) -> CachedResult:
    """재생이 믿고 쓸 수 있는 모양인지 확인한다(0부터 이어진 초, 숫자 필드, 길이)."""
    if not isinstance(data, dict) or not isinstance(data.get("seconds"), list):
        raise CacheInvalidError(f"캐시 형식이 맞지 않음: {name}")
    seconds = data["seconds"]
    if not seconds:
        raise CacheInvalidError(f"캐시에 초별 결과가 없음: {name}")
    for index, entry in enumerate(seconds):
        # 재생은 목록 순서를 초 번호로 쓰므로, 각 항목의 sec이 0, 1, 2... 순서와 같아야 한다
        if not isinstance(entry, dict) or entry.get("sec") != index:
            raise CacheInvalidError(f"초 번호가 0부터 이어지지 않음: {name}")
        count = entry.get("personCount")
        if not _is_number(count) or count < 0:
            raise CacheInvalidError(f"personCount가 숫자가 아님: {name}")
        # 방향별 유입(in)·유출(out)은 N/E/S/W 네 값이 모두 숫자여야 한다
        for key in ("in", "out"):
            counts = entry.get(key)
            if not isinstance(counts, dict) or not all(
                _is_number(counts.get(d)) for d in _DIRECTIONS
            ):
                raise CacheInvalidError(f"{key} 형식이 맞지 않음: {name}")
        # 체류 시간 목록: 숫자만 든 리스트여야 한다(비어 있어도 된다)
        dwell = entry.get("dwellSec")
        if not isinstance(dwell, list) or not all(_is_number(v) for v in dwell):
            raise CacheInvalidError(f"dwellSec 형식이 맞지 않음: {name}")
    # 영상 길이가 없으면 초 개수를 길이로 본다
    duration = data.get("durationSec", float(len(seconds)))
    if not _is_number(duration) or not math.isfinite(duration):
        raise CacheInvalidError(f"영상 길이가 숫자가 아님: {name}")
    # 길이는 양수이고 초별 결과 개수를 넘을 수 없다(마지막 초가 덜 찬 경우 개수보다 작을 수는 있다)
    if duration <= 0 or duration > len(seconds):
        raise CacheInvalidError(f"영상 길이가 초별 결과와 맞지 않음: {name}")
    return CachedResult(seconds=seconds, duration_sec=float(duration))
