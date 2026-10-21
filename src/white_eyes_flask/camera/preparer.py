import logging
import threading
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Protocol

from white_eyes_flask.analysis.detector import Detector
from white_eyes_flask.analysis.detector_provider import DetectorUnavailableError
from white_eyes_flask.analysis.dummy_cache import DummyCache
from white_eyes_flask.analysis.video_analysis import (
    AnalysisStopped,
    VideoReadError,
    analyze_video,
)

logger = logging.getLogger(__name__)

ERROR_VIDEO = "영상 파일을 읽을 수 없음"
ERROR_MODEL = "분석 모델을 불러올 수 없음"
ERROR_UNEXPECTED = "분석 중 오류가 발생함"


class PrepareOutcome(StrEnum):
    DONE = "DONE"  # 준비 끝(캐시 있음)
    STOPPED = "STOPPED"  # 정지 요청으로 취소
    FAILED = "FAILED"  # 오류(ERROR 보고함)


class Preparer(Protocol):
    def has_cache(self, source_uri: str) -> bool: ...

    def prepare(
        self, camera_id: int, source_uri: str, stop: threading.Event
    ) -> PrepareOutcome: ...


class NullPreparer:
    """준비를 하지 않는다. 캐시는 항상 없고, 정지될 때까지 준비 중으로 둔다."""

    def has_cache(self, source_uri: str) -> bool:
        return False

    def prepare(
        self, camera_id: int, source_uri: str, stop: threading.Event
    ) -> PrepareOutcome:
        stop.wait()
        return PrepareOutcome.STOPPED


class DummyPreparer:
    """더미 영상 분석 준비(FNC-S-009): 분석 -> 캐시·좌표 파일 -> 상태 보고."""

    def __init__(
        self,
        media_root: str | Path,
        cache: DummyCache,
        detector_provider: Callable[[], Detector],
        reporter: Any,
        *,
        fps: int,
        conf_thres: float,
        edge_margin: float,
    ) -> None:
        self._media_root = Path(media_root)
        self._cache = cache
        self._detector_provider = detector_provider
        self._reporter = reporter
        self._fps = fps
        self._conf_thres = conf_thres
        self._edge_margin = edge_margin

    def has_cache(self, source_uri: str) -> bool:
        return self._cache.has(source_uri)

    def prepare(
        self, camera_id: int, source_uri: str, stop: threading.Event
    ) -> PrepareOutcome:
        if stop.is_set():
            return PrepareOutcome.STOPPED
        if self._cache.has(source_uri):
            self._report(camera_id, "RUNNING", progress=100)
            return PrepareOutcome.DONE
        self._report(camera_id, "PREPARING", progress=0)
        try:
            # 같은 영상을 준비 중인 다른 캠이 있으면 끝날 때까지 기다린다
            with self._cache.lock(source_uri, stop) as acquired:
                if not acquired or stop.is_set():
                    return PrepareOutcome.STOPPED
                # 기다리는 동안 다른 캠이 이미 만들었을 수 있다
                if not self._cache.has(source_uri):
                    self._analyze(camera_id, source_uri, stop)
        except AnalysisStopped:
            return PrepareOutcome.STOPPED
        except VideoReadError:
            logger.exception("영상 읽기 실패 cameraId=%s uri=%s", camera_id, source_uri)
            return self._fail(camera_id, ERROR_VIDEO)
        except DetectorUnavailableError:
            logger.exception("탐지 모델 사용 불가 cameraId=%s", camera_id)
            return self._fail(camera_id, ERROR_MODEL)
        except Exception:
            logger.exception("더미 분석 준비 실패 cameraId=%s", camera_id)
            return self._fail(camera_id, ERROR_UNEXPECTED)
        if stop.is_set():
            # 파일을 올린 뒤에 정지가 들어왔다. 캐시는 그대로 쓸 수 있지만 RUNNING은 보고하지 않는다.
            return PrepareOutcome.STOPPED
        # 파일을 올린 뒤에만 RUNNING을 보고한다
        self._report(camera_id, "RUNNING", progress=100)
        return PrepareOutcome.DONE

    def _analyze(self, camera_id: int, source_uri: str, stop: threading.Event) -> None:
        detector = self._detector_provider()
        with self._cache.writer(source_uri, self._fps) as writer:
            seconds = analyze_video(
                self._media_root / source_uri,
                detector,
                fps=self._fps,
                conf_thres=self._conf_thres,
                edge_margin=self._edge_margin,
                writer=writer,
                stop=stop,
                on_progress=lambda percent: self._report(
                    camera_id, "PREPARING", progress=percent
                ),
            )
            if stop.is_set():
                raise AnalysisStopped()
            writer.commit(seconds)

    def _fail(self, camera_id: int, message: str) -> PrepareOutcome:
        self._report(camera_id, "ERROR", error_message=message)
        return PrepareOutcome.FAILED

    def _report(
        self,
        camera_id: int,
        status: str,
        progress: int | None = None,
        error_message: str | None = None,
    ) -> None:
        # 보고가 실패해도 분석은 계속한다
        try:
            self._reporter.report(camera_id, status, progress, error_message)
        except Exception:
            logger.warning("상태 보고 중 오류 cameraId=%s status=%s", camera_id, status)
