"""더미 영상 분석 준비 단계(FNC-S-009).

Preparer는 CameraJob이 부르는 규칙이고, DummyPreparer가 실제 구현이다. 영상을 분석해 초별
결과 캐시와 좌표 파일을 만들고, 진행 상황을 관리자 앱에 보고한다.
"""

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

# 관리자 앱에 ERROR와 함께 보내는 사유 문구(영상 문제 / 모델 문제 / 그 외).
ERROR_VIDEO = "영상 파일을 읽을 수 없음"
ERROR_MODEL = "분석 모델을 불러올 수 없음"
ERROR_UNEXPECTED = "분석 중 오류가 발생함"


class PrepareOutcome(StrEnum):
    """준비가 어떻게 끝났는지. CameraJob이 이 값으로 다음 단계를 정한다."""

    DONE = "DONE"  # 준비 끝(캐시 있음)
    STOPPED = "STOPPED"  # 정지 요청으로 취소
    FAILED = "FAILED"  # 오류(ERROR 보고함)


class Preparer(Protocol):
    """준비 구현이 지킬 규칙(Protocol). has_cache는 분석 결과가 이미 있는지, prepare는 없을 때 만드는 일이다."""

    def has_cache(self, source_uri: str) -> bool: ...

    def prepare(
        self, camera_id: int, source_uri: str, stop: threading.Event
    ) -> PrepareOutcome: ...


class NullPreparer:
    """준비를 하지 않는다. 캐시는 항상 없고, 정지될 때까지 준비 중으로 둔다."""

    def has_cache(self, source_uri: str) -> bool:
        """캐시는 항상 없다고 답한다."""
        return False

    def prepare(
        self, camera_id: int, source_uri: str, stop: threading.Event
    ) -> PrepareOutcome:
        # 정지 요청이 올 때까지 준비 중인 채로 기다린다.
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
        """더미 준비기를 만든다.

        :param media_root: 공유 폴더 루트(영상의 기준 경로)
        :param cache: 분석 결과 캐시 저장소
        :param detector_provider: 탐지기를 돌려주는 함수(필요할 때 호출)
        :param reporter: 상태 보고기
        :param fps: 분석할 초당 프레임 수
        :param conf_thres: 탐지 신뢰도 하한
        :param edge_margin: 유입·유출을 판정하는 화면 가장자리 띠의 폭(0~1 비율)
        """
        self._media_root = Path(media_root)
        self._cache = cache
        self._detector_provider = detector_provider
        self._reporter = reporter
        self._fps = fps
        self._conf_thres = conf_thres
        self._edge_margin = edge_margin

    def has_cache(self, source_uri: str) -> bool:
        """이 영상의 캐시·좌표 파일이 모두 있나."""
        return self._cache.has(source_uri)

    def prepare(
        self, camera_id: int, source_uri: str, stop: threading.Event
    ) -> PrepareOutcome:
        """캐시가 없으면 영상을 분석해 만들고, 진행률을 보고한다. 결과는 PrepareOutcome.

        같은 영상을 준비하는 다른 캠이 있으면 그 캠이 끝나길 기다린다(중복 분석 방지).
        """
        # 시작 전에 이미 정지됐다.
        if stop.is_set():
            return PrepareOutcome.STOPPED
        # 캐시가 이미 있으면 분석 없이 바로 RUNNING 100%를 보고한다.
        if self._cache.has(source_uri):
            self._report(camera_id, "RUNNING", progress=100)
            return PrepareOutcome.DONE
        # 분석 시작을 알린다(0%).
        self._report(camera_id, "PREPARING", progress=0)
        try:
            # 같은 영상을 준비 중인 다른 캠이 있으면 끝날 때까지 기다린다
            with self._cache.lock(source_uri, stop) as acquired:
                # 기다리는 중에 정지됐거나 잠금을 얻지 못했다.
                if not acquired or stop.is_set():
                    return PrepareOutcome.STOPPED
                # 기다리는 동안 다른 캠이 이미 만들었을 수 있다
                if not self._cache.has(source_uri):
                    self._analyze(camera_id, source_uri, stop)
        # 정지로 분석이 중단됐다. 오류가 아니므로 ERROR를 보고하지 않는다.
        except AnalysisStopped:
            return PrepareOutcome.STOPPED
        # 아래 세 가지는 원인별로 관리자 앱에 알릴 문구를 달리한다.
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
        """영상을 분석해 캐시·좌표 파일을 쓴다.

        쓰기는 writer가 맡고 commit으로 확정한다. 정지로 끝났으면 확정하지 않는다.
        """
        # 탐지기는 필요한 이때 얻는다. 쓸 수 없으면 DetectorUnavailableError가 올라간다.
        detector = self._detector_provider()
        with self._cache.writer(source_uri, self._fps) as writer:
            result = analyze_video(
                self._media_root / source_uri,
                detector,
                fps=self._fps,
                conf_thres=self._conf_thres,
                edge_margin=self._edge_margin,
                writer=writer,
                stop=stop,
                # 분석 진행률(%)이 바뀔 때마다 보고한다.
                on_progress=lambda percent: self._report(
                    camera_id, "PREPARING", progress=percent
                ),
            )
            # 정지 때문에 끝난 분석의 부분 결과는 캐시로 확정하지 않는다.
            if stop.is_set():
                raise AnalysisStopped()
            # 초별 결과와 영상 길이를 확정해 캐시·좌표 파일을 완성한다.
            writer.commit(result.seconds, result.duration_sec)

    def _fail(self, camera_id: int, message: str) -> PrepareOutcome:
        """ERROR를 보고하고 FAILED를 돌려준다."""
        self._report(camera_id, "ERROR", error_message=message)
        return PrepareOutcome.FAILED

    def _report(
        self,
        camera_id: int,
        status: str,
        progress: int | None = None,
        error_message: str | None = None,
    ) -> None:
        """상태를 관리자 앱에 보고한다. 보고 실패는 분석에 영향을 주지 않는다."""
        # 보고가 실패해도 분석은 계속한다
        try:
            self._reporter.report(camera_id, status, progress, error_message)
        except Exception:
            logger.warning("상태 보고 중 오류 cameraId=%s status=%s", camera_id, status)
