"""FNC-S-007 LIVE 캠 작업마다 필요한 부품(LivePlayer + LiveRecorder)을 만든다"""

from collections.abc import Callable
from typing import Any, Protocol

from white_eyes_flask.analysis.detector import Detector
from white_eyes_flask.camera.result_queue import ResultSink
from white_eyes_flask.live.live_player import LivePlayer
from white_eyes_flask.live.live_source import open_capture
from white_eyes_flask.recording.ffmpeg import RecorderSettings
from white_eyes_flask.recording.gap_recorder import GapSplittingRecorder
from white_eyes_flask.recording.publisher import RecordingPublisher
from white_eyes_flask.recording.recorder import LiveRecorder
from white_eyes_flask.stream.stream_service import StreamService


class LiveWorkFactory(Protocol):
    """캠 번호로 라이브 재생기와 녹화기 한 쌍을 만들어 주는 팩토리의 약속(시험에서 가짜로 바꿀 수 있게 분리)."""

    def create(
        self, camera_id: int
    ) -> tuple[LivePlayer, GapSplittingRecorder | None]: ...


class LiveFactory:
    """탐지기·전송 큐·게시기는 모든 캠이 나눠 쓰고, 재생기와 녹화기는 캠 작업마다 새로 만든다."""

    def __init__(
        self,
        *,
        detector_provider: Callable[[], Detector],
        sink: ResultSink,
        reporter: Any,
        streams: StreamService,
        publisher: RecordingPublisher,
        record_settings: RecorderSettings,
        fps: int,
        conf_thres: float,
        edge_margin: float,
        capture_factory: Callable[[str], Any] = open_capture,
        reconnect_fail_sec: float | None = None,
    ) -> None:
        """공유 부품과 설정을 받아 둔다. capture_factory는 RTSP를 여는 함수로, 시험에서 가짜로 바꿀 수 있다."""
        self._reconnect_fail_sec = reconnect_fail_sec
        self._detector_provider = detector_provider
        self._sink = sink
        self._reporter = reporter
        self._streams = streams
        self._publisher = publisher
        self._record_settings = record_settings
        self._fps = fps
        self._conf_thres = conf_thres
        self._edge_margin = edge_margin
        self._capture_factory = capture_factory

    def create(self, camera_id: int) -> tuple[LivePlayer, GapSplittingRecorder | None]:
        """캠 하나를 위한 (재생기, 녹화기)를 새로 만든다. 재생기가 프레임을 읽어 분석하고 같은 프레임을 녹화기에도 넘긴다."""
        # 스트림이 끊겼다 복구되면 녹화 파일을 새로 시작한다(LiveRecorder는 파일 하나의 ffmpeg 수명이다)
        recorder = GapSplittingRecorder(
            camera_id,
            lambda: LiveRecorder(camera_id, self._record_settings, self._publisher),
        )
        player = LivePlayer(
            self._detector_provider,
            self._sink,
            self._reporter,
            self._streams,
            recorder,
            fps=self._fps,
            conf_thres=self._conf_thres,
            edge_margin=self._edge_margin,
            capture_factory=self._capture_factory,
            reconnect_fail_sec=self._reconnect_fail_sec,
        )
        return player, recorder
