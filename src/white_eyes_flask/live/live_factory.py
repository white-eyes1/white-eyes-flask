"""FNC-S-007 LIVE 캠 작업마다 필요한 부품(LivePlayer + LiveRecorder)을 만든다"""

from collections.abc import Callable
from typing import Any, Protocol

from white_eyes_flask.analysis.detector import Detector
from white_eyes_flask.camera.result_queue import ResultSink
from white_eyes_flask.live.live_player import LivePlayer
from white_eyes_flask.live.live_source import open_capture
from white_eyes_flask.recording.ffmpeg import RecorderSettings
from white_eyes_flask.recording.publisher import RecordingPublisher
from white_eyes_flask.recording.recorder import LiveRecorder
from white_eyes_flask.stream.stream_service import StreamService


class LiveWorkFactory(Protocol):
    def create(self, camera_id: int) -> tuple[LivePlayer, LiveRecorder | None]: ...


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

    def create(self, camera_id: int) -> tuple[LivePlayer, LiveRecorder | None]:
        recorder = LiveRecorder(camera_id, self._record_settings, self._publisher)
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
