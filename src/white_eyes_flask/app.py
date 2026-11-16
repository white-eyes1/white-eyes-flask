"""Flask 앱 조립 모듈.

설정(Config)으로 탐지기·큐·전송기·캠 서비스 같은 부품을 만들어 서로 연결하고,
블루프린트(캠 제어, 스트림)와 요청 공통 처리(요청 ID, 내부 토큰 검사, health)를 등록한다.
build_* 함수는 부품 하나를 만드는 팩토리이고, create_app이 이들을 엮는다.
"""

import hmac
import uuid

from flask import Flask, g, jsonify, request

from white_eyes_flask.analysis.detector_provider import LazyDetectorProvider
from white_eyes_flask.analysis.dummy_cache import DummyCache
from white_eyes_flask.camera.admin_client import TOKEN_HEADER
from white_eyes_flask.camera.camera_controller import camera_bp
from white_eyes_flask.camera.camera_service import CameraService
from white_eyes_flask.camera.dummy_player import DummyPlayer
from white_eyes_flask.camera.kst import format_at
from white_eyes_flask.camera.player import Player
from white_eyes_flask.camera.preparer import DummyPreparer, Preparer
from white_eyes_flask.camera.recording_registrar import RecordingRegistrar
from white_eyes_flask.camera.restart_recovery import DesiredClient, RestartRecovery
from white_eyes_flask.camera.result_client import ResultClient
from white_eyes_flask.camera.result_queue import ResultQueue
from white_eyes_flask.camera.result_sender import ResultSender
from white_eyes_flask.camera.status_reporter import StatusReporter
from white_eyes_flask.config import Config
from white_eyes_flask.live.live_factory import LiveFactory, LiveWorkFactory
from white_eyes_flask.logging_setup import (
    NO_REQUEST_ID,
    accept_request_id,
    configure_logging,
    remember_request_id,
)
from white_eyes_flask.recording.ffmpeg import RecorderSettings
from white_eyes_flask.recording.publisher import RecordingPublisher
from white_eyes_flask.stream.stream_controller import stream_bp
from white_eyes_flask.stream.stream_service import StreamService

# 요청 추적용 헤더. 호출한 쪽(Spring)이 보내면 그대로 이어 쓰고, 응답에도 같은 값을 돌려준다.
REQUEST_ID_HEADER = "X-Request-Id"


def build_detector_provider(config: Config) -> LazyDetectorProvider:
    """탐지기 제공자를 만든다. 모델은 처음 쓸 때 불러오는 지연 로딩이라 서버 기동이 느려지지 않는다."""
    return LazyDetectorProvider(
        config.yolo_weights,
        config.detect_imgsz,
        config.detect_conf,
        config.detector,
        config.detect_dedupe_overlap,
    )


def build_recording_publisher(config: Config) -> RecordingPublisher:
    """녹화 파일을 공유 미디어 폴더로 옮기고 관리자 서버에 등록하는 게시기를 만든다."""
    return RecordingPublisher(
        config.media_root,
        RecordingRegistrar(config.admin_internal_base_url, config.internal_token),
    )


def build_live_factory(
    config: Config,
    detector_provider: LazyDetectorProvider,
    result_queue: ResultQueue,
    streams: StreamService,
    publisher: RecordingPublisher,
) -> LiveFactory:
    """LIVE(실시간 영상) 캠 작업에 필요한 부품을 한데 묶은 팩토리를 만든다.

    분석 결과는 sink(전송 큐)로, 상태 변화는 reporter로, 화면은 streams로,
    녹화 파일은 publisher로 보낸다. 나머지는 분석·녹화 설정값이다.
    """
    return LiveFactory(
        detector_provider=detector_provider,
        sink=result_queue,
        reporter=StatusReporter(config.admin_internal_base_url, config.internal_token),
        streams=streams,
        publisher=publisher,
        record_settings=RecorderSettings(
            width=config.record_width,
            height=config.record_height,
            fps=config.record_fps,
            analyze_fps=config.analyze_fps,
            tmp_dir=config.record_tmp_dir,
            ffmpeg_bin=config.ffmpeg_bin,
        ),
        fps=config.analyze_fps,
        conf_thres=config.detect_conf,
        edge_margin=config.edge_margin_ratio,
        reconnect_fail_sec=config.live_reconnect_fail_sec,
    )


def build_preparer(
    config: Config,
    cache: DummyCache,
    detector_provider: LazyDetectorProvider | None = None,
) -> DummyPreparer:
    """더미(업로드 영상) 캠의 준비 단계 담당자를 만든다. 영상을 미리 분석해 캐시에 저장한다.

    detector_provider를 넘기면 그것을 공유하고, 없으면 새로 만든다(탐지 모델을 여러 번 올리지 않으려고 공유한다).
    """
    return DummyPreparer(
        media_root=config.media_root,
        cache=cache,
        detector_provider=detector_provider or build_detector_provider(config),
        reporter=StatusReporter(config.admin_internal_base_url, config.internal_token),
        fps=config.analyze_fps,
        conf_thres=config.detect_conf,
        edge_margin=config.edge_margin_ratio,
    )


def build_player(
    config: Config,
    cache: DummyCache,
    result_queue: ResultQueue,
    streams: StreamService | None = None,
) -> DummyPlayer:
    """더미 캠의 재생기를 만든다. 준비 단계에서 만든 캐시를 1초씩 꺼내 결과 큐에 넣고, 녹화 등록·상태 보고·미리보기를 한다."""
    return DummyPlayer(
        cache,
        RecordingRegistrar(config.admin_internal_base_url, config.internal_token),
        result_queue,
        StatusReporter(config.admin_internal_base_url, config.internal_token),
        streams=streams,
        preview_fps=config.analyze_fps,
    )


def build_result_sender(config: Config, result_queue: ResultQueue) -> ResultSender:
    """전송 큐의 결과를 관리자 서버로 보내는 전송기를 만든다. 실패분은 설정된 시간(초)만큼 쌓아 두고 다시 보낸다."""
    return ResultSender(
        result_queue,
        ResultClient(config.admin_internal_base_url, config.internal_token),
        config.result_retry_buffer_sec,
    )


def build_restart_recovery(
    config: Config, camera_service: CameraService
) -> RestartRecovery:
    """서버 재시작 뒤 "켜져 있어야 하는 캠" 목록을 관리자 서버에서 받아 다시 켜는 복구 담당자를 만든다(FNC-S-012)."""
    return RestartRecovery(
        DesiredClient(config.admin_internal_base_url, config.internal_token),
        camera_service,
        StatusReporter(config.admin_internal_base_url, config.internal_token),
    )


def create_app(
    config: Config | None = None,
    preparer: Preparer | None = None,
    player: Player | None = None,
    result_sender: ResultSender | None = None,
    live: LiveWorkFactory | None = None,
) -> Flask:
    """Flask 앱을 만들고 모든 부품을 연결한다.

    preparer·player·result_sender·live는 시험에서 가짜 부품을 끼워 넣는 자리다.
    넘기지 않으면(운영) 설정으로 진짜 부품을 만든다.
    """
    app = Flask(__name__)
    # 설정을 직접 안 주면 환경 변수에서 읽는다
    config = config or Config.from_env()
    # 앱 부품들이 로그를 남기기 전에 먼저 로그 형식을 잡는다
    configure_logging(config.log_level)
    # 요청 처리 함수(토큰 검사 등)가 꺼내 쓰도록 앱에 설정을 걸어 둔다
    app.config["APP_CONFIG"] = config
    # 더미 캠의 사전 분석 결과 캐시(준비기가 쓰고 재생기가 읽는다)
    cache = DummyCache(config.cache_dir, config.media_root)
    # 1초 결과 전송 큐. 재생기가 넣고 전송기(FNC-S-006)가 비운다.
    result_queue = ResultQueue()
    app.extensions["result_queue"] = result_queue
    # 실시간 화면·미리보기용 최신 프레임. 캠 작업이 열고 닫고, 분석 쪽이 프레임을 넣는다.
    stream_service = StreamService()
    app.extensions["stream_service"] = stream_service
    # 실제 부품을 쓸 때만 LIVE 녹화 게시기와 라이브 부품을 만든다(시험은 가짜를 넣거나 LIVE를 기다리기만 하게 둔다).
    # 준비기와 라이브 팩토리가 같은 탐지기 제공자를 공유한다(모델을 한 번만 올린다)
    detector_provider = build_detector_provider(config)
    if live is None and player is None and preparer is None:
        publisher = build_recording_publisher(config)
        # 게시기는 백그라운드에서 주기적으로 녹화 파일을 옮기므로 앱을 만들 때 시작해 둔다
        publisher.start()
        # main()이 종료할 때 꺼낼 수 있게 앱에 등록한다
        app.extensions["recording_publisher"] = publisher
        live = build_live_factory(
            config, detector_provider, result_queue, stream_service, publisher
        )
    # 캠 시작·중지를 맡는 서비스. 컨트롤러와 main()이 앱 확장에서 꺼내 쓴다
    app.extensions["camera_service"] = CameraService(
        config.media_root,
        preparer or build_preparer(config, cache, detector_provider),
        player or build_player(config, cache, result_queue, stream_service),
        stream_service,
        live,
    )
    # 캠 제어 API와 영상 스트림 API를 앱에 붙인다
    app.register_blueprint(camera_bp)
    app.register_blueprint(stream_bp)
    # 실제 재생기를 쓸 때만 전송기를 만든다(가짜 재생기는 큐를 쓰지 않아 보낼 것이 없다).
    if result_sender is None and player is None:
        result_sender = build_result_sender(config, result_queue)
    if result_sender is not None:
        app.extensions["result_sender"] = result_sender
        result_sender.start()

    # 토큰 검사보다 먼저 등록해서 401 응답의 로그에도 요청 ID가 남게 한다.
    @app.before_request
    def assign_request_id():
        """요청마다 요청 ID를 정해 g에 담는다. 호출자가 안전한 ID를 보냈으면 그것을, 아니면 12자리 새 ID를 쓴다."""
        incoming = accept_request_id(request.headers.get(REQUEST_ID_HEADER))
        g.request_id = incoming or uuid.uuid4().hex[:12]
        remember_request_id(g.request_id)  # 요청이 끝난 뒤 찍히는 werkzeug 접속 로그용

    @app.after_request
    def return_request_id(response):
        """응답 헤더에 요청 ID를 실어 보내 호출자가 로그와 대조할 수 있게 한다."""
        response.headers[REQUEST_ID_HEADER] = g.get("request_id", NO_REQUEST_ID)
        return response

    @app.before_request
    def check_internal_token():
        """모든 요청에서 내부 API 토큰을 검사한다. 서버 간 호출만 허용하려는 것이다.

        before_request가 값을 돌려주면 그 값이 곧 응답이 되고 라우트는 실행되지 않는다.
        """
        expected = app.config["APP_CONFIG"].internal_token
        given = request.headers.get(TOKEN_HEADER, "")
        # 서버 토큰이 비어 있으면 모두 거부한다(설정 누락 시 열려 버리지 않게).
        # compare_digest는 비교 시간이 일정해 타이밍 공격으로 토큰을 추측하기 어렵다.
        if not expected or not hmac.compare_digest(given, expected):
            return jsonify(error="UNAUTHORIZED"), 401

    @app.get("/analyzer/v1/health")
    def health():
        """분석 서버 상태 확인. 캠별 상태와 마지막 결과 시각, 못 보낸 결과 수를 알려 준다."""
        # cameras는 캠 번호 순. lastResultAt은 그 캠이 이번에 시작한 뒤 전송 큐에 넣은 마지막
        # 결과 시각이고, 아직 없으면 null이다. bufferSize는 못 보낸 결과 수다.
        # 전송기가 없는 구성(가짜 재생기 시험 등)이면 못 보낸 결과도 0으로 본다
        sender = app.extensions.get("result_sender")
        pending = sender.pending if sender is not None else 0
        queue = app.extensions["result_queue"]
        cameras = []
        for camera_id, status, started_at in app.extensions[
            "camera_service"
        ].camera_states():
            last = queue.last_put_at(camera_id)
            # 캠을 껐다 켠 경우 이전 실행의 옛 결과 시각이 남아 있을 수 있어,
            # 이번 시작 시각 이후의 것만 보여 주고 아니면 null로 둔다
            cameras.append(
                {
                    "cameraId": camera_id,
                    "state": str(status),
                    "lastResultAt": (
                        format_at(last)
                        if last is not None and last >= started_at
                        else None
                    ),
                }
            )
        return jsonify(status="UP", cameras=cameras, bufferSize=pending)

    return app
