import hmac
import uuid

from flask import Flask, g, jsonify, request

from white_eyes_flask.analysis.detector_provider import LazyDetectorProvider
from white_eyes_flask.analysis.dummy_cache import DummyCache
from white_eyes_flask.camera.camera_controller import camera_bp
from white_eyes_flask.camera.camera_service import CameraService
from white_eyes_flask.camera.dummy_player import DummyPlayer
from white_eyes_flask.camera.player import Player
from white_eyes_flask.camera.preparer import DummyPreparer, Preparer
from white_eyes_flask.camera.recording_registrar import RecordingRegistrar
from white_eyes_flask.camera.restart_recovery import DesiredClient, RestartRecovery
from white_eyes_flask.camera.result_client import ResultClient
from white_eyes_flask.camera.result_queue import ResultQueue
from white_eyes_flask.camera.result_sender import ResultSender
from white_eyes_flask.camera.status_reporter import StatusReporter
from white_eyes_flask.config import Config
from white_eyes_flask.logging_setup import (
    NO_REQUEST_ID,
    accept_request_id,
    configure_logging,
)

TOKEN_HEADER = "X-Internal-Token"
REQUEST_ID_HEADER = "X-Request-Id"


def build_preparer(config: Config, cache: DummyCache) -> DummyPreparer:
    return DummyPreparer(
        media_root=config.media_root,
        cache=cache,
        detector_provider=LazyDetectorProvider(
            config.yolo_weights,
            config.detect_imgsz,
            config.detect_conf,
            config.detector,
            config.detect_dedupe_overlap,
        ),
        reporter=StatusReporter(config.admin_internal_base_url, config.internal_token),
        fps=config.analyze_fps,
        conf_thres=config.detect_conf,
        edge_margin=config.edge_margin_ratio,
    )


def build_player(
    config: Config, cache: DummyCache, result_queue: ResultQueue
) -> DummyPlayer:
    return DummyPlayer(
        cache,
        RecordingRegistrar(config.admin_internal_base_url, config.internal_token),
        result_queue,
        StatusReporter(config.admin_internal_base_url, config.internal_token),
    )


def build_result_sender(config: Config, result_queue: ResultQueue) -> ResultSender:
    return ResultSender(
        result_queue,
        ResultClient(config.admin_internal_base_url, config.internal_token),
        config.result_retry_buffer_sec,
    )


def build_restart_recovery(
    config: Config, camera_service: CameraService
) -> RestartRecovery:
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
) -> Flask:
    app = Flask(__name__)
    config = config or Config.from_env()
    configure_logging(config.log_level)
    app.config["APP_CONFIG"] = config
    cache = DummyCache(config.cache_dir, config.media_root)
    # 1초 결과 전송 큐. 재생기가 넣고 전송기(FNC-S-006)가 비운다.
    result_queue = ResultQueue()
    app.extensions["result_queue"] = result_queue
    app.extensions["camera_service"] = CameraService(
        config.media_root,
        preparer or build_preparer(config, cache),
        player or build_player(config, cache, result_queue),
    )
    app.register_blueprint(camera_bp)
    # 실제 재생기를 쓸 때만 전송기를 만든다(가짜 재생기는 큐를 쓰지 않아 보낼 것이 없다).
    if result_sender is None and player is None:
        result_sender = build_result_sender(config, result_queue)
    if result_sender is not None:
        app.extensions["result_sender"] = result_sender
        result_sender.start()

    # 토큰 검사보다 먼저 등록해서 401 응답의 로그에도 요청 ID가 남게 한다.
    @app.before_request
    def assign_request_id():
        incoming = accept_request_id(request.headers.get(REQUEST_ID_HEADER))
        g.request_id = incoming or uuid.uuid4().hex[:12]

    @app.after_request
    def return_request_id(response):
        response.headers[REQUEST_ID_HEADER] = g.get("request_id", NO_REQUEST_ID)
        return response

    @app.before_request
    def check_internal_token():
        expected = app.config["APP_CONFIG"].internal_token
        given = request.headers.get(TOKEN_HEADER, "")
        if not expected or not hmac.compare_digest(given, expected):
            return jsonify(error="UNAUTHORIZED"), 401

    @app.get("/analyzer/v1/health")
    def health():
        # 캠 작업 현황(FNC-S-013)은 캠 작업 구현 때 채운다. bufferSize는 못 보낸 결과 수다.
        sender = app.extensions.get("result_sender")
        pending = sender.pending if sender is not None else 0
        return jsonify(status="UP", cameras=[], bufferSize=pending)

    return app
