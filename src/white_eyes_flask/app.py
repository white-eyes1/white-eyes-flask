import hmac

from flask import Flask, jsonify, request

from white_eyes_flask.analysis.detector_provider import LazyDetectorProvider
from white_eyes_flask.analysis.dummy_cache import DummyCache
from white_eyes_flask.camera.camera_controller import camera_bp
from white_eyes_flask.camera.camera_service import CameraService
from white_eyes_flask.camera.preparer import DummyPreparer, Preparer
from white_eyes_flask.camera.status_reporter import StatusReporter
from white_eyes_flask.config import Config

TOKEN_HEADER = "X-Internal-Token"


def build_preparer(config: Config) -> DummyPreparer:
    return DummyPreparer(
        media_root=config.media_root,
        cache=DummyCache(config.cache_dir, config.media_root),
        detector_provider=LazyDetectorProvider(
            config.yolo_weights, config.detect_imgsz, config.detect_conf
        ),
        reporter=StatusReporter(config.admin_internal_base_url, config.internal_token),
        fps=config.analyze_fps,
        conf_thres=config.detect_conf,
        edge_margin=config.edge_margin_ratio,
    )


def create_app(config: Config | None = None, preparer: Preparer | None = None) -> Flask:
    app = Flask(__name__)
    config = config or Config.from_env()
    app.config["APP_CONFIG"] = config
    app.extensions["camera_service"] = CameraService(
        config.media_root, preparer or build_preparer(config)
    )
    app.register_blueprint(camera_bp)

    @app.before_request
    def check_internal_token():
        expected = app.config["APP_CONFIG"].internal_token
        given = request.headers.get(TOKEN_HEADER, "")
        if not expected or not hmac.compare_digest(given, expected):
            return jsonify(error="UNAUTHORIZED"), 401

    @app.get("/analyzer/v1/health")
    def health():
        # 캠 작업 현황(FNC-S-013)은 캠 작업 구현 때 채운다
        return jsonify(status="UP", cameras=[], bufferSize=0)

    return app
