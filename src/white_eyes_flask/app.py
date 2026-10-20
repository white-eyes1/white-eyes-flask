import hmac

from flask import Flask, jsonify, request

from white_eyes_flask.camera.camera_controller import camera_bp
from white_eyes_flask.camera.camera_service import CameraService
from white_eyes_flask.config import Config

TOKEN_HEADER = "X-Internal-Token"


def create_app(config: Config | None = None) -> Flask:
    app = Flask(__name__)
    app.config["APP_CONFIG"] = config or Config.from_env()
    app.extensions["camera_service"] = CameraService(
        app.config["APP_CONFIG"].media_root
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
