from dotenv import load_dotenv

from white_eyes_flask.app import build_restart_recovery, create_app
from white_eyes_flask.config import Config

__all__ = ["create_app", "main"]


def main() -> None:
    load_dotenv()
    config = Config.from_env()
    app = create_app(config)
    # 가동 중이던 캠을 다시 켠다(FNC-S-012). 서버 시작을 막지 않게 백그라운드에서 돈다.
    recovery = build_restart_recovery(config, app.extensions["camera_service"])
    recovery.start()
    try:
        app.run(host=config.host, port=config.port)
    finally:
        recovery.stop()
        # 끝나기 전에 아직 못 보낸 1초 결과를 한 번 더 보내 본다
        sender = app.extensions.get("result_sender")
        if sender is not None:
            sender.stop()
