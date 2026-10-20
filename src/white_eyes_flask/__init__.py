from dotenv import load_dotenv

from white_eyes_flask.app import create_app
from white_eyes_flask.config import Config

__all__ = ["create_app", "main"]


def main() -> None:
    load_dotenv()
    config = Config.from_env()
    app = create_app(config)
    try:
        app.run(host=config.host, port=config.port)
    finally:
        # 끝나기 전에 아직 못 보낸 1초 결과를 한 번 더 보내 본다
        sender = app.extensions.get("result_sender")
        if sender is not None:
            sender.stop()
