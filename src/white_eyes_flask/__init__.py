from dotenv import load_dotenv

from white_eyes_flask.app import create_app
from white_eyes_flask.config import Config

__all__ = ["create_app", "main"]


def main() -> None:
    load_dotenv()
    config = Config.from_env()
    create_app(config).run(host=config.host, port=config.port)
