"""white-eyes 분석 서버(Flask) 패키지의 진입점.

`main()`이 실행 스크립트의 시작점이다. 환경 변수를 읽어 앱을 만들고,
서버를 띄우고, 종료할 때 도는 작업을 정리한다.
"""

from dotenv import load_dotenv

from white_eyes_flask.app import build_restart_recovery, create_app
from white_eyes_flask.config import Config

# 패키지 밖에서 가져다 쓸 이름만 공개한다(앱 팩토리와 실행 함수).
__all__ = ["create_app", "main"]


def main() -> None:
    """서버를 시작하고, 종료될 때 도는 작업을 정리한다.

    순서: .env 읽기 -> 설정 만들기 -> 앱 만들기 -> 재시작 복구 시작 -> 서버 실행 -> 종료 정리.
    """
    # .env 파일이 있으면 그 값을 환경 변수로 올린다(Config.from_env가 환경 변수를 읽으므로 먼저 해야 한다).
    load_dotenv()
    # 환경 변수에서 설정값을 모아 불변 설정 객체를 만든다.
    config = Config.from_env()
    # 라우트·서비스·스레드 객체를 조립한 Flask 앱을 만든다.
    app = create_app(config)
    # 가동 중이던 캠을 다시 켠다(FNC-S-012). 서버 시작을 막지 않게 백그라운드에서 돈다.
    recovery = build_restart_recovery(config, app.extensions["camera_service"])
    recovery.start()
    try:
        # 리로더가 main()을 한 번 더 실행하면 복구·전송 스레드가 둘이 되므로 끈다.
        app.run(host=config.host, port=config.port, use_reloader=False)
    finally:
        recovery.stop()
        # 도는 캠을 멈추면서 녹화 파일을 닫고, 막 완성된 파일을 한 번 더 공유 폴더로 옮겨 등록해 본다
        app.extensions["camera_service"].shutdown()
        publisher = app.extensions.get("recording_publisher")
        if publisher is not None:
            publisher.stop()
            publisher.run_due()
        # 끝나기 전에 아직 못 보낸 1초 결과를 한 번 더 보내 본다
        sender = app.extensions.get("result_sender")
        if sender is not None:
            sender.stop()
