"""분석 서버 설정. 환경 변수에서 읽어 한 번 만들고 앱 전체가 공유한다."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """서버 설정 모음. frozen이라 만든 뒤에는 바꿀 수 없다(스레드 사이에서 안전하게 공유).

    값은 `from_env()`가 환경 변수에서 채우고, 환경 변수가 없으면 아래 기본값을 쓴다.
    """

    # 관리자 서버와 주고받는 내부 API 호출에 쓰는 인증 토큰(INTERNAL_API_TOKEN)
    internal_token: str
    # YOLO 가중치 파일 경로. 비어 있으면 None(YOLO_WEIGHTS)
    yolo_weights: str | None
    # Flask가 바인딩할 주소와 포트
    host: str
    port: int
    # 영상·녹화 파일이 놓이는 공유 미디어 폴더 루트
    media_root: str = "/srv/media"
    # 탐지 신뢰도 하한. 이보다 낮은 박스는 버린다
    detect_conf: float = 0.3
    # 탐지 모델에 넣을 이미지 한 변 크기(px)
    detect_imgsz: int = 960
    # 분석할 초당 프레임 수. 원본 fps보다 낮게 솎아서 부하를 줄인다
    analyze_fps: int = 5
    # 프레임 가장자리로 보는 띠의 비율(가장자리 진출입 판정용, 0.05 = 변 길이의 5%)
    edge_margin_ratio: float = 0.05
    # 분석 결과 캐시를 두는 폴더
    cache_dir: str = "/var/lib/analyzer/cache"
    # 관리자 서버의 내부 API 기준 주소
    admin_internal_base_url: str = "http://localhost:8082/admin"
    # 결과 전송에 실패했을 때 다시 보내려고 쌓아 두는 시간(초)
    result_retry_buffer_sec: int = 300
    # 사용할 탐지기 종류 이름(detector_provider가 해석한다)
    detector: str = "visdrone"
    # 타일 분할 탐지에서 겹친 박스를 중복으로 볼 겹침 비율. 0이면 중복 제거를 하지 않는다
    detect_dedupe_overlap: float = 0
    # 녹화 영상의 해상도와 초당 프레임
    record_width: int = 1280
    record_height: int = 720
    record_fps: int = 10
    # 녹화 중 임시 파일을 두는 폴더
    record_tmp_dir: str = "/var/lib/analyzer/tmp"
    # 녹화·인코딩에 쓰는 ffmpeg 실행 파일
    ffmpeg_bin: str = "ffmpeg"
    # 라이브 입력이 이 시간(초) 넘게 계속 실패하면 재연결을 포기하고 실패로 본다
    live_reconnect_fail_sec: int = 30
    # 앱 로거 레벨(logging_setup.configure_logging이 사용)
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Config":
        """환경 변수를 읽어 Config를 만든다. 없는 값은 기본값, 숫자는 형 변환한다."""
        return cls(
            internal_token=os.environ.get("INTERNAL_API_TOKEN", ""),
            # 빈 문자열도 "없음"으로 취급하려고 `or None`을 쓴다
            yolo_weights=os.environ.get("YOLO_WEIGHTS") or None,
            host=os.environ.get("HOST", "0.0.0.0"),
            port=int(os.environ.get("PORT", "5000")),
            media_root=os.environ.get("MEDIA_ROOT", "/srv/media"),
            detect_conf=float(os.environ.get("DETECT_CONF", "0.3")),
            detect_imgsz=int(os.environ.get("DETECT_IMGSZ", "960")),
            analyze_fps=int(os.environ.get("ANALYZE_FPS", "5")),
            edge_margin_ratio=float(os.environ.get("EDGE_MARGIN_RATIO", "0.05")),
            cache_dir=os.environ.get("CACHE_DIR", "/var/lib/analyzer/cache"),
            admin_internal_base_url=os.environ.get(
                "ADMIN_INTERNAL_BASE_URL", "http://localhost:8082/admin"
            ),
            result_retry_buffer_sec=int(
                os.environ.get("RESULT_RETRY_BUFFER_SEC", "300")
            ),
            detector=os.environ.get("DETECTOR", "visdrone"),
            detect_dedupe_overlap=float(os.environ.get("DETECT_DEDUPE_OVERLAP", "0")),
            record_width=int(os.environ.get("RECORD_WIDTH", "1280")),
            record_height=int(os.environ.get("RECORD_HEIGHT", "720")),
            record_fps=int(os.environ.get("RECORD_FPS", "10")),
            record_tmp_dir=os.environ.get("RECORD_TMP_DIR", "/var/lib/analyzer/tmp"),
            ffmpeg_bin=os.environ.get("FFMPEG_BIN", "ffmpeg"),
            live_reconnect_fail_sec=int(
                os.environ.get("LIVE_RECONNECT_FAIL_SEC", "30")
            ),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
        )
