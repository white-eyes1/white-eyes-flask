import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    internal_token: str
    yolo_weights: str | None
    host: str
    port: int
    media_root: str = "/srv/media"
    detect_conf: float = 0.3
    detect_imgsz: int = 960
    analyze_fps: int = 5
    edge_margin_ratio: float = 0.05
    cache_dir: str = "/var/lib/analyzer/cache"
    admin_internal_base_url: str = "http://localhost:8082"
    result_retry_buffer_sec: int = 300

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            internal_token=os.environ.get("INTERNAL_API_TOKEN", ""),
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
                "ADMIN_INTERNAL_BASE_URL", "http://localhost:8082"
            ),
            result_retry_buffer_sec=int(
                os.environ.get("RESULT_RETRY_BUFFER_SEC", "300")
            ),
        )
