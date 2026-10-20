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
        )
