import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    internal_token: str
    yolo_weights: str | None
    host: str
    port: int
    media_root: str = "/srv/media"

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            internal_token=os.environ.get("INTERNAL_TOKEN", ""),
            yolo_weights=os.environ.get("YOLO_WEIGHTS") or None,
            host=os.environ.get("HOST", "0.0.0.0"),
            port=int(os.environ.get("PORT", "5000")),
            media_root=os.environ.get("MEDIA_ROOT", "/srv/media"),
        )
