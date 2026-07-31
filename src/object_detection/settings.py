"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    """Inference settings shared by the UI and API."""

    model_path: str = "yolov8n.pt"
    device: str = "cpu"
    confidence: float = 0.25
    iou: float = 0.70
    image_size: int = 640

    @classmethod
    def from_env(cls) -> Settings:
        defaults = cls()
        settings = cls(
            model_path=os.getenv("MODEL_PATH", defaults.model_path),
            device=os.getenv("DEVICE", defaults.device),
            confidence=float(os.getenv("CONFIDENCE", str(defaults.confidence))),
            iou=float(os.getenv("IOU_THRESHOLD", str(defaults.iou))),
            image_size=int(os.getenv("IMAGE_SIZE", str(defaults.image_size))),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("CONFIDENCE must be between 0 and 1")
        if not 0 <= self.iou <= 1:
            raise ValueError("IOU_THRESHOLD must be between 0 and 1")
        if self.image_size <= 0:
            raise ValueError("IMAGE_SIZE must be positive")
