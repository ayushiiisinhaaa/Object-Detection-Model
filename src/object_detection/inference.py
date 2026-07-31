"""Model loading, inference, and prediction serialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO

from .settings import Settings


@dataclass(frozen=True, slots=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Prediction:
    detections: tuple[Detection, ...]
    inference_ms: float
    image_width: int
    image_height: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "detections": [asdict(item) for item in self.detections],
            "inference_ms": self.inference_ms,
            "image_width": self.image_width,
            "image_height": self.image_height,
        }


class Detector:
    """Thin, reusable wrapper around an Ultralytics YOLO checkpoint."""

    def __init__(self, settings: Settings, model: Any | None = None) -> None:
        self.settings = settings
        self.model = model if model is not None else self._load_model(settings.model_path)

    @staticmethod
    def _load_model(model_path: str) -> YOLO:
        path = Path(model_path)
        if model_path != "yolov8n.pt" and path.suffix == ".pt" and not path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        return YOLO(model_path)

    def predict(self, image: NDArray[np.uint8]) -> tuple[Prediction, NDArray[np.uint8]]:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must have three color channels")

        started = perf_counter()
        result = self.model.predict(
            source=image,
            conf=self.settings.confidence,
            iou=self.settings.iou,
            imgsz=self.settings.image_size,
            device=self.settings.device,
            verbose=False,
        )[0]
        elapsed_ms = (perf_counter() - started) * 1000

        detections: list[Detection] = []
        if result.boxes is not None:
            for box, confidence, class_id in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
                strict=True,
            ):
                numeric_id = int(class_id)
                detections.append(
                    Detection(
                        class_id=numeric_id,
                        class_name=str(result.names[numeric_id]),
                        confidence=round(float(confidence), 6),
                        box=tuple(round(float(value), 2) for value in box),
                    )
                )

        height, width = image.shape[:2]
        prediction = Prediction(tuple(detections), round(elapsed_ms, 2), width, height)
        annotated_rgb = result.plot()[:, :, ::-1]
        return prediction, annotated_rgb
