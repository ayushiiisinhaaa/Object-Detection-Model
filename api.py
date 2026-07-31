"""FastAPI service for structured object-detection predictions."""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from object_detection import Detector, Settings


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


class DetectionResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0, le=1)
    box: tuple[float, float, float, float]


class PredictionResponse(BaseModel):
    detections: list[DetectionResponse]
    inference_ms: float
    image_width: int
    image_height: int


@lru_cache(maxsize=1)
def get_detector() -> Detector:
    """Load one model per server process."""
    return Detector(Settings.from_env())


app = FastAPI(
    title="Object Detection API",
    description="Structured YOLOv8 inference for JPEG, PNG, and WebP images.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    detector: Detector = Depends(get_detector),
) -> dict[str, object]:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Supported formats: JPEG, PNG, WebP")
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit")

    encoded = np.frombuffer(payload, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=422, detail="Could not decode image")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    prediction, _ = detector.predict(image_rgb)
    return prediction.to_dict()
