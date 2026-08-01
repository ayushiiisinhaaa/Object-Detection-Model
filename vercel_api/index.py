"""Compact ONNX YOLOv8 API for Vercel's Python runtime."""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Annotated

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps

IMAGE_SIZE = 640
CONFIDENCE = 0.25
IOU_THRESHOLD = 0.70
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
CLASS_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)


@lru_cache(maxsize=1)
def session() -> ort.InferenceSession:
    model = Path(__file__).with_name("yolov8n.onnx")
    return ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])


def preprocess(image: Image.Image) -> tuple[np.ndarray, float, tuple[int, int]]:
    width, height = image.size
    scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
    resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BILINEAR)
    left = (IMAGE_SIZE - resized.width) // 2
    top = (IMAGE_SIZE - resized.height) // 2
    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (114, 114, 114))
    canvas.paste(resized, (left, top))
    tensor = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return np.expand_dims(tensor, 0), scale, (left, top)


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(box_area + areas - intersection, 1e-7)


def nms(boxes: np.ndarray, scores: np.ndarray) -> list[int]:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        order = remaining[box_iou(boxes[current], boxes[remaining]) <= IOU_THRESHOLD]
    return keep


def postprocess(
    output: np.ndarray,
    scale: float,
    padding: tuple[int, int],
    image_size: tuple[int, int],
) -> list[dict[str, object]]:
    predictions = output[0].T
    class_ids = predictions[:, 4:].argmax(axis=1)
    scores = predictions[np.arange(len(predictions)), class_ids + 4]
    selected = scores >= CONFIDENCE
    predictions, class_ids, scores = predictions[selected], class_ids[selected], scores[selected]
    if not len(predictions):
        return []

    centers = predictions[:, :4]
    boxes = np.column_stack((
        centers[:, 0] - centers[:, 2] / 2,
        centers[:, 1] - centers[:, 3] / 2,
        centers[:, 0] + centers[:, 2] / 2,
        centers[:, 1] + centers[:, 3] / 2,
    ))
    left, top = padding
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / scale
    width, height = image_size
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)

    detections: list[dict[str, object]] = []
    for class_id in np.unique(class_ids):
        indexes = np.where(class_ids == class_id)[0]
        for index in indexes[nms(boxes[indexes], scores[indexes])]:
            detections.append({
                "class_id": int(class_id),
                "class_name": CLASS_NAMES[int(class_id)],
                "confidence": round(float(scores[index]), 6),
                "box": [round(float(value), 2) for value in boxes[index]],
            })
    return sorted(detections, key=lambda item: float(item["confidence"]), reverse=True)


app = FastAPI(title="YOLOv8 ONNX API", version="1.0.0")


@app.get("/api/health")
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "runtime": "onnx"}


@app.post("/api/predict")
@app.post("/predict")
async def predict(file: Annotated[UploadFile, File()]) -> dict[str, object]:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Supported formats: JPEG, PNG, WebP")
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image exceeds the 10 MB limit")
    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(payload))).convert("RGB")
    except Exception as error:
        raise HTTPException(422, "Could not decode image") from error

    tensor, scale, padding = preprocess(image)
    started = perf_counter()
    model = session()
    output = model.run(None, {model.get_inputs()[0].name: tensor})[0]
    inference_ms = round((perf_counter() - started) * 1000, 2)
    return {
        "detections": postprocess(output, scale, padding, image.size),
        "inference_ms": inference_ms,
        "image_width": image.width,
        "image_height": image.height,
    }
