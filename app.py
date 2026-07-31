"""Gradio interface for YOLOv8 object detection."""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO


DEFAULT_MODEL = "yolov8n.pt"
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL)


def load_model(model_path: str) -> YOLO:
    """Load a local checkpoint or an official Ultralytics model by name."""
    path = Path(model_path)
    if model_path != DEFAULT_MODEL and path.suffix == ".pt" and not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    return YOLO(model_path)


model = load_model(MODEL_PATH)


def detect_objects(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Run object detection and return the annotated RGB image."""
    result = model.predict(source=image, verbose=False)[0]
    return result.plot()[:, :, ::-1]


demo = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(sources=["upload", "webcam"], type="numpy", label="Input image"),
    outputs=gr.Image(type="numpy", label="Detections"),
    title="YOLOv8n Object Detector",
    description="COCO-pretrained YOLOv8n baseline. Upload an image or use a webcam.",
)


if __name__ == "__main__":
    demo.launch()
