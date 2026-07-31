"""Gradio interface backed by the shared detector service."""

from __future__ import annotations

import gradio as gr
import numpy as np
from numpy.typing import NDArray

from object_detection import Detector, Settings


detector = Detector(Settings.from_env())


def detect_objects(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Run object detection and return the annotated RGB image."""
    _, annotated = detector.predict(image)
    return annotated


demo = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(sources=["upload", "webcam"], type="numpy", label="Input image"),
    outputs=gr.Image(type="numpy", label="Detections"),
    title="YOLOv8n Object Detector",
    description="COCO-pretrained YOLOv8n baseline. Upload an image or use a webcam.",
)


if __name__ == "__main__":
    demo.launch()
