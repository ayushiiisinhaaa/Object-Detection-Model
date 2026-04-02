import os
from ultralytics import YOLO
import gradio as gr

# Optional: Download weights if not present
if not os.path.exists("weights/yolov8n.pt"):
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id="ultralytics/yolov8", filename="yolov8n.pt", local_dir="weights")

model = YOLO("weights/yolov8n.pt")

def detect_objects(image):
    results = model(image)
    return results[0].plot()

interface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(sources=["upload", "webcam"], type="numpy", label="Input Image"),
    outputs=gr.Image(type="numpy", label="Detection Output"),
    live=False  # Set to False for upload, True for webcam (browser only)
)

interface.launch()