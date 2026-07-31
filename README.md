# YOLOv8 Object Detection Demo

An interactive Gradio demo using the official COCO-pretrained YOLOv8n checkpoint.
This repository does **not** currently include a custom-trained checkpoint or
custom-dataset metrics. The included notebook is a sanitized, historical YOLOv5
training experiment and is not the model served by the app.

## Model status

| Component | Current state |
|---|---|
| Deployed model | Ultralytics YOLOv8n, pretrained on COCO |
| Custom weights | Not included |
| Custom metrics | Not available; no successful run artifacts are committed |
| UI | Gradio image upload and webcam inference |

This explicit baseline avoids presenting an incomplete YOLOv5 training attempt as
the deployed model. Once a validated custom checkpoint exists, set `MODEL_PATH`
to its path and publish its metrics and plots here.

## Pipeline

```text
Image upload/webcam -> YOLOv8 preprocessing -> backbone and detection head
                    -> non-maximum suppression -> boxes and labels -> Gradio UI
```

## Run locally

Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

On first use, Ultralytics downloads the official `yolov8n.pt` checkpoint. To use
a local validated checkpoint instead:

```bash
# Windows PowerShell
$env:MODEL_PATH = "weights/best.pt"
python app.py
```

## Reproduce evaluation

The evaluator requires a YOLO-format dataset YAML with separate train and
validation splits. It prints aggregate metrics and writes a confusion matrix,
PR curves, and prediction plots under `runs/val/`.

```bash
python evaluate.py --model weights/best.pt --data path/to/data.yaml --device cpu
```

Record results only from the held-out validation or test split:

| Model | Dataset split | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Inference |
|---|---|---:|---:|---:|---:|---:|
| Custom model | Pending a reproducible training run | - | - | - | - | - |

## Model and dataset choices

YOLOv8n is used for the demo because its small checkpoint favors low-latency CPU
inference and easy deployment. That speed comes at an accuracy cost relative to
larger YOLO variants. A custom model should only replace it after evaluation on a
held-out split demonstrates that the specialization is worthwhile.

The historical notebook explored combining wildlife and miner datasets. That
experiment did not establish a valid benchmark: its generated configuration used
the training images as validation data and the captured training run failed.

## Known limitations

- The demo recognizes the 80 COCO classes; it is not specialized for the notebook datasets.
- No repository-owned validation dataset or custom `best.pt` is currently available.
- Latency varies by hardware and has not yet been benchmarked here.
- The historical notebook needs a clean train/validation split and label audit before retraining.

## Security note

Notebook outputs were cleared because an uploaded Kaggle credential was captured
in the original output. The exposed token must be revoked and regenerated in
Kaggle account settings; deleting it from this working tree does not invalidate it
or remove it from any prior Git history.

## License

This project is licensed under the [MIT License](LICENSE). Ultralytics models and
software have their own licensing terms; review them before commercial use.
