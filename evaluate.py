"""Evaluate a YOLO checkpoint on a YOLO-format validation dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov8n.pt", help="Checkpoint path or model name")
    parser.add_argument("--data", required=True, type=Path, help="Dataset YAML path")
    parser.add_argument("--device", default="cpu", help="Ultralytics device, e.g. cpu or 0")
    parser.add_argument("--imgsz", default=640, type=int, help="Square inference image size")
    parser.add_argument("--project", default="runs/val", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {args.data}")

    metrics = YOLO(args.model).val(
        data=str(args.data),
        device=args.device,
        imgsz=args.imgsz,
        project=args.project,
        plots=True,
    )
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"precision: {metrics.box.mp:.4f}")
    print(f"recall: {metrics.box.mr:.4f}")
    print(f"artifacts: {metrics.save_dir}")


if __name__ == "__main__":
    main()
