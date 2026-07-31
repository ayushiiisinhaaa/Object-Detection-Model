"""Evaluate a YOLO checkpoint on a YOLO-format validation dataset."""

from __future__ import annotations

import argparse
import json
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
    summary = {
        "model": args.model,
        "data": str(args.data),
        "device": args.device,
        "image_size": args.imgsz,
        "map50": round(float(metrics.box.map50), 6),
        "map50_95": round(float(metrics.box.map), 6),
        "precision": round(float(metrics.box.mp), 6),
        "recall": round(float(metrics.box.mr), 6),
        "speed_ms": {key: round(float(value), 3) for key, value in metrics.speed.items()},
        "per_class_map50_95": {
            str(metrics.names[index]): round(float(value), 6)
            for index, value in enumerate(metrics.box.maps)
        },
    }
    output_path = Path(metrics.save_dir) / "metrics.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"artifacts: {metrics.save_dir}")


if __name__ == "__main__":
    main()
