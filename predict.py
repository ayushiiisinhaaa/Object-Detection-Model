"""Run object detection on one image or a directory of images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from object_detection import Detector, Settings

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Image file or directory")
    parser.add_argument("--output", type=Path, default=Path("runs/predict"))
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    return parser.parse_args()


def image_paths(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        return [source]
    if source.is_dir():
        return sorted(path for path in source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    raise FileNotFoundError(f"No supported image source found: {source}")


def main() -> None:
    args = parse_args()
    settings = Settings(args.model, args.device, args.confidence, args.iou)
    settings.validate()
    detector = Detector(settings)
    args.output.mkdir(parents=True, exist_ok=True)

    records = []
    for path in image_paths(args.source):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not decode image: {path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        prediction, annotated_rgb = detector.predict(image_rgb)
        cv2.imwrite(str(args.output / path.name), cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
        records.append({"source": str(path), **prediction.to_dict()})

    (args.output / "predictions.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Processed {len(records)} image(s); outputs: {args.output}")


if __name__ == "__main__":
    main()
