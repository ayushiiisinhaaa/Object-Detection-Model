"""Audit YOLO labels and split overlap before training."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="YOLO dataset YAML")
    return parser.parse_args()


def resolve_split(root: Path, value: str | list[str]) -> list[Path]:
    entries = [value] if isinstance(value, str) else value
    images: list[Path] = []
    for entry in entries:
        path = (root / entry).resolve()
        if path.is_dir():
            images.extend(item for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES)
        elif path.is_file() and path.suffix == ".txt":
            images.extend((root / line.strip()).resolve() for line in path.read_text().splitlines() if line.strip())
        else:
            raise FileNotFoundError(f"Split path not found: {path}")
    return sorted(images)


def label_path(image: Path) -> Path:
    parts = list(image.parts)
    if "images" not in parts:
        raise ValueError(f"Image path does not contain an 'images' directory: {image}")
    parts[parts.index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    root_value = Path(config.get("path", args.data.parent))
    root = root_value if root_value.is_absolute() else (args.data.parent / root_value).resolve()
    class_count = len(config["names"])
    errors: list[str] = []
    class_instances: Counter[int] = Counter()
    seen_hashes: dict[str, str] = {}

    for split in ("train", "val", "test"):
        if split not in config:
            continue
        images = resolve_split(root, config[split])
        print(f"{split}: {len(images)} images")
        for image in images:
            if not image.is_file():
                errors.append(f"Missing image: {image}")
                continue
            image_hash = digest(image)
            previous_split = seen_hashes.get(image_hash)
            if previous_split and previous_split != split:
                errors.append(f"Duplicate image across {previous_split}/{split}: {image}")
            seen_hashes[image_hash] = split

            label = label_path(image)
            if not label.is_file():
                errors.append(f"Missing label: {label}")
                continue
            for line_number, line in enumerate(label.read_text().splitlines(), start=1):
                fields = line.split()
                try:
                    class_id = int(fields[0])
                    coordinates = [float(value) for value in fields[1:]]
                except (ValueError, IndexError):
                    errors.append(f"Malformed label: {label}:{line_number}")
                    continue
                if len(coordinates) != 4 or any(value < 0 or value > 1 for value in coordinates):
                    errors.append(f"Invalid normalized box: {label}:{line_number}")
                if not 0 <= class_id < class_count:
                    errors.append(f"Class ID out of range: {label}:{line_number}")
                class_instances[class_id] += 1

    print("class instances:", dict(sorted(class_instances.items())))
    if errors:
        print("\n".join(errors[:100]))
        raise SystemExit(f"Dataset audit failed with {len(errors)} issue(s)")
    print("Dataset audit passed")


if __name__ == "__main__":
    main()
