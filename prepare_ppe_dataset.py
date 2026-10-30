#!/usr/bin/env python3
"""
prepare_ppe_dataset.py

Create a proper train/valid split from a Roboflow export that only includes
train/images and train/labels.

This keeps YOLO annotation lines as-is, so it works for detection or
segmentation labels.

Example:
    python prepare_ppe_dataset.py --source "datasets/with-or-without-helmet.yolov8" --output "datasets/with-or-without-helmet_prepared.yolov8"
"""

from __future__ import annotations

import argparse
import ast
import random
import shutil
from pathlib import Path


def prepare_split(source_root: Path, output_root: Path, val_ratio: float, seed: int) -> Path:
    source_images = source_root / "train" / "images"
    source_labels = source_root / "train" / "labels"

    if not source_images.is_dir() or not source_labels.is_dir():
        raise FileNotFoundError(f"Expected train/images and train/labels in {source_root}")

    if output_root.exists():
        shutil.rmtree(output_root)

    (output_root / "train" / "images").mkdir(parents=True, exist_ok=True)
    (output_root / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (output_root / "valid" / "images").mkdir(parents=True, exist_ok=True)
    (output_root / "valid" / "labels").mkdir(parents=True, exist_ok=True)

    images = [path for path in sorted(source_images.iterdir()) if path.is_file()]
    rng = random.Random(seed)
    rng.shuffle(images)

    valid_count = max(1, int(len(images) * val_ratio)) if len(images) > 1 else 0
    valid_stems = {path.stem for path in images[:valid_count]}

    for image_path in images:
        label_path = source_labels / f"{image_path.stem}.txt"
        if image_path.stem in valid_stems:
            image_target = output_root / "valid" / "images" / image_path.name
            label_target = output_root / "valid" / "labels" / f"{image_path.stem}.txt"
        else:
            image_target = output_root / "train" / "images" / image_path.name
            label_target = output_root / "train" / "labels" / f"{image_path.stem}.txt"

        shutil.copy2(image_path, image_target)
        if label_path.exists():
            shutil.copy2(label_path, label_target)
        else:
            label_target.write_text("", encoding="utf-8")

    source_yaml = source_root / "data.yaml"
    names = ["helmet", "no_helmet"]
    if source_yaml.exists():
        for line in source_yaml.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("names:"):
                raw_names = line.split("names:", 1)[1].strip()
                try:
                    parsed = ast.literal_eval(raw_names)
                    if isinstance(parsed, (list, tuple)) and parsed:
                        names = [str(name) for name in parsed]
                except Exception:
                    pass
                break

    data_yaml = output_root / "data.yaml"
    data_yaml.write_text(
        "train: train/images\n"
        "val: valid/images\n\n"
        f"nc: {len(names)}\n"
        f"names: {names!r}\n",
        encoding="utf-8",
    )
    return output_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a PPE dataset with train/valid splits")
    parser.add_argument("--source", required=True, help="Source Roboflow export root")
    parser.add_argument("--output", required=True, help="Output prepared dataset root")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    source_root = Path(args.source).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    prepared = prepare_split(source_root, output_root, args.val_ratio, args.seed)
    print(f"Prepared dataset at: {prepared}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())