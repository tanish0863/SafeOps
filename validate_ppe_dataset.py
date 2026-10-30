#!/usr/bin/env python3
"""
validate_ppe_dataset.py

Validate a YOLOv8 PPE dataset layout and optionally write a data.yaml file.

Expected structure:
dataset/
 ├── train/
 │    ├── images/
 │    ├── labels/
 ├── valid/
 │    ├── images/
 │    ├── labels/
 ├── data.yaml

Run:
    python validate_ppe_dataset.py --dataset D:/path/to/dataset
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_image_files(directory: Path) -> list[Path]:
    return [path for path in sorted(directory.iterdir()) if path.suffix.lower() in IMAGE_EXTENSIONS]


def validate_split(split_dir: Path) -> dict:
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    issues = []
    class_counts = Counter()
    image_files = []
    label_files = []

    if not images_dir.is_dir():
        issues.append(f"Missing directory: {images_dir}")
    else:
        image_files = find_image_files(images_dir)

    if not labels_dir.is_dir():
        issues.append(f"Missing directory: {labels_dir}")
    else:
        label_files = sorted(labels_dir.glob("*.txt"))

    missing_labels = []
    missing_images = []
    invalid_labels = []
    row_kinds = Counter()

    for image_file in image_files:
        label_file = labels_dir / f"{image_file.stem}.txt"
        if not label_file.exists():
            missing_labels.append(image_file.name)

    for label_file in label_files:
        image_exists = any((images_dir / f"{label_file.stem}{ext}").exists() for ext in IMAGE_EXTENSIONS)
        if not image_exists:
            missing_images.append(label_file.name)

        for line_number, line in enumerate(label_file.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            try:
                class_id = int(float(parts[0]))
                [float(value) for value in parts[1:]]
            except ValueError:
                invalid_labels.append(f"{label_file.name}:{line_number} -> non-numeric values")
                continue

            if len(parts) == 5:
                row_kinds["detect"] += 1
            elif len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
                row_kinds["segment"] += 1
            else:
                invalid_labels.append(
                    f"{label_file.name}:{line_number} -> unsupported YOLO row format ({len(parts)} values)"
                )
                continue

            class_counts[class_id] += 1

    return {
        "images_dir": images_dir,
        "labels_dir": labels_dir,
        "image_count": len(image_files),
        "label_count": len(label_files),
        "missing_labels": missing_labels,
        "missing_images": missing_images,
        "invalid_labels": invalid_labels,
        "class_counts": class_counts,
        "row_kinds": row_kinds,
        "issues": issues,
    }


def write_data_yaml(dataset_root: Path, names: list[str]) -> Path:
    data_yaml = dataset_root / "data.yaml"
    content = (
        "train: train/images\n"
        "val: valid/images\n\n"
        f"nc: {len(names)}\n"
        f"names: {names!r}\n"
    )
    data_yaml.write_text(content, encoding="utf-8")
    return data_yaml


def print_summary(split_name: str, summary: dict) -> None:
    print(f"[{split_name}]")
    print(f"  images: {summary['image_count']}")
    print(f"  labels: {summary['label_count']}")
    if summary["missing_labels"]:
        print(f"  missing label files: {len(summary['missing_labels'])}")
    if summary["missing_images"]:
        print(f"  labels without images: {len(summary['missing_images'])}")
    if summary["invalid_labels"]:
        print(f"  invalid label rows: {len(summary['invalid_labels'])}")
    if summary["class_counts"]:
        print("  class counts:")
        for class_id, count in sorted(summary["class_counts"].items()):
            print(f"    class {class_id}: {count}")
    if summary["row_kinds"]:
        print("  row types:")
        for kind, count in sorted(summary["row_kinds"].items()):
            print(f"    {kind}: {count}")
    if summary["issues"]:
        print("  issues:")
        for issue in summary["issues"]:
            print(f"    - {issue}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a YOLOv8 PPE dataset")
    parser.add_argument("--dataset", required=True, help="Path to the dataset root")
    parser.add_argument("--write-data-yaml", action="store_true", help="Write data.yaml if missing")
    parser.add_argument("--names", nargs=2, default=["helmet", "no_helmet"], help="Class names in YOLO order")
    args = parser.parse_args()

    dataset_root = Path(args.dataset).expanduser().resolve()
    if not dataset_root.exists():
        print(f"Dataset root not found: {dataset_root}")
        return 1

    train_summary = validate_split(dataset_root / "train")
    valid_summary = validate_split(dataset_root / "valid")

    print(f"Dataset root: {dataset_root}")
    print_summary("train", train_summary)
    print_summary("valid", valid_summary)

    data_yaml = dataset_root / "data.yaml"
    if data_yaml.exists():
        print(f"[data.yaml] found: {data_yaml}")
    elif args.write_data_yaml:
        data_yaml = write_data_yaml(dataset_root, args.names)
        print(f"[data.yaml] written: {data_yaml}")
    else:
        print("[data.yaml] missing (use --write-data-yaml to generate it)")

    exit_code = 0
    if train_summary["issues"] or valid_summary["issues"]:
        exit_code = 1
    if train_summary["invalid_labels"] or valid_summary["invalid_labels"]:
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())