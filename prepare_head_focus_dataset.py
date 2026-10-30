#!/usr/bin/env python3
"""
prepare_head_focus_dataset.py

Create a head-focused derivative dataset by cropping the upper portion of each
image and rewriting YOLO labels to the cropped coordinate space.

This is intentionally pragmatic:
  - Works with YOLO detect rows (class xc yc w h)
  - Works with YOLO segment rows by converting polygons to bounding boxes
  - Outputs detect-style labels, which makes the result suitable for YOLO detect
    training on head-focused crops

Example:
    python prepare_head_focus_dataset.py --source ./datasets/with-or-without-helmet.yolov8 --output ./datasets/helmet_head_focus --head-fraction 0.40 --keep-class-ids 1 2
"""

from __future__ import annotations

import argparse
import ast
import math
import shutil
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_names(data_yaml: Path) -> list[str]:
    names = ["helmet", "no_helmet"]
    if not data_yaml.exists():
        return names

    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("names:"):
            raw = line.split("names:", 1)[1].strip()
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, (list, tuple)) and parsed:
                    return [str(item) for item in parsed]
            except Exception:
                return names
    return names


def find_image_files(directory: Path) -> list[Path]:
    return [path for path in sorted(directory.iterdir()) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def row_to_bbox(row: list[str], image_w: int, image_h: int) -> tuple[int, float, float, float, float] | None:
    try:
        class_id = int(float(row[0]))
    except ValueError:
        return None

    values = [float(value) for value in row[1:]]
    if len(values) == 4:
        xc, yc, width, height = values
        x1 = (xc - width / 2.0) * image_w
        y1 = (yc - height / 2.0) * image_h
        x2 = (xc + width / 2.0) * image_w
        y2 = (yc + height / 2.0) * image_h
        return class_id, x1, y1, x2, y2

    if len(values) >= 6 and len(values) % 2 == 0:
        xs = values[0::2]
        ys = values[1::2]
        x1 = min(xs) * image_w
        y1 = min(ys) * image_h
        x2 = max(xs) * image_w
        y2 = max(ys) * image_h
        return class_id, x1, y1, x2, y2

    return None


def clip_bbox(x1: float, y1: float, x2: float, y2: float, crop_h: int, image_w: int) -> tuple[float, float, float, float] | None:
    x1 = max(0.0, min(float(image_w), x1))
    x2 = max(0.0, min(float(image_w), x2))
    y1 = max(0.0, min(float(crop_h), y1))
    y2 = max(0.0, min(float(crop_h), y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def bbox_to_yolo(x1: float, y1: float, x2: float, y2: float, image_w: int, image_h: int) -> str:
    xc = ((x1 + x2) / 2.0) / image_w
    yc = ((y1 + y2) / 2.0) / image_h
    width = (x2 - x1) / image_w
    height = (y2 - y1) / image_h
    return f"{xc:.6f} {yc:.6f} {width:.6f} {height:.6f}"


def process_split(source_split: Path, output_split: Path, head_fraction: float, keep_class_ids: set[int] | None) -> dict:
    images_dir = source_split / "images"
    labels_dir = source_split / "labels"
    output_images_dir = output_split / "images"
    output_labels_dir = output_split / "labels"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "images": 0,
        "labels": 0,
        "kept": 0,
        "skipped": 0,
    }

    if not images_dir.is_dir() or not labels_dir.is_dir():
        return summary

    image_paths = find_image_files(images_dir)
    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            summary["skipped"] += 1
            continue

        image_h, image_w = frame.shape[:2]
        crop_h = max(1, int(math.floor(image_h * head_fraction)))
        crop_h = min(crop_h, image_h)
        crop = frame[:crop_h, :].copy()

        label_path = labels_dir / f"{image_path.stem}.txt"
        output_rows: list[str] = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                parsed = row_to_bbox(parts, image_w, image_h)
                if parsed is None:
                    continue
                class_id, x1, y1, x2, y2 = parsed
                if keep_class_ids is not None and class_id not in keep_class_ids:
                    continue

                clipped = clip_bbox(x1, y1, x2, y2, crop_h, image_w)
                if clipped is None:
                    continue
                clipped_x1, clipped_y1, clipped_x2, clipped_y2 = clipped
                output_rows.append(f"{class_id} {bbox_to_yolo(clipped_x1, clipped_y1, clipped_x2, clipped_y2, image_w, crop_h)}")

        image_target = output_images_dir / image_path.name
        label_target = output_labels_dir / f"{image_path.stem}.txt"
        cv2.imwrite(str(image_target), crop)
        label_target.write_text("\n".join(output_rows) + ("\n" if output_rows else ""), encoding="utf-8")

        summary["images"] += 1
        summary["labels"] += 1
        summary["kept"] += len(output_rows)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a head-focused YOLO detection dataset")
    parser.add_argument("--source", required=True, help="Source YOLO dataset root")
    parser.add_argument("--output", required=True, help="Output dataset root")
    parser.add_argument("--head-fraction", type=float, default=0.40, help="Fraction of the image height to keep from the top")
    parser.add_argument("--keep-class-ids", nargs="*", type=int, default=None, help="Optional list of class IDs to keep")
    parser.add_argument("--force", action="store_true", help="Delete the output directory if it already exists")
    args = parser.parse_args()

    source_root = Path(args.source).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()

    if not source_root.exists():
        print(f"Source dataset not found: {source_root}")
        return 1

    if output_root.exists():
        if not args.force:
            print(f"Output directory already exists: {output_root} (use --force to overwrite)")
            return 1
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    names = parse_names(source_root / "data.yaml")
    keep_class_ids = set(args.keep_class_ids) if args.keep_class_ids is not None and len(args.keep_class_ids) > 0 else None

    split_summaries = {}
    for split_name in ("train", "valid", "test"):
        source_split = source_root / split_name
        if not source_split.exists():
            continue
        split_summaries[split_name] = process_split(
            source_split=source_split,
            output_split=output_root / split_name,
            head_fraction=args.head_fraction,
            keep_class_ids=keep_class_ids,
        )

    data_yaml = output_root / "data.yaml"
    yaml_lines = ["train: train/images"]
    if (output_root / "valid").exists():
        yaml_lines.append("val: valid/images")
    if (output_root / "test").exists():
        yaml_lines.append("test: test/images")
    yaml_lines.extend(["", f"nc: {len(names)}", f"names: {names!r}"])
    data_yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    print(f"Source dataset: {source_root}")
    print(f"Output dataset: {output_root}")
    print(f"Head fraction: {args.head_fraction:.2f}")
    if keep_class_ids is not None:
        print(f"Keeping class IDs: {sorted(keep_class_ids)}")
    for split_name, summary in split_summaries.items():
        print(
            f"[{split_name}] images={summary['images']} labels={summary['labels']} "
            f"kept_boxes={summary['kept']} skipped={summary['skipped']}"
        )
    print(f"data.yaml written: {data_yaml}")
    print("Train the output with --task detect for the cropped head dataset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
