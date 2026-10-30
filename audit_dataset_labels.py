#!/usr/bin/env python3
"""
audit_dataset_labels.py

Render a random set of labeled images so you can inspect label quality quickly.

This helps catch:
  - loose bounding boxes
  - missing labels
  - wrong class assignments
  - broken segmentation polygons

Outputs:
  review/
    annotated/
    manifest.csv
    contact_sheet.jpg

Example:
    python audit_dataset_labels.py --dataset ./datasets/with-or-without-helmet.yolov8 --count 20 --output ./label_review
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_image_files(directory: Path) -> list[Path]:
    return [path for path in sorted(directory.iterdir()) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def parse_label_rows(label_path: Path) -> list[list[str]]:
    if not label_path.exists():
        return []
    rows: list[list[str]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(stripped.split())
    return rows


def draw_label_overlay(image: np.ndarray, rows: list[list[str]]) -> np.ndarray:
    annotated = image.copy()
    height, width = annotated.shape[:2]

    for row in rows:
        try:
            class_id = int(float(row[0]))
        except ValueError:
            continue

        values = [float(value) for value in row[1:]]
        color = (0, 200, 0) if class_id % 2 == 0 else (0, 120, 220)
        label_text = f"class {class_id}"

        if len(values) == 4:
            xc, yc, box_w, box_h = values
            x1 = int((xc - box_w / 2.0) * width)
            y1 = int((yc - box_h / 2.0) * height)
            x2 = int((xc + box_w / 2.0) * width)
            y2 = int((yc + box_h / 2.0) * height)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label_text, (max(0, x1), max(18, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            continue

        if len(values) >= 6 and len(values) % 2 == 0:
            points = np.array([(int(values[i] * width), int(values[i + 1] * height)) for i in range(0, len(values), 2)], dtype=np.int32)
            cv2.polylines(annotated, [points], isClosed=True, color=color, thickness=2)
            x, y = points[:, 0].min(), points[:, 1].min()
            cv2.putText(annotated, label_text, (max(0, int(x)), max(18, int(y) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return annotated


def letterbox_to_tile(image: np.ndarray, tile_width: int = 320, tile_height: int = 240) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(tile_width / max(width, 1), tile_height / max(height, 1))
    resized = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
    y_offset = (tile_height - resized.shape[0]) // 2
    x_offset = (tile_width - resized.shape[1]) // 2
    canvas[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized
    return canvas


def build_contact_sheet(images: list[np.ndarray], columns: int = 4, tile_width: int = 320, tile_height: int = 240) -> np.ndarray:
    if not images:
        return np.zeros((tile_width, tile_width, 3), dtype=np.uint8)

    resized_tiles = [letterbox_to_tile(image, tile_width=tile_width, tile_height=tile_height) for image in images]

    rows = int(math.ceil(len(resized_tiles) / columns))
    row_images = []
    blank = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)

    for row_index in range(rows):
        start = row_index * columns
        end = start + columns
        row_tiles = resized_tiles[start:end]
        while len(row_tiles) < columns:
            row_tiles.append(blank.copy())
        row_images.append(cv2.hconcat(row_tiles))

    return cv2.vconcat(row_images)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit random YOLO labels visually")
    parser.add_argument("--dataset", required=True, help="Dataset root containing train/valid splits")
    parser.add_argument("--count", type=int, default=20, help="Number of random images to review")
    parser.add_argument("--output", required=True, help="Output review directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    dataset_root = Path(args.dataset).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    annotated_dir = output_root / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    image_candidates: list[Path] = []
    for split_name in ("train", "valid", "test"):
        split_dir = dataset_root / split_name / "images"
        if split_dir.is_dir():
            image_candidates.extend(find_image_files(split_dir))

    if not image_candidates:
        print(f"No images found in: {dataset_root}")
        return 1

    rng = random.Random(args.seed)
    sample_count = min(args.count, len(image_candidates))
    selected = rng.sample(image_candidates, sample_count)

    manifest_path = output_root / "manifest.csv"
    contact_tiles: list[np.ndarray] = []

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "label_file", "rows", "has_labels"])

        for image_path in selected:
            label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
            frame = cv2.imread(str(image_path))
            if frame is None:
                writer.writerow([str(image_path), str(label_path), 0, False])
                continue

            rows = parse_label_rows(label_path)
            annotated = draw_label_overlay(frame, rows)
            output_image_path = annotated_dir / f"{image_path.stem}.jpg"
            cv2.imwrite(str(output_image_path), annotated)
            contact_tiles.append(annotated)
            writer.writerow([str(image_path), str(label_path), len(rows), bool(rows)])

    contact_sheet = build_contact_sheet(contact_tiles, columns=4, tile_width=320, tile_height=240)
    contact_sheet_path = output_root / "contact_sheet.jpg"
    cv2.imwrite(str(contact_sheet_path), contact_sheet)

    print(f"Dataset: {dataset_root}")
    print(f"Reviewed images: {sample_count}")
    print(f"Annotated images: {annotated_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Contact sheet: {contact_sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
