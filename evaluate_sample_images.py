#!/usr/bin/env python3
"""
evaluate_sample_images.py

Run the trained person + PPE pipeline over a balanced sample set:
  samples/
    helmet/
    no_helmet/

Each subfolder is treated as the expected image-level label.

Usage:
    python evaluate_sample_images.py --samples .\\samples
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2

import config
from detection import PersonDetector
from ppe_detection import PPEDetector
from tracking import PersonTracker


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(directory: Path) -> list[Path]:
    return [path for path in sorted(directory.iterdir()) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def validate_samples_root(samples_root: Path) -> dict:
    helmet_dir = samples_root / "helmet"
    no_helmet_dir = samples_root / "no_helmet"

    result = {
        "root": samples_root,
        "helmet_dir": helmet_dir,
        "no_helmet_dir": no_helmet_dir,
        "issues": [],
        "counts": {},
    }

    if not helmet_dir.is_dir():
        result["issues"].append(f"Missing directory: {helmet_dir}")
    if not no_helmet_dir.is_dir():
        result["issues"].append(f"Missing directory: {no_helmet_dir}")

    helmet_images = iter_images(helmet_dir) if helmet_dir.is_dir() else []
    no_helmet_images = iter_images(no_helmet_dir) if no_helmet_dir.is_dir() else []
    result["counts"] = {
        "helmet": len(helmet_images),
        "no_helmet": len(no_helmet_images),
        "total": len(helmet_images) + len(no_helmet_images),
    }

    if result["counts"]["helmet"] and result["counts"]["no_helmet"]:
        larger = max(result["counts"]["helmet"], result["counts"]["no_helmet"])
        smaller = min(result["counts"]["helmet"], result["counts"]["no_helmet"])
        if smaller > 0 and larger / smaller >= 2:
            result["issues"].append(
                f"Image-count imbalance detected: helmet={result['counts']['helmet']}, no_helmet={result['counts']['no_helmet']}"
            )

    return result


def classify_tracks(tracks) -> str:
    has_unknown = False
    for track in tracks:
        if track.has_helmet is False:
            return "no_helmet"
        if track.has_helmet is None:
            has_unknown = True
    if any(track.has_helmet is True for track in tracks):
        return "helmet"
    return "unknown" if has_unknown or tracks else "unknown"


def evaluate_image(image_path: Path, expected_label: str, person_detector: PersonDetector, ppe_detector: PPEDetector, tracker: PersonTracker) -> dict:
    frame = cv2.imread(str(image_path))
    if frame is None:
        return {
            "image": image_path.name,
            "expected": expected_label,
            "predicted": "read_error",
            "correct": False,
            "helmet": 0,
            "no_helmet": 0,
            "unknown": 0,
            "tracks": 0,
        }

    tracker.reset()
    # DEEPSORT_MAX_AGE frames for n_init to be reached in one go
    detections = person_detector.detect(frame)
    tracks = []
    # Feed the SAME image multiple times to 'trick' the tracker into confirming the track
    # since tracker.update only returns results for confirmed tracks (n_init).
    for _ in range(config.DEEPSORT_N_INIT):
        tracks = tracker.update(detections, frame)
    
    ppe_detector.process_tracks(tracks, frame)

    counts = Counter()
    for track in tracks:
        if track.has_helmet is True:
            counts["helmet"] += 1
        elif track.has_helmet is False:
            counts["no_helmet"] += 1
        else:
            counts["unknown"] += 1

    predicted = classify_tracks(tracks)
    correct = predicted == expected_label

    return {
        "image": image_path.name,
        "expected": expected_label,
        "predicted": predicted,
        "correct": correct,
        "helmet": counts["helmet"],
        "no_helmet": counts["no_helmet"],
        "unknown": counts["unknown"],
        "tracks": len(tracks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the PPE model on a balanced sample image set")
    parser.add_argument("--samples", default="samples", help="Path to the samples root")
    parser.add_argument("--person-model", default=str(Path(__file__).resolve().parent / "yolov8n.pt"), help="Person detector model path")
    parser.add_argument("--device", default=None, help="Inference device override")
    args = parser.parse_args()

    samples_root = Path(args.samples).expanduser().resolve()
    if not samples_root.exists():
        print(f"Samples root not found: {samples_root}")
        return 1

    validation = validate_samples_root(samples_root)
    print(f"Samples root: {samples_root}")
    print(f"  helmet images: {validation['counts'].get('helmet', 0)}")
    print(f"  no_helmet images: {validation['counts'].get('no_helmet', 0)}")
    if validation["issues"]:
        print("Warnings:")
        for issue in validation["issues"]:
            print(f"  - {issue}")

    person_detector = PersonDetector(
        model_path=args.person_model,
        conf_threshold=config.PERSON_CONF_THRESHOLD,
        iou_threshold=config.PERSON_IOU_THRESHOLD,
    )
    ppe_detector = PPEDetector(device=args.device)
    tracker = PersonTracker()

    print(f"Using PPE model: {ppe_detector.model_path}")
    print(f"PPE stub mode: {ppe_detector.is_stub}")
    print("-")

    rows: list[dict] = []
    for expected_label, subdir_name in (("helmet", "helmet"), ("no_helmet", "no_helmet")):
        image_dir = samples_root / subdir_name
        for image_path in iter_images(image_dir):
            result = evaluate_image(image_path, expected_label, person_detector, ppe_detector, tracker)
            rows.append(result)
            print(
                f"{result['image']:<30} expected={result['expected']:<9} predicted={result['predicted']:<9} "
                f"tracks={result['tracks']:<2d} helmet={result['helmet']:<2d} no_helmet={result['no_helmet']:<2d} unknown={result['unknown']:<2d} "
                f"{'OK' if result['correct'] else 'MISS'}"
            )

    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    helmet_total = sum(1 for row in rows if row["expected"] == "helmet")
    no_helmet_total = sum(1 for row in rows if row["expected"] == "no_helmet")
    helmet_correct = sum(1 for row in rows if row["expected"] == "helmet" and row["correct"])
    no_helmet_correct = sum(1 for row in rows if row["expected"] == "no_helmet" and row["correct"])

    print("-")
    print(f"Total images: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {(100.0 * correct / total) if total else 0.0:.2f}%")
    if helmet_total:
        print(f"Helmet accuracy: {(100.0 * helmet_correct / helmet_total):.2f}% ({helmet_correct}/{helmet_total})")
    if no_helmet_total:
        print(f"No-helmet accuracy: {(100.0 * no_helmet_correct / no_helmet_total):.2f}% ({no_helmet_correct}/{no_helmet_total})")

    confusion = Counter((row["expected"], row["predicted"]) for row in rows)
    print("Confusion matrix counts:")
    print(f"  helmet -> helmet: {confusion[('helmet', 'helmet')]}")
    print(f"  helmet -> no_helmet: {confusion[('helmet', 'no_helmet')]}")
    print(f"  helmet -> unknown: {confusion[('helmet', 'unknown')]}")
    print(f"  no_helmet -> helmet: {confusion[('no_helmet', 'helmet')]}")
    print(f"  no_helmet -> no_helmet: {confusion[('no_helmet', 'no_helmet')]}")
    print(f"  no_helmet -> unknown: {confusion[('no_helmet', 'unknown')]}")

    return 0 if total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())