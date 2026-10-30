"""
scripts/test_ppe_pipeline.py — Integration test for the full SafeOps PPE Pipeline.

Runs PersonDetector + PPEHelmetAdapter + HelmetAssociator on sample images.

Outputs:
    artifacts/ppe_pipeline/annotated/<name>_annotated.jpg
    artifacts/ppe_pipeline/predictions.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ppe.ppe_pipeline import PPEPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_samples(root: Path) -> list[Path]:
    found: list[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        found.extend(root.rglob(f"*{ext}"))
        found.extend(root.rglob(f"*{ext.upper()}"))
    return sorted(set(found))


def parse_args():
    parser = argparse.ArgumentParser(description="Test Full SafeOps PPE Pipeline")
    parser.add_argument(
        "--person-model",
        type=str,
        default=str(PROJECT_ROOT / "yolov8n.pt"),
        help="Path to person detector model",
    )
    parser.add_argument(
        "--helmet-model",
        type=str,
        default=str(PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"),
        help="Path to helmet detector checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda or cpu)",
    )
    parser.add_argument(
        "--person-conf",
        type=float,
        default=0.40,
        help="Confidence threshold for person detector",
    )
    parser.add_argument(
        "--helmet-conf",
        type=float,
        default=0.25,
        help="Confidence threshold for helmet detector",
    )
    parser.add_argument(
        "--samples-dir",
        type=str,
        default=str(PROJECT_ROOT / "samples"),
        help="Directory containing sample images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "ppe_pipeline"),
        help="Output directory for annotated images and JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    samples_dir = Path(args.samples_dir)
    output_dir = Path(args.output_dir)
    annotated_dir = output_dir / "annotated"
    predictions_file = output_dir / "predictions.json"

    annotated_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing PPEPipeline on device='%s'", args.device)
    pipeline = PPEPipeline(
        person_model_path=args.person_model,
        helmet_model_path=args.helmet_model,
        device=args.device,
        person_conf=args.person_conf,
        helmet_conf=args.helmet_conf,
    )

    sample_paths = discover_samples(samples_dir)
    if not sample_paths:
        logger.warning("No sample images found under '%s'. Exiting.", samples_dir)
        sys.exit(0)

    logger.info("Discovered %d sample images under '%s'.", len(sample_paths), samples_dir)

    all_predictions = []
    skipped = []

    for idx, img_path in enumerate(sample_paths, start=1):
        logger.info("[%d/%d] Processing: %s", idx, len(sample_paths), img_path.name)
        try:
            result, annotated_frame = pipeline.predict_and_annotate(img_path)
        except Exception as exc:
            logger.error("  Error processing '%s': %s", img_path.name, exc)
            skipped.append(str(img_path))
            continue

        annotated_name = f"{img_path.stem}_annotated.jpg"
        annotated_path = annotated_dir / annotated_name
        cv2.imwrite(str(annotated_path), annotated_frame)

        relative_annotated = str(annotated_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        result["annotated_output"] = relative_annotated
        all_predictions.append(result)

        summary = result["summary"]
        logger.info(
            "  -> Persons: %d | Associated: %d | Unmatched: %d | Ambiguous: %d",
            summary["persons_detected"],
            summary["helmets_associated"],
            summary["unmatched_helmets"],
            summary["ambiguous_associations"],
        )

    with predictions_file.open("w", encoding="utf-8") as fh:
        json.dump(all_predictions, fh, indent=2)

    logger.info("Predictions written to: %s", predictions_file)

    print("\n" + "=" * 60)
    print(" SafeOps PPE Pipeline — Test Complete")
    print("=" * 60)
    print(f" Samples found    : {len(sample_paths)}")
    print(f" Processed        : {len(all_predictions)}")
    print(f" Skipped/Error    : {len(skipped)}")
    print(f" Predictions JSON : {predictions_file}")
    print(f" Annotated images : {annotated_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
