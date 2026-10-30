"""
scripts/test_helmet.py — Integration test: run PPEHelmetAdapter on sample images.

Usage
-----
    python scripts/test_helmet.py

Outputs
-------
    artifacts/helmet_integration/annotated/<name>_annotated.jpg  — one per sample
    artifacts/helmet_integration/predictions.json                — all results
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import cv2

# ── Ensure project root is on sys.path when running the script directly ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ppe.helmet_adapter import PPEHelmetAdapter

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────
MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
SAMPLES_ROOT = PROJECT_ROOT / "samples"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "helmet_integration"
ANNOTATED_DIR = OUTPUT_DIR / "annotated"
PREDICTIONS_JSON = OUTPUT_DIR / "predictions.json"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEVICE = "cuda"
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


def discover_samples(root: Path) -> list[Path]:
    """
    Recursively find all supported image files under *root*.
    Returns the list sorted for deterministic ordering.
    """
    found: list[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        found.extend(root.rglob(f"*{ext}"))
        found.extend(root.rglob(f"*{ext.upper()}"))
    return sorted(set(found))


def run_test() -> None:
    # ── Create output directories ────────────────────────────────────────
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Initialize adapter ───────────────────────────────────────────────
    logger.info("Initializing PPEHelmetAdapter (device=%s)", DEVICE)
    try:
        adapter = PPEHelmetAdapter(
            model_path=MODEL_PATH,
            device=DEVICE,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
        )
    except Exception as exc:
        logger.error("Failed to initialize adapter: %s", exc)
        sys.exit(1)

    # ── Discover sample images ───────────────────────────────────────────
    sample_paths = discover_samples(SAMPLES_ROOT)

    if not sample_paths:
        logger.warning(
            "No sample images found under '%s'. "
            "Sample-image testing is BLOCKED — no images to process.",
            SAMPLES_ROOT,
        )
        sys.exit(0)

    logger.info("Found %d sample image(s) under '%s'.", len(sample_paths), SAMPLES_ROOT)

    # ── Run inference ────────────────────────────────────────────────────
    all_results: list[dict] = []
    skipped: list[str] = []

    for idx, img_path in enumerate(sample_paths, start=1):
        logger.info("[%d/%d] Processing: %s", idx, len(sample_paths), img_path.name)

        # Read image with OpenCV first so we can validate it
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning("  SKIP — cv2.imread returned None for: %s", img_path.name)
            skipped.append(str(img_path))
            continue

        # Single-pass inference — no duplicate YOLO call for annotation
        try:
            predictions, raw_result = adapter.predict_with_result(frame)
        except Exception as exc:
            logger.error("  SKIP — inference error for '%s': %s", img_path.name, exc)
            skipped.append(str(img_path))
            continue

        # Save annotated image (uses the already-computed result, no re-inference)
        stem = img_path.stem
        annotated_filename = f"{stem}_annotated.jpg"
        annotated_path = ANNOTATED_DIR / annotated_filename
        annotated_bgr = raw_result.plot()
        cv2.imwrite(str(annotated_path), annotated_bgr)

        relative_annotated = str(annotated_path.relative_to(PROJECT_ROOT)).replace("\\", "/")

        # Accumulate JSON record
        all_results.append(
            {
                "image": img_path.name,
                "predictions": predictions,
                "annotated_output": relative_annotated,
            }
        )

        # Console summary
        if predictions:
            for det in predictions:
                logger.info(
                    "  → %s  conf=%.4f  bbox=%s",
                    det["class_name"],
                    det["confidence"],
                    det["bbox"],
                )
        else:
            logger.info("  → No helmet-class detections above threshold.")

        logger.info("  Annotated image saved: %s", annotated_path)

    # ── Save predictions JSON ────────────────────────────────────────────
    with PREDICTIONS_JSON.open("w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2)
    logger.info("Predictions saved to: %s", PREDICTIONS_JSON)

    # ── Final report ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" PPEHelmetAdapter — Integration Test Complete")
    print("=" * 60)
    print(f" Samples found    : {len(sample_paths)}")
    print(f" Processed        : {len(all_results)}")
    print(f" Skipped/error    : {len(skipped)}")
    print(f" JSON output      : {PREDICTIONS_JSON}")
    print(f" Annotated images : {ANNOTATED_DIR}")
    print("=" * 60)

    if skipped:
        print("\nSkipped files:")
        for s in skipped:
            print(f"  {s}")


if __name__ == "__main__":
    run_test()
