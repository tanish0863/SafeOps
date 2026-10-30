"""Validation script for PPE Guard Phase 2 datasets.

Checks that all required datasets are present, structurally correct, and
contain the expected COCO keypoint annotations.

Usage:
    python scripts/validate_datasets.py [--quick]

The --quick flag skips per-image OpenCV readability checks (useful when you
only want to verify counts and JSON structure without loading every image).

Exit code:
    0  all required checks pass
    1  one or more required checks failed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Force UTF-8 output on Windows so box-drawing and tick characters render
# correctly in PowerShell / cmd without requiring -X utf8 on the command line.
import os as _os
if not _os.environ.get("PYTHONUTF8"):
    import subprocess as _sp
    _os.environ["PYTHONUTF8"] = "1"
    raise SystemExit(_sp.run([sys.executable, "-X", "utf8"] + sys.argv).returncode)
del _os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parent.parent
COCO8_DIR = ROOT / "datasets" / "coco8-pose"
COCO_DIR  = ROOT / "datasets" / "coco-pose"
DOWNLOADS = ROOT / "datasets" / "downloads"

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
EXPECTED_KP_COUNT = 17

PASS  = "  ✓"
FAIL  = "  ✗"
WARN  = "  !"
SKIP  = "  -"
SEP   = "─" * 62


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def _ok(msg: str) -> None:
    print(f"{PASS}  {msg}")


def _fail(msg: str) -> None:
    print(f"{FAIL}  {msg}")


def _warn(msg: str) -> None:
    print(f"{WARN}  {msg}")


def _skip(msg: str) -> None:
    print(f"{SKIP}  {msg}")


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def require(self, condition: bool, ok_msg: str, fail_msg: str) -> bool:
        if condition:
            _ok(ok_msg)
        else:
            _fail(fail_msg)
            self.failures.append(fail_msg)
        return condition

    @property
    def ok(self) -> bool:
        return len(self.failures) == 0


# ---------------------------------------------------------------------------
# Section 1: Archive files
# ---------------------------------------------------------------------------

def check_archives(rep: Report) -> None:
    _section("1. Archive files (datasets/downloads/)")
    for name, min_bytes in [
        ("coco8-pose.zip",                 100_000),
        ("val2017.zip",                    500_000_000),
        ("annotations_trainval2017.zip",    50_000_000),
    ]:
        p = DOWNLOADS / name
        if not p.exists():
            _warn(f"{name}  — not downloaded yet (run download_datasets.py)")
            continue
        sz = p.stat().st_size
        ok_size = sz >= min_bytes
        rep.require(
            ok_size,
            f"{name}  {sz:,} bytes",
            f"{name}  only {sz:,} bytes (expected >= {min_bytes:,}; may be incomplete)",
        )


# ---------------------------------------------------------------------------
# Section 2: COCO8-Pose
# ---------------------------------------------------------------------------

def check_coco8(rep: Report) -> None:
    _section("2. COCO8-Pose")

    images_dir = COCO8_DIR / "images"
    labels_dir = COCO8_DIR / "labels"
    yaml_files = list(COCO8_DIR.glob("*.yaml"))

    rep.require(images_dir.is_dir(), "images/ directory exists", "images/ directory MISSING")
    rep.require(labels_dir.is_dir(), "labels/ directory exists", "labels/ directory MISSING")
    rep.require(bool(yaml_files),
                f"YAML config found: {yaml_files[0].name}" if yaml_files else "YAML config found",
                "No *.yaml config file found in coco8-pose/")

    if not images_dir.is_dir() or not labels_dir.is_dir():
        return

    n_img = sum(1 for _ in images_dir.rglob("*.jpg")) + \
            sum(1 for _ in images_dir.rglob("*.png"))
    n_lbl = sum(1 for _ in labels_dir.rglob("*.txt"))

    _ok(f"{n_img} image(s) found")
    _ok(f"{n_lbl} label file(s) found")

    # Spot-check one label for 17-keypoint YOLO pose format
    # Format: class cx cy w h [x y v]*17  → 5 + 17*3 = 56 tokens per row
    sample_labels = list(labels_dir.rglob("*.txt"))
    if sample_labels:
        lf = sample_labels[0]
        lines = [l.strip() for l in lf.read_text().splitlines() if l.strip()]
        if lines:
            tokens = lines[0].split()
            rep.require(
                len(tokens) == 56,
                f"Label token count = 56 (5 bbox + 17×3 kp) in {lf.name}",
                f"Label token count = {len(tokens)} (expected 56) in {lf.name}",
            )


# ---------------------------------------------------------------------------
# Section 3: COCO val2017 images
# ---------------------------------------------------------------------------

def check_val2017(rep: Report, quick: bool) -> dict:
    _section("3. COCO val2017 images")
    val_dir = COCO_DIR / "images" / "val2017"

    if not rep.require(val_dir.is_dir(),
                       "val2017/ directory exists",
                       "val2017/ directory MISSING — run download_datasets.py"):
        return {}

    jpg_files = list(val_dir.glob("*.jpg"))
    n = len(jpg_files)
    in_range = 4_900 <= n <= 5_100
    if in_range:
        _ok(f"{n:,} JPEG images (expected ~5,000)")
    else:
        _warn(f"{n:,} JPEG images (expected ~5,000; count outside expected range)")

    if quick or n == 0:
        _skip("per-image OpenCV readability check skipped (--quick or no images)")
        return {"total": n, "valid": n, "corrupt": []}

    try:
        import cv2
    except ImportError:
        _warn("opencv-python not installed; skipping readability check")
        return {"total": n, "valid": n, "corrupt": []}

    print(f"  Checking readability of {n:,} images …", end="", flush=True)
    corrupt: list[str] = []
    check_every = max(1, n // 200)   # sample at most 200 images for speed
    sampled = jpg_files[::check_every]
    for p in sampled:
        img = cv2.imread(str(p))
        if img is None:
            corrupt.append(p.name)
    print(" done")

    rep.require(
        len(corrupt) == 0,
        f"All {len(sampled)} sampled images readable",
        f"{len(corrupt)} unreadable images in sample: {corrupt[:5]}",
    )
    return {"total": n, "valid": n - len(corrupt), "corrupt": corrupt}


# ---------------------------------------------------------------------------
# Section 4: COCO annotations
# ---------------------------------------------------------------------------

def check_annotations(rep: Report) -> dict:
    _section("4. COCO 2017 annotations")
    ann_path = COCO_DIR / "annotations" / "person_keypoints_val2017.json"

    if not rep.require(ann_path.is_file(),
                       "person_keypoints_val2017.json exists",
                       "person_keypoints_val2017.json MISSING — run download_datasets.py"):
        return {}

    _ok(f"File size: {ann_path.stat().st_size:,} bytes")

    try:
        data = json.loads(ann_path.read_text())
    except json.JSONDecodeError as exc:
        rep.require(False, "", f"JSON parse failed: {exc}")
        return {}
    _ok("JSON parses successfully")

    for key in ("images", "annotations", "categories"):
        rep.require(key in data, f'Key "{key}" present', f'Key "{key}" MISSING')

    n_img  = len(data.get("images", []))
    n_ann  = len(data.get("annotations", []))
    n_cat  = len(data.get("categories", []))
    _ok(f"{n_img:,} image records")
    _ok(f"{n_ann:,} annotation records")
    _ok(f"{n_cat} categor{'y' if n_cat == 1 else 'ies'}")

    # Verify person category has 17 keypoints
    person_cats = [c for c in data.get("categories", []) if c.get("name") == "person"]
    if person_cats:
        kp_names = person_cats[0].get("keypoints", [])
        rep.require(
            len(kp_names) == EXPECTED_KP_COUNT,
            f"Person category declares {EXPECTED_KP_COUNT} keypoints",
            f"Person category declares {len(kp_names)} keypoints (expected {EXPECTED_KP_COUNT})",
        )
        # Check that our schema names match COCO's declared names
        mismatches = [
            (a, b) for a, b in zip(COCO_KEYPOINT_NAMES, kp_names) if a != b
        ]
        rep.require(
            not mismatches,
            "COCO keypoint names match src/schemas.py KEYPOINT_NAMES",
            f"Keypoint name mismatches: {mismatches}",
        )

    # Spot-check annotation keypoint counts
    sample_anns = [a for a in data.get("annotations", [])
                   if a.get("category_id") == 1 and a.get("keypoints")]
    if sample_anns:
        kp_count = len(sample_anns[0]["keypoints"]) // 3
        rep.require(
            kp_count == EXPECTED_KP_COUNT,
            f"Sample annotation has {EXPECTED_KP_COUNT} keypoints ({kp_count*3} values)",
            f"Sample annotation has {kp_count} keypoints (expected {EXPECTED_KP_COUNT})",
        )

    # Check image ID ↔ filename consistency (sample first 20 images)
    val_dir = COCO_DIR / "images" / "val2017"
    if val_dir.is_dir():
        images_sample = data.get("images", [])[:20]
        missing_files = []
        for img_rec in images_sample:
            fname = val_dir / img_rec["file_name"]
            if not fname.exists():
                missing_files.append(img_rec["file_name"])
        rep.require(
            not missing_files,
            f"All {len(images_sample)} sampled annotation image IDs have matching files",
            f"{len(missing_files)} image files missing for sampled annotations: "
            f"{missing_files[:5]}",
        )

    return {"n_images": n_img, "n_annotations": n_ann}


# ---------------------------------------------------------------------------
# Section 5: Schema consistency
# ---------------------------------------------------------------------------

def check_schema_consistency(rep: Report) -> None:
    _section("5. Schema consistency (src/schemas.py vs COCO)")
    try:
        sys.path.insert(0, str(ROOT))
        from src.schemas import KEYPOINT_NAMES as SCHEMA_NAMES  # noqa: PLC0415
        match = SCHEMA_NAMES == COCO_KEYPOINT_NAMES
        rep.require(
            match,
            "src/schemas.py KEYPOINT_NAMES matches COCO 17-point order",
            "src/schemas.py KEYPOINT_NAMES does NOT match COCO order",
        )
    except ImportError as exc:
        _warn(f"Could not import src.schemas: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="Skip per-image OpenCV readability checks")
    args = ap.parse_args()

    print("=" * 62)
    print("  PPE Guard — Phase 2 dataset validation")
    print(f"  Project root : {ROOT}")
    print("=" * 62)

    rep = Report()
    check_archives(rep)
    check_coco8(rep)
    img_stats  = check_val2017(rep, args.quick)
    ann_stats  = check_annotations(rep)
    check_schema_consistency(rep)

    # Final summary
    _section("Summary")
    if img_stats:
        _ok(f"val2017 images total : {img_stats.get('total', 'n/a'):,}")
        _ok(f"val2017 images valid : {img_stats.get('valid', 'n/a'):,}")
        corrupt = img_stats.get("corrupt", [])
        if corrupt:
            _warn(f"Corrupt images       : {len(corrupt)}")
    if ann_stats:
        _ok(f"Annotation images    : {ann_stats.get('n_images', 'n/a'):,}")
        _ok(f"Annotations (person) : {ann_stats.get('n_annotations', 'n/a'):,}")

    print()
    if rep.ok:
        print("  ✓  All checks passed.\n")
        return 0
    else:
        print(f"  ✗  {len(rep.failures)} check(s) failed:\n")
        for f in rep.failures:
            print(f"       • {f}")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
