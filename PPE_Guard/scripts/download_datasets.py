"""Dataset download and extraction helper for PPE Guard Phase 2.

Downloads COCO8-Pose, COCO 2017 val images, and COCO 2017 annotations.
All archives land in datasets/downloads/ and are extracted in place.

Usage:
    python scripts/download_datasets.py [--skip-coco-val]

The --skip-coco-val flag skips the large (~1 GB) COCO val2017 image archive;
useful when you only need the annotations or the small COCO8-Pose set.

Resumable: curl -C - is used so an interrupted download continues from where it
stopped. Re-running the script is always safe; it never overwrites a valid archive
or a fully-extracted directory.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root — two levels up from scripts/
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS = ROOT / "datasets" / "downloads"
COCO8_DIR = ROOT / "datasets" / "coco8-pose"
COCO_DIR  = ROOT / "datasets" / "coco-pose"

DATASETS = {
    "coco8-pose": {
        "url":      "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8-pose.zip",
        "archive":  DOWNLOADS / "coco8-pose.zip",
        "dest":     ROOT / "datasets",           # extracts to datasets/coco8-pose/
        "sentinel": COCO8_DIR / "images",        # existence check after extraction
        "min_size": 100_000,                     # bytes – sanity floor
    },
    "val2017": {
        "url":      "https://images.cocodataset.org/zips/val2017.zip",
        "archive":  DOWNLOADS / "val2017.zip",
        "dest":     COCO_DIR / "images",         # extracts to .../images/val2017/
        "sentinel": COCO_DIR / "images" / "val2017",
        "min_size": 500_000_000,                 # ~1 GB; floor at 500 MB
    },
    "annotations": {
        "url":      "https://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "archive":  DOWNLOADS / "annotations_trainval2017.zip",
        "dest":     None,                        # handled selectively
        "sentinel": COCO_DIR / "annotations" / "person_keypoints_val2017.json",
        "min_size": 50_000_000,                  # ~250 MB; floor at 50 MB
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str, *, prefix: str = "  ") -> None:
    print(prefix + msg, flush=True)


def download(url: str, dest: Path, min_size: int = 0) -> bool:
    """Download url to dest using curl with resume support.
    Returns True if a (possibly resumed) complete file is available.

    Note: images.cocodataset.org is served from Amazon CloudFront with a
    wildcard cert that Windows Schannel sometimes rejects (SEC_E_WRONG_PRINCIPAL).
    --insecure is used for that specific CDN; the content is the well-known
    public COCO dataset and carries no credentials.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size >= min_size:
        log(f"[skip]  {dest.name}  (already downloaded, {dest.stat().st_size:,} bytes)")
        return True

    log(f"[down]  {url}")
    log(f"        -> {dest}")
    # Use --insecure for the COCO CDN (cocodataset.org → CloudFront wildcard cert)
    insecure = "cocodataset.org" in url
    cmd = [
        "curl.exe", "--location",          # follow redirects
        "--continue-at", "-",              # resume
        "--retry", "3",
        "--retry-delay", "5",
        "--output", str(dest),
    ]
    if insecure:
        cmd.append("--insecure")
    cmd.append(url)
    t0 = time.time()
    result = subprocess.run(cmd + [url])
    elapsed = time.time() - t0

    if result.returncode != 0:
        log(f"[FAIL]  curl exited {result.returncode}")
        return False

    if not dest.exists():
        log("[FAIL]  file not created after download")
        return False

    sz = dest.stat().st_size
    log(f"[ok]    {sz:,} bytes in {elapsed:.0f}s ({sz/max(elapsed,1)/1e6:.1f} MB/s)")
    if sz < min_size:
        log(f"[WARN]  file is smaller than expected minimum ({min_size:,} bytes) — may be incomplete")
    return True


def extract_zip(archive: Path, dest: Path, *, members: list[str] | None = None) -> bool:
    """Extract archive to dest, optionally filtering to a specific member list."""
    log(f"[unzip] {archive.name}  -> {dest}")
    try:
        with zipfile.ZipFile(archive) as zf:
            if members:
                for m in members:
                    zf.extract(m, dest)
            else:
                zf.extractall(dest)
        return True
    except zipfile.BadZipFile as exc:
        log(f"[FAIL]  bad zip: {exc}")
        return False
    except Exception as exc:
        log(f"[FAIL]  extraction error: {exc}")
        return False


def check_sentinel(path: Path) -> bool:
    if path.is_dir():
        return any(path.iterdir())   # non-empty dir
    return path.is_file() and path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Per-dataset steps
# ---------------------------------------------------------------------------

def setup_coco8() -> bool:
    ds = DATASETS["coco8-pose"]
    print("\n━━━  COCO8-Pose  ━━━")

    if check_sentinel(ds["sentinel"]):
        log("[skip]  already extracted (images/ dir found)")
        return True

    ok = download(ds["url"], ds["archive"], ds["min_size"])
    if not ok:
        return False

    ok = extract_zip(ds["archive"], ds["dest"])
    if not ok:
        return False

    if not check_sentinel(ds["sentinel"]):
        log("[FAIL]  extraction succeeded but images/ dir not found")
        return False

    log("[ok]    COCO8-Pose ready")
    return True


def setup_val2017(skip: bool) -> bool:
    ds = DATASETS["val2017"]
    print("\n━━━  COCO val2017 images  ━━━")

    if skip:
        log("[skip]  --skip-coco-val passed; skipping large image archive")
        return True

    if check_sentinel(ds["sentinel"]):
        n = sum(1 for _ in ds["sentinel"].glob("*.jpg"))
        log(f"[skip]  already extracted ({n:,} images in val2017/)")
        return True

    ok = download(ds["url"], ds["archive"], ds["min_size"])
    if not ok:
        return False

    # Extracts to .../images/val2017/ directly
    ok = extract_zip(ds["archive"], ds["dest"])
    if not ok:
        return False

    n = sum(1 for _ in (COCO_DIR / "images" / "val2017").glob("*.jpg"))
    log(f"[ok]    {n:,} images extracted")
    return True


def setup_annotations() -> bool:
    ds = DATASETS["annotations"]
    print("\n━━━  COCO 2017 annotations  ━━━")

    target = COCO_DIR / "annotations" / "person_keypoints_val2017.json"
    if target.exists() and target.stat().st_size > 1_000_000:
        log(f"[skip]  {target.name} already present ({target.stat().st_size:,} bytes)")
        return True

    ok = download(ds["url"], ds["archive"], ds["min_size"])
    if not ok:
        return False

    # Inspect the zip to find the exact inner path before extracting
    log("[scan]  listing zip contents …")
    try:
        with zipfile.ZipFile(ds["archive"]) as zf:
            names = zf.namelist()
    except zipfile.BadZipFile as exc:
        log(f"[FAIL]  bad zip: {exc}")
        return False

    wanted = [n for n in names if "person_keypoints_val2017" in n]
    if not wanted:
        log("[FAIL]  person_keypoints_val2017.json not found inside archive")
        log(f"        top-level entries: {names[:10]}")
        return False

    member = wanted[0]
    log(f"[found] {member}")

    # Extract just that one file flat into annotations/
    ann_dir = COCO_DIR / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(ds["archive"]) as zf:
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    except Exception as exc:
        log(f"[FAIL]  selective extraction failed: {exc}")
        return False

    log(f"[ok]    {target.name}  ({target.stat().st_size:,} bytes)")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--skip-coco-val", action="store_true",
                    help="Skip the large COCO val2017 image archive (~1 GB)")
    args = ap.parse_args()

    print("=" * 60)
    print("  PPE Guard — Phase 2 dataset download")
    print(f"  Project root : {ROOT}")
    print(f"  Downloads dir: {DOWNLOADS}")
    print("=" * 60)

    DOWNLOADS.mkdir(parents=True, exist_ok=True)

    results = {
        "coco8":       setup_coco8(),
        "val2017":     setup_val2017(args.skip_coco_val),
        "annotations": setup_annotations(),
    }

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    all_ok = True
    for name, ok in results.items():
        status = "✓ OK   " if ok else "✗ FAIL "
        print(f"  {status}  {name}")
        if not ok:
            all_ok = False

    if not all_ok:
        print("\n  One or more datasets failed. Check output above.")
        return 1
    print("\n  All datasets ready. Run:  python scripts/validate_datasets.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
