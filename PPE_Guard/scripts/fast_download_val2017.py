"""Fast multi-threaded downloader for val2017.zip using HTTP Range chunks."""
from __future__ import annotations

import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

URL = "http://images.cocodataset.org/zips/val2017.zip"
TARGET = Path(r"D:\Projects\SafeOps\PPE_Guard\datasets\downloads\val2017.zip")
TOTAL_SIZE = 815585330
NUM_WORKERS = 16


def download_chunk(start: int, end: int, chunk_idx: int, part_path: Path) -> tuple[int, int]:
    existing = part_path.stat().st_size if part_path.exists() else 0
    cur_start = start + existing
    if cur_start > end:
        return chunk_idx, part_path.stat().st_size

    req = urllib.request.Request(
        URL,
        headers={
            "Range": f"bytes={cur_start}-{end}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        with open(part_path, "ab" if existing > 0 else "wb") as f:
            while True:
                buf = resp.read(64 * 1024)
                if not buf:
                    break
                f.write(buf)
    return chunk_idx, part_path.stat().st_size


def main() -> None:
    temp_dir = TARGET.parent / "val2017_parts"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Check if target already exists with partial data from curl
    existing_target_size = TARGET.stat().st_size if TARGET.exists() else 0
    print(f"Target file: {TARGET}")
    print(f"Total size: {TOTAL_SIZE:,} bytes ({TOTAL_SIZE / 1024 / 1024:.2f} MB)")
    print(f"Existing sequential bytes: {existing_target_size:,} bytes ({existing_target_size / 1024 / 1024:.2f} MB)")

    remaining_start = existing_target_size
    remaining_bytes = TOTAL_SIZE - remaining_start

    if remaining_bytes <= 0:
        print("Target is already fully downloaded.")
        return

    chunk_size = remaining_bytes // NUM_WORKERS
    tasks = []
    part_files = []

    for i in range(NUM_WORKERS):
        start = remaining_start + (i * chunk_size)
        end = remaining_start + ((i + 1) * chunk_size - 1) if i < NUM_WORKERS - 1 else TOTAL_SIZE - 1
        part_path = temp_dir / f"part_{i:02d}.bin"
        part_files.append((part_path, start, end))
        tasks.append((start, end, i, part_path))

    print(f"Spawning {NUM_WORKERS} parallel chunk download workers...")
    t0 = time.time()
    last_print = t0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(download_chunk, *t): t[2] for t in tasks}

        while True:
            done = [f for f in futures if f.done()]
            cur_downloaded = sum(p.stat().st_size for p, _, _ in part_files if p.exists())
            total_now = existing_target_size + cur_downloaded
            elapsed = max(1.0, time.time() - t0)
            speed_mb = (cur_downloaded / 1024 / 1024) / elapsed

            if time.time() - last_print >= 5:
                print(
                    f"Progress: {total_now / 1024 / 1024:.1f}/{TOTAL_SIZE / 1024 / 1024:.1f} MB "
                    f"({total_now / TOTAL_SIZE * 100:.1f}%) | "
                    f"Speed: {speed_mb:.2f} MB/s | "
                    f"Completed workers: {len(done)}/{NUM_WORKERS}",
                    flush=True,
                )
                last_print = time.time()

            if len(done) == len(futures):
                break
            time.sleep(1)

    print("\nAll chunks downloaded. Appending chunks to target file...")
    with open(TARGET, "ab" if existing_target_size > 0 else "wb") as dst:
        for p, _, _ in part_files:
            with open(p, "rb") as src:
                while True:
                    buf = src.read(1024 * 1024)
                    if not buf:
                        break
                    dst.write(buf)

    final_size = TARGET.stat().st_size
    print(f"Assembly complete! Final size: {final_size:,} bytes (expected {TOTAL_SIZE:,})")

    if final_size == TOTAL_SIZE:
        print("Cleaning up temporary chunk files...")
        for p, _, _ in part_files:
            if p.exists():
                p.unlink()
        if temp_dir.exists():
            try:
                temp_dir.rmdir()
            except Exception:
                pass
        print("Download and assembly verified 100% complete!")
    else:
        print(f"WARNING: Size mismatch ({final_size} != {TOTAL_SIZE})")


if __name__ == "__main__":
    main()

