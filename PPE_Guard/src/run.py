"""CLI: python -m src.run --input data/input --output data/output [--model models/yolo26n-pose.pt]"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from .pipeline import Phase2Pipeline, write_jsonl_line
from .visualize import draw

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/input")
    ap.add_argument("--output", default="data/output")
    ap.add_argument("--model", default="models/yolo26n-pose.pt")
    ap.add_argument("--conf", type=float, default=0.35, help="person confidence")
    ap.add_argument("--kp-conf", type=float, default=0.5, help="keypoint confidence")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth video frame")
    ap.add_argument("--save-video", action="store_true", help="write annotated video")
    a = ap.parse_args()

    inp, out = Path(a.input), Path(a.output)
    (out / "annotated").mkdir(parents=True, exist_ok=True)
    pipe = Phase2Pipeline(a.model, conf=a.conf, keypoint_conf=a.kp_conf)
    print(f"device: {pipe.engine.device}")

    files = sorted(inp.iterdir()) if inp.is_dir() else [inp]
    images = [f for f in files if f.suffix.lower() in IMG_EXT]
    videos = [f for f in files if f.suffix.lower() in VID_EXT]

    if images:
        with open(out / "images.jsonl", "w", encoding="utf-8") as fh:
            for i, f in enumerate(images):
                frame = cv2.imread(str(f))
                if frame is None:
                    print(f"skip unreadable: {f}")
                    continue
                rec = pipe.process_frame(frame, frame_number=i)
                write_jsonl_line(fh, rec)
                cv2.imwrite(str(out / "annotated" / f"{f.stem}_annotated.jpg"), draw(frame, rec))
                print(f"{f.name}: {len(rec.persons)} person(s)")

    for v in videos:
        t0, n, writer = time.time(), 0, None
        with open(out / f"{v.stem}.jsonl", "w", encoding="utf-8") as fh:
            for frame, rec in pipe.process_video(v, a.stride):
                write_jsonl_line(fh, rec)
                n += 1
                if a.save_video:
                    if writer is None:
                        h, w = frame.shape[:2]
                        writer = cv2.VideoWriter(str(out / f"{v.stem}_annotated.mp4"),
                                                 cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
                    writer.write(draw(frame, rec))
        if writer:
            writer.release()
        dt = max(time.time() - t0, 1e-9)
        print(f"{v.name}: {n} frames in {dt:.1f}s ({n / dt:.1f} fps measured, incl. I/O)")


if __name__ == "__main__":
    main()
