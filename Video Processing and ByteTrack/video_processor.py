"""Phase 3: video -> tracked workers (YOLOv8 Pose + ByteTrack) -> annotated video + JSONL.

Run from the project root:
    python -m src.video.video_processor --video videos/test_video.mp4
    python -m src.video.video_processor --video videos/test_video.mp4 --ppe-model models/ppe_helmet.pt
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from src.association.ppe_association import PPEModel, associate_ppe
from src.tracking.worker_tracker import WorkerTracker

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]


@dataclass
class VideoInfo:
    fps: float
    width: int
    height: int
    total_frames: int

    @property
    def duration(self) -> float:
        return self.total_frames / self.fps if self.fps else 0.0


def read_video_info(cap: cv2.VideoCapture) -> VideoInfo:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 0:
        print("WARNING: video reports no FPS; assuming 30.0 (timestamps will be approximate)")
        fps = 30.0
    return VideoInfo(
        fps=float(fps),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )


def _color(track_id: int) -> tuple[int, int, int]:
    return ((track_id * 67) % 200 + 55, (track_id * 131) % 200 + 55, (track_id * 29) % 200 + 55)


def annotate(frame, workers, compliance=None, kp_thresh: float = 0.5) -> None:
    """Draw boxes, track IDs, skeletons and (optionally) PPE status in place."""
    for w in workers:
        tid = w["track_id"]
        color = _color(tid)
        x1, y1, x2, y2 = map(int, w["bbox_xyxy"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"ID {tid}"
        if compliance and tid in compliance:
            c = compliance[tid]
            label += f" H:{'Y' if c['helmet'] else 'N'} V:{'Y' if c['vest'] else 'N'}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        top = max(y1, th + 8)
        cv2.rectangle(frame, (x1, top - th - 8), (x1 + tw + 6, top), color, -1)
        cv2.putText(frame, label, (x1 + 3, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)

        kxy, kc = w.get("keypoints_xy"), w.get("keypoints_conf")
        if kxy is None or kc is None:
            continue
        for a, b in COCO_SKELETON:
            if kc[a] >= kp_thresh and kc[b] >= kp_thresh:
                cv2.line(frame, tuple(map(int, kxy[a])), tuple(map(int, kxy[b])), color, 2)
        for i, (x, y) in enumerate(kxy):
            if kc[i] >= kp_thresh:
                cv2.circle(frame, (int(x), int(y)), 3, color, -1)


def _round(vals, nd=1):
    if vals is None:
        return None
    return [[round(v, nd) for v in p] if isinstance(p, list) else round(p, nd) for p in vals]


def process_video(
    video_path: str,
    out_video: str,
    out_jsonl: str,
    tracker: WorkerTracker,
    ppe_model: PPEModel | None = None,
    max_frames: int | None = None,
    log_every: int = 50,
) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    info = read_video_info(cap)

    print("VIDEO INFORMATION\n----------------------")
    print(f"FPS: {info.fps}\nWidth: {info.width}\nHeight: {info.height}")
    print(f"Total frames: {info.total_frames}\nDuration: {info.duration:.1f} seconds\n")

    Path(out_video).parent.mkdir(parents=True, exist_ok=True)
    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), info.fps,
                             (info.width, info.height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video writer for {out_video}")

    tracker.reset()
    total = info.total_frames
    if max_frames:
        total = min(total, max_frames) if total else max_frames
    frame_number, n_records, t0 = 0, 0, time.time()
    try:
        with open(out_jsonl, "w", encoding="utf-8") as jf:
            while True:
                if max_frames is not None and frame_number >= max_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp = frame_number / info.fps  # zero-based frame index

                # Sequential on purpose: pose/track first, then PPE (keeps VRAM low on 6GB).
                workers = tracker.update(frame)
                compliance = None
                if ppe_model is not None:
                    compliance = associate_ppe(workers, ppe_model.detect(frame))

                for w in workers:
                    rec = {
                        "frame_number": frame_number,
                        "timestamp_seconds": round(timestamp, 3),
                        "track_id": w["track_id"],
                        "bbox_xyxy": [round(v, 1) for v in w["bbox_xyxy"]],
                        "confidence": round(w["confidence"], 4),
                        "keypoints_xy": _round(w["keypoints_xy"]),
                        "keypoints_conf": _round(w["keypoints_conf"], 3),
                    }
                    if compliance is not None:
                        rec["ppe"] = compliance[w["track_id"]]
                    jf.write(json.dumps(rec) + "\n")
                    n_records += 1

                annotate(frame, workers, compliance)
                writer.write(frame)

                frame_number += 1
                if frame_number % log_every == 0:
                    print(f"Processed {frame_number}/{total} frames")
    finally:
        cap.release()
        writer.release()

    elapsed = time.time() - t0
    print(f"Processed {frame_number}/{total} frames")
    print("\nVIDEO PROCESSING COMPLETED")
    print(f"{frame_number} frames in {elapsed:.1f}s ({frame_number / max(elapsed, 1e-9):.1f} FPS), "
          f"{n_records} track records")
    print(f"Video:  {out_video}\nTracks: {out_jsonl}")
    return {"frames": frame_number, "records": n_records, "elapsed": elapsed, "info": info}


def main() -> None:
    ap = argparse.ArgumentParser(description="SafeOps Phase 3: worker tracking")
    ap.add_argument("--video", default="videos/test_video.mp4")
    ap.add_argument("--pose-model", default="models/yolov8s-pose.pt")
    ap.add_argument("--tracker", default="bytetrack.yaml",
                    help="bytetrack.yaml or src/tracking/custom_bytetrack.yaml")
    ap.add_argument("--ppe-model", default=None, help="optional custom PPE weights, e.g. models/ppe_helmet.pt")
    ap.add_argument("--out-video", default="outputs/tracked_video.mp4")
    ap.add_argument("--out-jsonl", default="outputs/tracks.jsonl")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-frames", type=int, default=None)
    a = ap.parse_args()

    tracker = WorkerTracker(a.pose_model, tracker=a.tracker, conf=a.conf, imgsz=a.imgsz)
    ppe = PPEModel(a.ppe_model) if a.ppe_model else None
    process_video(a.video, a.out_video, a.out_jsonl, tracker, ppe, a.max_frames)


if __name__ == "__main__":
    main()
