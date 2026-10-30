#!/usr/bin/env python3
"""
mine_unknown_cases.py

Run the current PPE pipeline on images and videos and save only UNKNOWN cases.

Why this exists:
  - Hard-example mining for the current model's blind spots
  - Unknown crops can be manually labeled and fed back into training

Outputs:
  output/
    unknown/
      frames/
      crops/
      manifest.csv

Examples:
    python mine_unknown_cases.py --source ./videos --output ./hard_examples
    python mine_unknown_cases.py --source ./samples --output ./hard_examples --save-frame --save-crop
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import cv2

import config
from detection import PersonDetector
from ppe_detection import PPEDetector
from tracking import PersonTracker


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}


def iter_sources(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    files: list[Path] = []
    for child in sorted(path.rglob("*")):
        if not child.is_file():
            continue
        suffix = child.suffix.lower()
        if suffix in IMAGE_EXTENSIONS or suffix in VIDEO_EXTENSIONS:
            files.append(child)
    return files


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def ensure_output_dirs(output_root: Path) -> tuple[Path, Path]:
    unknown_root = output_root / "unknown"
    frames_dir = unknown_root / "frames"
    crops_dir = unknown_root / "crops"
    frames_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    return frames_dir, crops_dir


def classify_unknown_tracks(tracks) -> list:
    return [track for track in tracks if track.has_helmet is None]


def repeated_updates(tracker: PersonTracker, detections, frame, count: int) -> list:
    tracks = []
    for _ in range(max(1, count)):
        tracks = tracker.update(detections, frame)
    return tracks


def save_unknown_example(
    *,
    source_path: Path,
    frame_index: int,
    frame,
    track,
    frames_dir: Path,
    crops_dir: Path,
    save_frame: bool,
    save_crop: bool,
) -> dict:
    stem = source_path.stem
    tag = f"{stem}_f{frame_index:06d}_track{track.track_id}"
    record = {
        "source": str(source_path),
        "frame_index": frame_index,
        "track_id": track.track_id,
        "saved_frame": "no",
        "saved_crop": "no",
        "frame_path": "",
        "crop_path": "",
    }

    if save_frame:
        frame_path = frames_dir / f"{tag}.jpg"
        cv2.imwrite(str(frame_path), frame)
        record["saved_frame"] = "yes"
        record["frame_path"] = str(frame_path)

    if save_crop:
        crop = track.crop_frame(frame)
        crop_path = crops_dir / f"{tag}.jpg"
        cv2.imwrite(str(crop_path), crop)
        record["saved_crop"] = "yes"
        record["crop_path"] = str(crop_path)

    return record


def process_image(
    *,
    image_path: Path,
    person_detector: PersonDetector,
    ppe_detector: PPEDetector,
    save_frame: bool,
    save_crop: bool,
    frames_dir: Path,
    crops_dir: Path,
    tracker_init_frames: int,
) -> list[dict]:
    frame = cv2.imread(str(image_path))
    if frame is None:
        return []

    tracker = PersonTracker()
    tracker.reset()
    detections = person_detector.detect(frame)
    tracks = repeated_updates(tracker, detections, frame, tracker_init_frames)
    ppe_detector.process_tracks(tracks, frame)

    records: list[dict] = []
    for track in classify_unknown_tracks(tracks):
        records.append(
            save_unknown_example(
                source_path=image_path,
                frame_index=0,
                frame=frame,
                track=track,
                frames_dir=frames_dir,
                crops_dir=crops_dir,
                save_frame=save_frame,
                save_crop=save_crop,
            )
        )
    return records


def process_video(
    *,
    video_path: Path,
    person_detector: PersonDetector,
    ppe_detector: PPEDetector,
    save_frame: bool,
    save_crop: bool,
    frames_dir: Path,
    crops_dir: Path,
    tracker_init_frames: int,
    max_frames: int | None,
) -> list[dict]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []

    tracker = PersonTracker()
    records: list[dict] = []
    frame_index = 0

    while True:
        success, frame = capture.read()
        if not success:
            break
        if max_frames is not None and frame_index >= max_frames:
            break

        detections = person_detector.detect(frame)
        tracks = tracker.update(detections, frame)
        ppe_detector.process_tracks(tracks, frame)

        for track in classify_unknown_tracks(tracks):
            records.append(
                save_unknown_example(
                    source_path=video_path,
                    frame_index=frame_index,
                    frame=frame,
                    track=track,
                    frames_dir=frames_dir,
                    crops_dir=crops_dir,
                    save_frame=save_frame,
                    save_crop=save_crop,
                )
            )

        frame_index += 1

    capture.release()
    return records


def write_manifest(records: Iterable[dict], manifest_path: Path) -> None:
    fieldnames = ["source", "frame_index", "track_id", "saved_frame", "saved_crop", "frame_path", "crop_path"]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine UNKNOWN PPE examples from images or videos")
    parser.add_argument("--source", required=True, help="Input file or directory containing images/videos")
    parser.add_argument("--output", required=True, help="Output directory for hard examples")
    parser.add_argument("--person-model", default=str(Path(__file__).resolve().parent / "yolov8n.pt"), help="Person detector model path")
    parser.add_argument("--device", default=None, help="Inference device override")
    parser.add_argument("--save-frame", action="store_true", help="Save full frames for UNKNOWN cases")
    parser.add_argument("--save-crop", action="store_true", help="Save person crops for UNKNOWN cases")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames processed per video")
    parser.add_argument("--tracker-init-frames", type=int, default=config.DEEPSORT_N_INIT, help="Number of repeated updates used to confirm static-image tracks")
    args = parser.parse_args()

    source_root = Path(args.source).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    frames_dir, crops_dir = ensure_output_dirs(output_root)

    if not args.save_frame and not args.save_crop:
        args.save_crop = True

    sources = iter_sources(source_root)
    if not sources:
        print(f"No images or videos found at: {source_root}")
        return 1

    person_detector = PersonDetector(
        model_path=args.person_model,
        conf_threshold=config.PERSON_CONF_THRESHOLD,
        iou_threshold=config.PERSON_IOU_THRESHOLD,
        device=args.device,
    )
    ppe_detector = PPEDetector(device=args.device)

    print(f"Source root: {source_root}")
    print(f"Output root: {output_root}")
    print(f"Using PPE model: {ppe_detector.model_path}")
    print(f"PPE stub mode: {ppe_detector.is_stub}")

    all_records: list[dict] = []
    for source_path in sources:
        print(f"Processing: {source_path}")
        if is_image(source_path):
            records = process_image(
                image_path=source_path,
                person_detector=person_detector,
                ppe_detector=ppe_detector,
                save_frame=args.save_frame,
                save_crop=args.save_crop,
                frames_dir=frames_dir,
                crops_dir=crops_dir,
                tracker_init_frames=args.tracker_init_frames,
            )
        elif is_video(source_path):
            records = process_video(
                video_path=source_path,
                person_detector=person_detector,
                ppe_detector=ppe_detector,
                save_frame=args.save_frame,
                save_crop=args.save_crop,
                frames_dir=frames_dir,
                crops_dir=crops_dir,
                tracker_init_frames=args.tracker_init_frames,
                max_frames=args.max_frames,
            )
        else:
            records = []

        all_records.extend(records)
        print(f"  UNKNOWN examples saved: {len(records)}")

    manifest_path = output_root / "unknown" / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(all_records, manifest_path)
    print(f"Manifest written to: {manifest_path}")
    print(f"Total UNKNOWN examples: {len(all_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
