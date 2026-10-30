"""SafeOps Production Video Processing Pipeline.

Coordinates:
1. Streaming Video Ingestion (OpenCV)
2. Worker Detection & Pose Tracking (YOLOv8s-Pose + ByteTrack)
3. Modular PPE Detection (Helmet + Vest Interface)
4. Keypoint-Guided PPE Association
5. Temporal Observation Smoothing
6. Safety Policy Compliance Evaluation
7. Violation Event Lifecycle Tracking
8. High-Resolution Visual Overlays & MP4 Generation
9. Output Contract Generation (tracks.jsonl + report.json)
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from src.association.ppe_association import associate_ppe_to_workers
from src.compliance.compliance_engine import ComplianceEngine
from src.compliance.violation_manager import ViolationManager
from src.contracts.schemas import (
    ComplianceStatus,
    PPEItemType,
    ProcessingSummary,
    SafetyPolicyConfig,
    VideoAnalysisReport,
    VideoMetadata,
    WorkerComplianceState,
    WorkerSummary,
    WorkerTrackingRecord,
)

# Backward-compatibility alias
VideoInfo = VideoMetadata


def _color(track_id: int):
    return ((track_id * 67) % 200 + 55, (track_id * 131) % 200 + 55, (track_id * 29) % 200 + 55)


def _round(vals, nd=1):
    if vals is None:
        return None
    return [[round(v, nd) for v in p] if isinstance(p, list) else round(p, nd) for p in vals]


def annotate(frame, workers, compliance=None, kp_thresh: float = 0.4):
    ann = VideoAnnotator(kp_thresh=kp_thresh)
    return ann.annotate_frame(frame, workers, compliance_states={}, associations={})
from src.ppe.detector import ModularPPEEngine
from src.tracking.worker_tracker import WorkerTracker
from src.video.annotator import VideoAnnotator
from src.video.video_stream import open_video_stream


def run_pipeline(
    video_path: str,
    out_video: str = "outputs/tracked_video.mp4",
    out_jsonl: str = "outputs/tracks.jsonl",
    out_report: Optional[str] = "outputs/report.json",
    tracker: Optional[WorkerTracker] = None,
    ppe_engine: Optional[ModularPPEEngine] = None,
    compliance_engine: Optional[ComplianceEngine] = None,
    policy: Optional[SafetyPolicyConfig] = None,
    max_frames: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    log_every: int = 50,
) -> VideoAnalysisReport:
    """Execute the end-to-end SafeOps PPE monitoring pipeline on a video file."""
    t_start = time.perf_counter()

    # Initialize components if not provided
    active_tracker = tracker or WorkerTracker()
    active_ppe = ppe_engine or ModularPPEEngine()
    active_policy = policy or SafetyPolicyConfig()
    active_compliance = compliance_engine or ComplianceEngine(policy=active_policy)
    violation_manager = ViolationManager(
        recovery_confirmation_seconds=active_policy.recovery_confirmation_seconds
    )
    annotator = VideoAnnotator()

    # Reset states for clean video run
    active_tracker.reset()
    active_compliance.reset()
    violation_manager.reset()

    # Ensure output directories exist
    Path(out_video).parent.mkdir(parents=True, exist_ok=True)
    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    if out_report:
        Path(out_report).parent.mkdir(parents=True, exist_ok=True)

    with open_video_stream(video_path) as (frame_stream, meta):
        print(f"=== SafeOps Video Processing ===")
        print(f"Video: {meta.filename} ({meta.width}x{meta.height} @ {meta.fps} FPS)")
        print(f"Duration: {meta.duration_seconds}s | Total Frames: {meta.total_frames}")

        writer = cv2.VideoWriter(
            str(out_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            meta.fps,
            (meta.width, meta.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for {out_video}")

        target_frames = min(meta.total_frames, max_frames) if max_frames else meta.total_frames
        frame_count = 0
        total_worker_records = 0
        workers_seen: Dict[int, Dict[str, Any]] = {}
        last_timestamp = 0.0

        try:
            with open(out_jsonl, "w", encoding="utf-8") as jf:
                for frame_number, timestamp, frame in frame_stream:
                    if max_frames is not None and frame_number >= max_frames:
                        break

                    last_timestamp = timestamp

                    # 1. Detect & Track Workers (COCO Person Class 0)
                    workers = active_tracker.update(frame, frame_number=frame_number)

                    # 2. Detect PPE
                    raw_ppe_dict = active_ppe.detect_all(frame)
                    all_ppe_dets = raw_ppe_dict[PPEItemType.HELMET] + raw_ppe_dict[PPEItemType.VEST]

                    # 3. Associate PPE with Workers
                    associations, unassociated = associate_ppe_to_workers(
                        workers=workers,
                        ppe_detections=all_ppe_dets,
                        has_vest_model=active_ppe.has_vest_model,
                    )

                    # 4. Evaluate Compliance & Violations per Worker
                    frame_compliance_states: Dict[int, WorkerComplianceState] = {}
                    for w in workers:
                        tid = w["track_id"]
                        assoc = associations[tid]
                        c_state = active_compliance.evaluate_worker(assoc, timestamp)
                        frame_compliance_states[tid] = c_state

                        # Process violation events
                        violation_manager.process_worker_compliance(c_state, frame_number, timestamp)

                        # Track worker summary stats
                        if tid not in workers_seen:
                            workers_seen[tid] = {
                                "first_seen": timestamp,
                                "last_seen": timestamp,
                                "frames": 0,
                                "final_status": c_state.overall_status,
                                "violations": 0,
                            }
                        workers_seen[tid]["last_seen"] = timestamp
                        workers_seen[tid]["frames"] += 1
                        workers_seen[tid]["final_status"] = c_state.overall_status

                        # 5. Write structured tracking record to JSONL
                        rec = {
                            "frame_number": frame_number,
                            "timestamp_seconds": timestamp,
                            "track_id": tid,
                            "bbox_xyxy": [round(v, 1) for v in w["bbox_xyxy"]],
                            "confidence": round(w["confidence"], 4),
                            "keypoints_xy": w.get("keypoints_xy"),
                            "keypoints_conf": w.get("keypoints_conf"),
                            "continuity_state": w.get("continuity_state", "confirmed"),
                            "ppe": {
                                "helmet": assoc.helmet.status == "detected",
                                "vest": assoc.vest.status == "detected",
                                "helmet_status": assoc.helmet.status.value,
                                "vest_status": assoc.vest.status.value,
                                "helmet_conf": assoc.helmet.confidence,
                                "vest_conf": assoc.vest.confidence,
                            },
                            "compliance": {
                                "overall_status": c_state.overall_status.value,
                                "active_violation": c_state.active_violation,
                                "details": c_state.details,
                            },
                        }
                        jf.write(json.dumps(rec) + "\n")
                        total_worker_records += 1

                    # 6. Annotate Frame & Write Video
                    active_viols = sum(1 for cs in frame_compliance_states.values() if cs.active_violation)
                    annotated_frame = annotator.annotate_frame(
                        frame=frame,
                        workers=workers,
                        compliance_states=frame_compliance_states,
                        associations=associations,
                        raw_ppe_detections=all_ppe_dets,
                        active_violations_count=active_viols,
                    )
                    writer.write(annotated_frame)

                    frame_count += 1
                    if frame_count % log_every == 0 or frame_count == target_frames:
                        pct = (frame_count / target_frames * 100.0) if target_frames else 0.0
                        print(f"Processed {frame_count}/{target_frames} frames ({pct:.1f}%)")
                        if progress_callback:
                            progress_callback(frame_count, target_frames, pct)

        finally:
            writer.release()

    t_end = time.perf_counter()
    elapsed = max(t_end - t_start, 1e-6)
    avg_fps = frame_count / elapsed

    # Finalize any ongoing violation events
    all_violations = violation_manager.finalize_stream(last_timestamp)

    # Compute worker summaries
    worker_summaries: List[WorkerSummary] = []
    compliant_workers_count = 0
    for tid, s in sorted(workers_seen.items()):
        v_count = sum(1 for v in all_violations if v.track_id == tid)
        is_compliant = s["final_status"] == ComplianceStatus.COMPLIANT
        if is_compliant:
            compliant_workers_count += 1
        worker_summaries.append(
            WorkerSummary(
                track_id=tid,
                first_seen_timestamp=s["first_seen"],
                last_seen_timestamp=s["last_seen"],
                total_frames_present=s["frames"],
                compliance_status=s["final_status"],
                violations_count=v_count,
            )
        )

    compliance_rate = (compliant_workers_count / len(workers_seen) * 100.0) if workers_seen else 100.0

    summary = ProcessingSummary(
        total_frames_processed=frame_count,
        duration_seconds=round(last_timestamp, 3),
        average_fps=round(avg_fps, 2),
        unique_workers_tracked=len(workers_seen),
        total_violations_detected=len(all_violations),
        compliance_rate_percent=round(compliance_rate, 2),
    )

    report = VideoAnalysisReport(
        version="1.0.0",
        video_id=meta.video_id,
        metadata=meta,
        policy=active_policy,
        summary=summary,
        workers=worker_summaries,
        violations=all_violations,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    if out_report:
        with open(out_report, "w", encoding="utf-8") as rf:
            rf.write(report.model_dump_json(indent=2))

    print(f"\nPROCESSING COMPLETED in {elapsed:.1f}s ({avg_fps:.1f} FPS)")
    print(f"Unique Workers: {summary.unique_workers_tracked} | Total Violations: {summary.total_violations_detected}")
    print(f"Track Records: {total_worker_records}")
    print(f"Output Video:  {out_video}")
    print(f"Output JSONL:  {out_jsonl}")
    if out_report:
        print(f"Output Report: {out_report}")

    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="SafeOps Production PPE Compliance Pipeline")
    ap.add_argument("--video", default="videos/test_video.mp4")
    ap.add_argument("--pose-model", default="models/yolov8s-pose.pt")
    ap.add_argument("--tracker", default="src/tracking/custom_bytetrack.yaml")
    ap.add_argument("--ppe-model", default="models/ppe_helmet.pt")
    ap.add_argument("--out-video", default="outputs/tracked_video.mp4")
    ap.add_argument("--out-jsonl", default="outputs/tracks.jsonl")
    ap.add_argument("--out-report", default="outputs/report.json")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--require-helmet", action="store_true", default=True)
    ap.add_argument("--require-vest", action="store_true", default=False)
    a = ap.parse_args()

    tracker = WorkerTracker(model_path=a.pose_model, tracker=a.tracker, conf=a.conf, imgsz=a.imgsz)
    ppe_engine = ModularPPEEngine(helmet_model_path=a.ppe_model)
    policy = SafetyPolicyConfig(require_helmet=a.require_helmet, require_vest=a.require_vest)

    run_pipeline(
        video_path=a.video,
        out_video=a.out_video,
        out_jsonl=a.out_jsonl,
        out_report=a.out_report,
        tracker=tracker,
        ppe_engine=ppe_engine,
        policy=policy,
        max_frames=a.max_frames,
    )


if __name__ == "__main__":
    main()
