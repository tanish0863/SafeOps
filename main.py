#!/usr/bin/env python3
"""
main.py
-------
Command-line entry point for the Intelligent Camera Quality Guardian.

Usage examples
--------------
    # Analyse a single image
    python main.py image samples/frame.jpg

    # Analyse a video file (every 30th frame)
    python main.py video samples/clip.mp4 --interval 30

    # Run a live webcam check (device 0, capture 90 frames)
    python main.py webcam --device 0 --frames 90

    # Analyse an image and display the result (requires matplotlib)
    python main.py image samples/frame.jpg --show

    # Run all tests
    python main.py test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_image(args: argparse.Namespace) -> int:
    from quality_guardian.image_processor import process_image, print_result_summary
    from quality_guardian.assessment import AssessmentConfig

    config = AssessmentConfig()

    try:
        result, json_path, annotated_path = process_image(
            image_path=args.path,
            config=config,
            save_report=not args.no_report,
            save_annotated=not args.no_annotated,
            show_display=args.show,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print_result_summary(result, source=str(args.path))

    if json_path:
        print(f"[INFO] JSON report saved  : {json_path}")
    if annotated_path:
        print(f"[INFO] Annotated image saved : {annotated_path}")

    # Return exit code based on status
    return {"GOOD": 0, "WARNING": 1, "POOR": 2}.get(result.overall_status, 1)


def _cmd_video(args: argparse.Namespace) -> int:
    from quality_guardian.video_processor import process_video, print_video_summary
    from quality_guardian.assessment import AssessmentConfig

    config = AssessmentConfig()

    try:
        frame_results, summary, report_path = process_video(
            video_path=args.path,
            config=config,
            sample_interval=args.interval,
            save_report=not args.no_report,
            save_annotated_frames=args.save_frames,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print_video_summary(summary, source=str(args.path))

    if report_path:
        print(f"[INFO] Video report saved : {report_path}")

    status = summary.get("overall_video_status", "POOR")
    return {"GOOD": 0, "WARNING": 1, "POOR": 2}.get(status, 1)


def _cmd_webcam(args: argparse.Namespace) -> int:
    from quality_guardian.video_processor import process_webcam, print_video_summary
    from quality_guardian.assessment import AssessmentConfig

    config = AssessmentConfig()

    print(f"[INFO] Opening webcam device {args.device} …")

    try:
        frame_results, summary, report_path = process_webcam(
            device_index=args.device,
            config=config,
            max_frames=args.frames,
            sample_interval=args.interval,
            save_report=not args.no_report,
        )
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print_video_summary(summary, source=f"webcam_{args.device}")

    if report_path:
        print(f"[INFO] Webcam report saved : {report_path}")

    status = summary.get("overall_video_status", "POOR")
    return {"GOOD": 0, "WARNING": 1, "POOR": 2}.get(status, 1)


def _cmd_test(_args: argparse.Namespace) -> int:
    """Run the pytest test suite."""
    try:
        import pytest
    except ImportError:
        print("[ERROR] pytest is not installed. Run: pip install pytest", file=sys.stderr)
        return 1

    test_dir = Path(__file__).parent / "tests"
    return pytest.main([str(test_dir), "-v"])


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="camera-quality-guardian",
        description="Intelligent Camera Quality Guardian – PPE-ready frame analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- image ---
    img_p = sub.add_parser("image", help="Analyse a single image file")
    img_p.add_argument("path", type=str, help="Path to image (.jpg, .jpeg, .png)")
    img_p.add_argument("--show", action="store_true", help="Display result with matplotlib")
    img_p.add_argument("--no-report", action="store_true", help="Skip saving the JSON report")
    img_p.add_argument("--no-annotated", action="store_true", help="Skip saving the annotated image")
    img_p.set_defaults(func=_cmd_image)

    # --- video ---
    vid_p = sub.add_parser("video", help="Analyse a recorded video file")
    vid_p.add_argument("path", type=str, help="Path to video (.mp4, .avi)")
    vid_p.add_argument(
        "--interval", type=int, default=30,
        help="Analyse every Nth frame (default: 30)",
    )
    vid_p.add_argument("--save-frames", action="store_true",
                       help="Save annotated images for problematic frames")
    vid_p.add_argument("--no-report", action="store_true", help="Skip saving the JSON report")
    vid_p.set_defaults(func=_cmd_video)

    # --- webcam ---
    wc_p = sub.add_parser("webcam", help="Run a live quality check using a webcam")
    wc_p.add_argument("--device", type=int, default=0,
                      help="Webcam device index (default: 0)")
    wc_p.add_argument("--frames", type=int, default=150,
                      help="Number of raw frames to capture (default: 150)")
    wc_p.add_argument(
        "--interval", type=int, default=15,
        help="Analyse every Nth frame (default: 15)",
    )
    wc_p.add_argument("--no-report", action="store_true", help="Skip saving the JSON report")
    wc_p.set_defaults(func=_cmd_webcam)

    # --- test ---
    tst_p = sub.add_parser("test", help="Run the pytest test suite")
    tst_p.set_defaults(func=_cmd_test)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
