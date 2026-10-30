"""
debug_yolo_webcam.py

A debugging-focused YOLOv8 webcam pipeline.

What this script emphasizes:
- Clear visual overlays for internal state
- Verbose console logs for each frame and each detection
- Easy slow/step modes for frame-by-frame inspection

Usage examples:
  py -3 debug_yolo_webcam.py
  py -3 debug_yolo_webcam.py --source 0 --conf 0.45
  py -3 debug_yolo_webcam.py --delay-ms 200
  py -3 debug_yolo_webcam.py --step-mode
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Tuple

import cv2
from ultralytics import YOLO


WINDOW_NAME = "YOLOv8 Debug Pipeline"


def open_camera(source: int) -> cv2.VideoCapture:
    # Try DirectShow first on Windows for more stable webcam behavior.
    candidates = [
        ("DSHOW", cv2.CAP_DSHOW),
        ("MSMF", cv2.CAP_MSMF),
        ("ANY", cv2.CAP_ANY),
    ]
    for name, backend in candidates:
        cap = cv2.VideoCapture(source, backend)
        if cap.isOpened():
            print(f"[INIT] Camera opened with backend={name}")
            return cap
        cap.release()
    return cv2.VideoCapture(source)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 real-time debug pipeline")
    parser.add_argument("--source", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 model path")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--delay-ms", type=int, default=1, help="Extra delay per frame in ms for slow mode")
    parser.add_argument(
        "--step-mode",
        action="store_true",
        help="Pause on each frame and wait for key press (frame-by-frame debugging)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size for YOLO",
    )
    return parser.parse_args()


def draw_text_block(
    frame,
    text: str,
    org: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    bg: Tuple[int, int, int] = (0, 0, 0),
    scale: float = 0.6,
    thickness: int = 2,
) -> None:
    """Draw readable text with a filled background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    cv2.rectangle(frame, (x - 4, y - th - baseline - 4), (x + tw + 4, y + 4), bg, -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def main() -> None:
    args = parse_args()

    # 1) Load model
    print(f"[INIT] Loading YOLO model: {args.model}")
    try:
        model = YOLO(args.model)
    except Exception as exc:
        print(f"[ERROR] Failed to load model '{args.model}': {exc}")
        sys.exit(1)

    # 2) Open camera
    print(f"[INIT] Opening camera source: {args.source}")
    cap = open_camera(args.source)
    if not cap.isOpened():
        print(f"[ERROR] Camera open failed for source {args.source}. Exiting.")
        sys.exit(1)

    print("[OK] Camera opened successfully.")
    print("[INFO] Press 'q' or ESC to quit.")
    print("[INFO] Press 'p' to toggle step mode during runtime.")

    frame_id = 0
    prev_time = time.perf_counter()
    fps = 0.0
    step_mode = args.step_mode

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[ERROR] Failed to read frame from camera. Stopping.")
            break

        frame_id += 1
        print(f"[FRAME {frame_id}] Frame received")

        # 3) Inference
        t0 = time.perf_counter()
        results = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
        t1 = time.perf_counter()

        result = results[0]
        boxes = result.boxes
        det_count = 0 if boxes is None else len(boxes)

        print(f"[FRAME {frame_id}] Processed in {(t1 - t0) * 1000:.1f} ms | detections={det_count}")

        # 4) Draw detection boxes + labels
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().item())
                cls_id = int(boxes.cls[i].cpu().item())
                cls_name = result.names.get(cls_id, str(cls_id))

                x1, y1, x2, y2 = xyxy.tolist()

                # Green for person, orange for non-person
                color = (0, 220, 0) if cls_name == "person" else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name} {conf:.2f}"
                draw_text_block(frame, label, (x1, max(20, y1 - 8)), color=(255, 255, 255), bg=color)

                print(f"  -> det#{i + 1}: class={cls_name}, conf={conf:.3f}, box=({x1},{y1},{x2},{y2})")

            status_text = "RUNNING"
            status_color = (0, 220, 0)
        else:
            status_text = "NO DETECTION"
            status_color = (0, 0, 255)
            draw_text_block(
                frame,
                "NO OBJECT DETECTED",
                (40, frame.shape[0] - 30),
                color=(255, 255, 255),
                bg=(0, 0, 200),
                scale=0.8,
                thickness=2,
            )

        # 5) Compute FPS (instantaneous with smoothing)
        now = time.perf_counter()
        dt = max(now - prev_time, 1e-6)
        inst_fps = 1.0 / dt
        fps = inst_fps if frame_id == 1 else (0.9 * fps + 0.1 * inst_fps)
        prev_time = now

        # 6) Debug overlays
        draw_text_block(frame, "CAMERA ACTIVE", (20, 30), color=(0, 0, 0), bg=(0, 255, 255), scale=0.7)
        draw_text_block(frame, f"FPS: {fps:.1f}", (20, 60), color=(255, 255, 255), bg=(50, 50, 50))
        draw_text_block(frame, f"Detections: {det_count}", (20, 90), color=(255, 255, 255), bg=(50, 50, 50))
        draw_text_block(
            frame,
            f"Status: {status_text}",
            (20, 120),
            color=(255, 255, 255),
            bg=status_color,
        )
        draw_text_block(frame, f"Frame: {frame_id}", (20, 150), color=(255, 255, 255), bg=(50, 50, 50))

        mode_label = "STEP MODE" if step_mode else f"SLOW MODE ({args.delay_ms} ms)"
        draw_text_block(frame, mode_label, (20, 180), color=(255, 255, 255), bg=(120, 80, 0))

        cv2.imshow(WINDOW_NAME, frame)

        # 7) Slow mode / step mode controls
        if step_mode:
            key = cv2.waitKey(0) & 0xFF
            # In step mode, any key advances one frame; q/esc exits.
        else:
            key = cv2.waitKey(max(1, args.delay_ms)) & 0xFF

        if key in (ord("q"), 27):
            print("[INFO] Quit key received. Exiting.")
            break
        if key == ord("p"):
            step_mode = not step_mode
            print(f"[INFO] Step mode toggled: {step_mode}")

    cap.release()
    cv2.destroyAllWindows()
    print("[DONE] Camera released and windows closed.")


if __name__ == "__main__":
    main()
