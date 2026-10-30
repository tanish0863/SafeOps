"""
main.py — Industrial Safety Vision Pipeline Orchestrator.

Pipeline per-frame:
  Frame
    → [SKIP?] → pass through if frame_skip satisfied
    → PersonDetector.detect()
    → direct per-frame person records
    → ZoneChecker.check_all()           (geometric, very cheap)
    → PPEDetector.process_tracks()      (batched crop inference)
    → AlertManager.process()            (cooldown + confirmation logic)
    → visualization.compose_frame()
    → cv2.imshow()

Run:
  python main.py --source 0                  # webcam
  python main.py --source /path/to/video.mp4
  python main.py --source 0 --no-display     # headless mode
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional

import cv2
import numpy as np

import config
from detection     import PersonDetector
from tracking      import TrackedPerson
from ppe_detection import PPEDetector
from zone_logic    import ZoneChecker
from alert_manager import AlertEvent, ViolationType
from input_handler import VideoSource, VideoSourceError
import visualization as vis

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("safety_monitor.log"),
    ],
)
logger = logging.getLogger(__name__)


class SafetyPipeline:
    """
    Top-level pipeline object. Owns all sub-components and drives the main loop.

    Responsibilities:
      - Initialize all sub-components
      - Frame-skip logic (person detection only — tracking always runs)
      - FPS measurement (rolling window)
      - Alert state for banner display
      - Graceful shutdown on 'q' or source exhaustion
    """

    FPS_WINDOW = 30   # Rolling window size for FPS calculation

    @staticmethod
    def _is_file_source(source: str | int) -> bool:
        if isinstance(source, int):
            return False
        source_text = str(source).strip()
        return not source_text.isdigit()

    def __init__(
        self,
        source:     str | int,
        display:    bool = True,
        headless:   bool = False,
        profile:    str = "balanced",
    ) -> None:
        self.source  = source
        self.display = display and not headless
        self.profile = profile.lower().strip()
        accuracy_mode = self.profile in {"accuracy", "live_accuracy", "high_accuracy"}
        fast_mode = self.profile in {"fast", "lite", "speed"}
        file_source = self._is_file_source(source)

        if fast_mode:
            person_model_path = config.PERSON_MODEL_PATH
            person_conf = 0.60
            person_iou = config.PERSON_IOU_THRESHOLD
            person_input_size = 640
            ppe_conf = config.PPE_CONF_THRESHOLD
            ppe_iou = config.PPE_IOU_THRESHOLD
        elif accuracy_mode:
            person_model_path = config.PERSON_MODEL_PATH
            person_conf = config.PERSON_CONF_THRESHOLD if not file_source else 0.52
            person_iou = config.PERSON_IOU_THRESHOLD
            person_input_size = config.INFERENCE_INPUT_SIZE if not file_source else 1280
            ppe_conf = 0.08
            ppe_iou = 0.40
        else:
            person_model_path = config.PERSON_MODEL_PATH
            person_conf = config.PERSON_CONF_THRESHOLD
            person_iou = config.PERSON_IOU_THRESHOLD
            person_input_size = config.INFERENCE_INPUT_SIZE
            ppe_conf = config.PPE_CONF_THRESHOLD
            ppe_iou = config.PPE_IOU_THRESHOLD

        logger.info("=" * 60)
        logger.info("Industrial Safety Vision System — Initializing")
        logger.info(f"Pipeline profile: {self.profile}")
        logger.info("=" * 60)

        # ── Sub-component initialization ──────────────────────────────────────
        self.person_detector = PersonDetector(
            model_path=person_model_path,
            conf_threshold=person_conf,
            iou_threshold=person_iou,
            input_size=person_input_size,
        )
        self.ppe_detector    = PPEDetector(
            conf=ppe_conf,
            iou=ppe_iou,
        )
        self.zone_checker    = ZoneChecker() if getattr(config, "ENABLE_ZONE_DETECTION", False) else None

        # ── State ─────────────────────────────────────────────────────────────
        self._frame_times: Deque[float] = deque(maxlen=self.FPS_WINDOW)
        self._active_alert_msgs: List[str] = []
        self._alert_banner_ttl: int = 0        # Frames to show alert banner

        logger.info("All components initialized. Ready to process.")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Open the video source and start the processing loop."""
        try:
            with VideoSource(source=self.source) as video:
                if self.display:
                    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(config.WINDOW_NAME, config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)

                if config.SHOW_ZONE_OVERLAY and self.zone_checker:
                    logger.info(f"Zone checker loaded {len(self.zone_checker.zones)} zone(s).")

                for frame, frame_num, timestamp in video.stream():
                    t0 = time.perf_counter()
                    should_stop = self._process_frame(frame, frame_num)
                    t1 = time.perf_counter()
                    self._frame_times.append(t1 - t0)

                    if should_stop:
                        break

        except VideoSourceError as e:
            logger.error(f"Video source error: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            if self.display:
                cv2.destroyAllWindows()
            logger.info("Pipeline shut down cleanly.")

    def process_single_frame(
        self,
        frame: np.ndarray,
        frame_num: int = 0,
    ) -> tuple[np.ndarray, List[TrackedPerson], List[AlertEvent]]:
        """
        Process a single frame and return (annotated_frame, tracks, alerts).
        Useful for unit testing and integration into other systems.
        """
        tracks, alerts = self._run_pipeline(frame, frame_num)
        fps = self._compute_fps()
        output = frame.copy()
        if config.SHOW_ZONE_OVERLAY and self.zone_checker:
            self.zone_checker.draw_zones(output)
        vis.compose_frame(output, tracks, fps, self._active_alert_msgs or None)
        return output, tracks, alerts

    # ─────────────────────────────────────────────────────────────────────────
    # Internal pipeline stages
    # ─────────────────────────────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray, frame_num: int) -> bool:
        """
        Full pipeline for one frame.
        Returns True if the pipeline should stop.
        """
        tracks, alerts = self._run_pipeline(frame, frame_num)
        fps = self._compute_fps()

        # ── Update alert banner state ─────────────────────────────────────────
        if alerts:
            self._active_alert_msgs = [str(a) for a in alerts[-3:]]  # Show latest 3
            self._alert_banner_ttl  = 60   # Show banner for 60 frames
        elif self._alert_banner_ttl > 0:
            self._alert_banner_ttl -= 1
            if self._alert_banner_ttl == 0:
                self._active_alert_msgs = []

        # ── Visualization ─────────────────────────────────────────────────────
        if self.display:
            output = frame  # Draw in-place for performance
            if config.SHOW_ZONE_OVERLAY and self.zone_checker:
                self.zone_checker.draw_zones(output)

            vis.compose_frame(
                output, tracks, fps,
                self._active_alert_msgs if self._alert_banner_ttl > 0 else None,
            )

            display_frame = vis.resize_for_display(output)
            cv2.imshow(config.WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or ESC
                return True
        return False

    def _run_pipeline(
        self,
        frame: np.ndarray,
        frame_num: int,
    ) -> tuple[List[TrackedPerson], List[AlertEvent]]:
        """
        Core pipeline stages: detect → track → zone → PPE → alert.
        Returns (tracks, new_alerts).
        """
        # ── Stage 1: Person Detection (with frame skip) ───────────────────────
        detections = self.person_detector.detect(frame)
        tracks = self._detections_to_tracks(detections, frame)

        # ── Stage 2: Tracking ─────────────────────────────────────────────────
        # Tracking removed: each frame is processed independently

        # ── Stage 3: Zone Check (cheap — runs before PPE to skip compliant) ───
        if self.zone_checker is not None:
            self.zone_checker.check_all(tracks)

        # ── Stage 4: PPE Detection on per-person crops ────────────────────────
        self.ppe_detector.process_tracks(tracks, frame)

        # ── Stage 5: Alert evaluation ─────────────────────────────────────────
        alerts = self._build_frame_alerts(tracks, frame_num)

        return tracks, alerts

    def _detections_to_tracks(self, detections, frame: np.ndarray) -> List[TrackedPerson]:
        frame_h, frame_w = frame.shape[:2]
        tracks: List[TrackedPerson] = []
        for idx, det in enumerate(detections):
            tracks.append(
                TrackedPerson(
                    track_id=idx + 1,
                    bbox_xyxy=np.array(det.bbox_xyxy, dtype=np.float32),
                    confidence=float(det.confidence),
                    frame_h=frame_h,
                    frame_w=frame_w,
                    detection_state=getattr(det, "detection_state", "SAFE"),
                )
            )
        return tracks

    @staticmethod
    def _build_frame_alerts(tracks: List[TrackedPerson], frame_num: int) -> List[AlertEvent]:
        alerts: List[AlertEvent] = []
        for track in tracks:
            if track.has_helmet is False:
                alerts.append(
                    AlertEvent(
                        track_id=track.track_id,
                        violation_type=ViolationType.NO_HELMET,
                        zone_ids=[],
                        frame_number=frame_num,
                    )
                )
            if getattr(config, "ENABLE_ZONE_DETECTION", False) and track.in_restricted_zone:
                alerts.append(
                    AlertEvent(
                        track_id=track.track_id,
                        violation_type=ViolationType.ZONE_BREACH,
                        zone_ids=list(track.zone_ids),
                        frame_number=frame_num,
                    )
                )
        return alerts

    def _compute_fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        return len(self._frame_times) / sum(self._frame_times)


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Industrial Safety Vision System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --source 0                        # Default webcam
  python main.py --source 1                        # Second webcam
  python main.py --source /path/to/video.mp4       # Video file
  python main.py --source /path/to/video.mp4 --no-display  # Headless
        """,
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video source: webcam index (0,1,...) or path to video file",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run without GUI (headless mode, e.g., on servers)",
    )
    parser.add_argument(
        "--list-cameras", action="store_true",
        help="List available webcam indices and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_cameras:
        cams = VideoSource.list_webcams()
        print(f"Available webcams: {cams}")
        sys.exit(0)

    # Coerce numeric strings to int for webcam indices
    source: str | int = args.source
    if source.isdigit():
        source = int(source)
    elif not Path(str(source)).exists() and source != "0":
        logger.error(f"File not found: {source}")
        sys.exit(1)

    pipeline = SafetyPipeline(
        source=source,
        headless=args.no_display,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
