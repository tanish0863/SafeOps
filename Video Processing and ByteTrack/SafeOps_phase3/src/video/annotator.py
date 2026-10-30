"""High-Resolution Video Annotation Subsystem for SafeOps.

Renders:
- Worker bounding boxes color-coded by compliance state:
  * Green: Fully Compliant
  * Red: Sustained Violation / Non-Compliant
  * Orange/Yellow: Insufficient Evidence / Uncertain Continuity
- 17 COCO body keypoints & skeleton lines
- Detected PPE bounding boxes (Cyan for helmets, Yellow for vests)
- Informative status badges showing Worker ID, Helmet state, Vest state, and overall Status
- Active violation alerts
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.contracts.schemas import ComplianceStatus, PPEDetectionItem, PPEItemType, WorkerComplianceState, WorkerPPEAssociation

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]

COLOR_COMPLIANT = (0, 200, 0)          # Green
COLOR_NON_COMPLIANT = (0, 0, 220)       # Red
COLOR_INSUFFICIENT = (0, 165, 255)      # Orange
COLOR_UNKNOWN = (128, 128, 128)         # Gray
COLOR_HELMET_BOX = (255, 255, 0)        # Cyan (BGR: Yellow-Cyan)
COLOR_VEST_BOX = (0, 255, 255)          # Yellow


class VideoAnnotator:
    """Renders professional CCTV safety overlays."""

    def __init__(
        self,
        draw_skeletons: bool = True,
        draw_ppe_boxes: bool = True,
        draw_hud: bool = True,
        kp_thresh: float = 0.4,
    ):
        self.draw_skeletons = draw_skeletons
        self.draw_ppe_boxes = draw_ppe_boxes
        self.draw_hud = draw_hud
        self.kp_thresh = kp_thresh

    def _get_status_color(self, status: ComplianceStatus) -> Tuple[int, int, int]:
        if status == ComplianceStatus.COMPLIANT:
            return COLOR_COMPLIANT
        elif status == ComplianceStatus.NON_COMPLIANT:
            return COLOR_NON_COMPLIANT
        elif status == ComplianceStatus.INSUFFICIENT_EVIDENCE:
            return COLOR_INSUFFICIENT
        return COLOR_UNKNOWN

    def annotate_frame(
        self,
        frame: np.ndarray,
        workers: List[Dict],
        compliance_states: Dict[int, WorkerComplianceState],
        associations: Dict[int, WorkerPPEAssociation],
        raw_ppe_detections: Optional[List[PPEDetectionItem]] = None,
        active_violations_count: int = 0,
    ) -> np.ndarray:
        """Annotate frame in-place and return annotated frame."""
        h, w, _ = frame.shape

        # 1. Draw raw PPE detections if enabled
        if self.draw_ppe_boxes and raw_ppe_detections:
            for ppe in raw_ppe_detections:
                px1, py1, px2, py2 = map(int, ppe.bbox_xyxy)
                color = COLOR_HELMET_BOX if ppe.item_type == PPEItemType.HELMET else COLOR_VEST_BOX
                cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
                cv2.putText(
                    frame,
                    f"{ppe.item_type.value.upper()} {ppe.confidence:.2f}",
                    (px1, max(15, py1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        # 2. Draw worker bounding boxes, skeletons, and badges
        for worker in workers:
            tid = worker["track_id"]
            c_state = compliance_states.get(tid)
            overall_status = c_state.overall_status if c_state else ComplianceStatus.UNKNOWN
            box_color = self._get_status_color(overall_status)

            x1, y1, x2, y2 = map(int, worker["bbox_xyxy"])
            line_thickness = 3 if (c_state and c_state.active_violation) else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, line_thickness)

            # Draw skeletons
            if self.draw_skeletons:
                kxy = worker.get("keypoints_xy")
                kc = worker.get("keypoints_conf")
                if kxy and kc:
                    for a, b in COCO_SKELETON:
                        if kc[a] >= self.kp_thresh and kc[b] >= self.kp_thresh:
                            pt1 = (int(kxy[a][0]), int(kxy[a][1]))
                            pt2 = (int(kxy[b][0]), int(kxy[b][1]))
                            cv2.line(frame, pt1, pt2, box_color, 2)
                    for pt_idx, (kx, ky) in enumerate(kxy):
                        if kc[pt_idx] >= self.kp_thresh:
                            cv2.circle(frame, (int(kx), int(ky)), 3, box_color, -1)

            # Build badge label
            assoc = associations.get(tid)
            h_str = assoc.helmet.status.value.upper() if assoc else "UNKNOWN"
            v_str = assoc.vest.status.value.upper() if assoc else "UNKNOWN"
            status_text = overall_status.value.replace("_", " ").upper()

            badge_text = f"ID {tid} | H:{h_str} | V:{v_str} | {status_text}"
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

            top = max(y1, th + 8)
            cv2.rectangle(frame, (x1, top - th - 6), (x1 + tw + 6, top), box_color, -1)
            cv2.putText(
                frame,
                badge_text,
                (x1 + 3, top - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 3. Draw Top Status HUD
        if self.draw_hud:
            hud_bg_color = (30, 30, 30)
            cv2.rectangle(frame, (0, 0), (w, 36), hud_bg_color, -1)

            hud_text = f"SafeOps Monitor | Workers: {len(workers)} | Active Violations: {active_violations_count}"
            hud_color = (0, 0, 255) if active_violations_count > 0 else (0, 255, 0)
            cv2.putText(frame, hud_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2, cv2.LINE_AA)

        return frame

