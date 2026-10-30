"""
zone_logic.py — Polygon-based restricted zone detection.

Responsibilities:
  - Define and manage zone polygons
  - Determine if a person is inside a restricted zone
  - Use foot-point (bottom-center of bbox) as the primary position probe
  - Optional: require centroid also inside zone to reduce false positives
    for persons who are merely leaning over a boundary
  - Support dynamic zone updates at runtime (e.g., via config reload)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config
from tracking import TrackedPerson

logger = logging.getLogger(__name__)


@dataclass
class ZoneDefinition:
    """Represents a single restricted polygon zone."""
    zone_id:              int
    name:                 str
    points:               np.ndarray   # Shape (N, 2), dtype int32
    color:                Tuple[int, int, int] = (0, 0, 220)
    require_full_inside:  bool = False  # If True, foot AND centroid must be inside

    def __post_init__(self):
        # Ensure correct dtype for cv2.pointPolygonTest
        self.points = np.array(self.points, dtype=np.int32)
        if self.points.ndim == 2 and self.points.shape[1] == 2:
            # cv2 expects shape (N, 1, 2)
            self._poly = self.points.reshape((-1, 1, 2))
        else:
            raise ValueError(f"Zone {self.zone_id}: points must be shape (N, 2)")

    def contains_point(self, point: Tuple[int, int]) -> bool:
        """
        cv2.pointPolygonTest returns:
          +dist  if inside
          0      if exactly on boundary
          -dist  if outside
        """
        result = cv2.pointPolygonTest(self._poly, (float(point[0]), float(point[1])), False)
        return result >= 0

    def contains_person(self, track: TrackedPerson) -> bool:
        """
        Determines if a tracked person is inside this zone.

        Logic:
          1. Primary check: foot point (bottom-center of bbox) inside polygon
          2. If require_full_inside: also check body centroid
        
        Using foot point prevents false positives where a person walks
        close to a zone boundary (their upper body overlaps but feet don't).
        """
        foot = track.foot_point
        if not self.contains_point(foot):
            return False

        if self.require_full_inside:
            x1, y1, x2, y2 = track.bbox_xyxy.astype(int)
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            if not self.contains_point(centroid):
                return False

        return True


class ZoneChecker:
    """
    Manages a collection of restricted zones and evaluates
    all tracked persons against them each frame.

    Optimizations:
      - Zones are pre-compiled once at init
      - Per-track check is O(Z) where Z = number of zones (typically small)
      - No frame-level rasterization; pure geometric checks
    """

    def __init__(self, zone_definitions: Optional[List[dict]] = None) -> None:
        if zone_definitions is None:
            zone_definitions = config.RESTRICTED_ZONES

        self.zones: List[ZoneDefinition] = []
        for zd in zone_definitions:
            try:
                zone = ZoneDefinition(
                    zone_id=zd["id"],
                    name=zd["name"],
                    points=zd["points"],
                    color=zd.get("color", (0, 0, 220)),
                    require_full_inside=zd.get("require_full_inside", False),
                )
                self.zones.append(zone)
                logger.info(f"[ZoneChecker] Loaded zone {zone.zone_id}: '{zone.name}'")
            except Exception as e:
                logger.error(f"[ZoneChecker] Failed to load zone {zd.get('id')}: {e}")

    def check_all(self, tracks: List[TrackedPerson]) -> None:
        """
        Check all tracks against all zones IN-PLACE.
        Populates `in_restricted_zone` and `zone_ids` on each TrackedPerson.
        """
        for track in tracks:
            track.in_restricted_zone = False
            track.zone_ids = []
            for zone in self.zones:
                if zone.contains_person(track):
                    track.in_restricted_zone = True
                    track.zone_ids.append(zone.zone_id)

    def update_zones(self, zone_definitions: List[dict]) -> None:
        """Hot-swap zone definitions at runtime without restarting the pipeline."""
        old_count = len(self.zones)
        self.zones = []
        for zd in zone_definitions:
            try:
                self.zones.append(ZoneDefinition(
                    zone_id=zd["id"],
                    name=zd["name"],
                    points=zd["points"],
                    color=zd.get("color", (0, 0, 220)),
                    require_full_inside=zd.get("require_full_inside", False),
                ))
            except Exception as e:
                logger.error(f"[ZoneChecker] Zone reload error: {e}")
        logger.info(f"[ZoneChecker] Zones updated: {old_count} → {len(self.zones)}")

    def draw_zones(self, frame: np.ndarray, alpha: float = config.VIS.zone_fill_alpha) -> np.ndarray:
        """
        Overlay all zone polygons on a frame with semi-transparent fill.
        Returns a modified copy of the frame (does not mutate input).
        """
        overlay = frame.copy()
        for zone in self.zones:
            poly = zone.points.reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [poly], zone.color)
            cv2.polylines(frame, [poly], isClosed=True, color=zone.color, thickness=2)
            # Zone label at centroid
            M = cv2.moments(poly)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(
                    frame, zone.name,
                    (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, zone.color, 2,
                )

        # Blend fill
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame


# ─────────────────────────────────────────────────────────────────────────────
# Zone Calibration Helper
# Run this standalone to define zones interactively on a sample frame.
# ─────────────────────────────────────────────────────────────────────────────
class ZoneCalibrator:
    """
    Interactive tool to define zone polygons by clicking on a frame.

    Usage:
        calibrator = ZoneCalibrator()
        points = calibrator.collect_points(sample_frame, zone_name="Zone A")
        print(points)  # Copy-paste into config.py

    Keys:
        Left-click → add point
        'u'        → undo last point
        'c'        → confirm and close polygon
        'q'        → quit without saving
    """

    def __init__(self):
        self._points: List[Tuple[int, int]] = []
        self._done = False

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._points.append((x, y))

    def collect_points(
        self,
        frame: np.ndarray,
        zone_name: str = "Zone",
    ) -> Optional[List[Tuple[int, int]]]:
        self._points = []
        self._done = False

        win = f"Calibrate: {zone_name} — Click points, [c]=confirm, [u]=undo, [q]=quit"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, self._mouse_callback)

        while True:
            display = frame.copy()
            for pt in self._points:
                cv2.circle(display, pt, 5, (0, 255, 0), -1)
            if len(self._points) > 1:
                pts = np.array(self._points, dtype=np.int32)
                cv2.polylines(display, [pts], False, (0, 255, 255), 2)

            cv2.putText(
                display,
                f"Points: {len(self._points)} | [c]=confirm [u]=undo [q]=quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )
            cv2.imshow(win, display)

            key = cv2.waitKey(10) & 0xFF
            if key == ord("c") and len(self._points) >= 3:
                cv2.destroyWindow(win)
                return self._points
            elif key == ord("u") and self._points:
                self._points.pop()
            elif key == ord("q"):
                cv2.destroyWindow(win)
                return None

        return None
