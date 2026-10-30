"""
tests_isolated.py — Tests for pure-logic modules only.
Runs without torch, ultralytics, or deep_sort_realtime installed.
Uses locally defined stub dataclasses to mirror the real TrackedPerson API.
"""

from __future__ import annotations
import sys
import unittest
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Stub TrackedPerson (mirrors the real class without importing tracking.py)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrackedPerson:
    track_id:    int
    bbox_xyxy:   np.ndarray
    confidence:  float
    frame_h:     int
    frame_w:     int
    has_helmet:       Optional[bool]  = None
    ppe_confidence:   Optional[float] = None
    in_restricted_zone: bool = False
    zone_ids:           List[int] = field(default_factory=list)

    @property
    def foot_point(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox_xyxy.astype(int)
        return ((x1 + x2) // 2, y2)

    @property
    def has_violation(self) -> bool:
        return self.in_restricted_zone or (self.has_helmet is False)

    @property
    def violation_label(self) -> str:
        parts = []
        if self.has_helmet is False:
            parts.append("NO HELMET")
        if self.in_restricted_zone:
            zone_str = ",".join(str(z) for z in self.zone_ids)
            parts.append(f"ZONE VIOLATION[{zone_str}]")
        return " | ".join(parts) if parts else "OK"

    def crop_frame(self, frame: np.ndarray, padding: float = 0.1) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox_xyxy.astype(float)
        pw = (x2 - x1) * padding
        ph = (y2 - y1) * padding
        x1 = max(0, int(x1 - pw)); y1 = max(0, int(y1 - ph))
        x2 = min(self.frame_w - 1, int(x2 + pw))
        y2 = min(self.frame_h - 1, int(y2 + ph))
        crop = frame[y1:y2, x1:x2]
        return crop.copy() if crop.size > 0 else np.zeros((1,1,3), dtype=np.uint8)


def make_track(track_id, x1, y1, x2, y2, frame_h=720, frame_w=1280,
               has_helmet=None, in_zone=False, zone_ids=None):
    t = TrackedPerson(
        track_id=track_id,
        bbox_xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
        confidence=0.9, frame_h=frame_h, frame_w=frame_w,
    )
    t.has_helmet = has_helmet
    t.ppe_confidence = 0.85 if has_helmet is not None else None
    t.in_restricted_zone = in_zone
    t.zone_ids = zone_ids or []
    return t


def make_blank_frame(h=720, w=1280):
    return np.zeros((h, w, 3), dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Patch tracking module so zone_logic / alert_manager can import it
# ─────────────────────────────────────────────────────────────────────────────
import types, sys

_stub_tracking = types.ModuleType("tracking")
_stub_tracking.TrackedPerson = TrackedPerson
sys.modules["tracking"] = _stub_tracking

_stub_deepsort = types.ModuleType("deep_sort_realtime")
_stub_deepsort.deepsort_tracker = types.ModuleType("deep_sort_realtime.deepsort_tracker")
_stub_deepsort.deepsort_tracker.DeepSort = object
sys.modules["deep_sort_realtime"] = _stub_deepsort
sys.modules["deep_sort_realtime.deepsort_tracker"] = _stub_deepsort.deepsort_tracker

import config  # noqa: must come after stub injection


# ─────────────────────────────────────────────────────────────────────────────
# Zone Logic Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestZoneLogic(unittest.TestCase):

    def setUp(self):
        from zone_logic import ZoneDefinition, ZoneChecker
        self.ZoneDefinition = ZoneDefinition
        self.ZoneChecker    = ZoneChecker
        self.zone_square = ZoneDefinition(
            zone_id=1, name="Test Zone",
            points=np.array([[100,100],[400,100],[400,400],[100,400]], dtype=np.int32),
        )

    def test_point_inside_zone(self):
        self.assertTrue(self.zone_square.contains_point((250, 250)))

    def test_point_outside_zone(self):
        self.assertFalse(self.zone_square.contains_point((50, 50)))

    def test_point_on_boundary(self):
        self.assertTrue(self.zone_square.contains_point((100, 250)))

    def test_foot_point_inside_zone(self):
        track = make_track(1, 150, 50, 300, 380)  # foot at (225, 380) → inside
        self.assertTrue(self.zone_square.contains_person(track))

    def test_foot_point_outside_zone(self):
        track = make_track(1, 150, 50, 300, 95)   # foot at (225, 95) → above zone
        self.assertFalse(self.zone_square.contains_person(track))

    def test_person_partially_overlaps_zone_from_left(self):
        track = make_track(1, 20, 100, 120, 400)  # foot at (70, 400) → x=70 < 100
        self.assertFalse(self.zone_square.contains_person(track))

    def test_zone_checker_multiple_zones(self):
        checker = self.ZoneChecker(zone_definitions=[
            {"id":1,"name":"A","color":(0,0,220),
             "points":np.array([[0,0],[200,0],[200,200],[0,200]]),"require_full_inside":False},
            {"id":2,"name":"B","color":(0,165,255),
             "points":np.array([[300,300],[500,300],[500,500],[300,500]]),"require_full_inside":False},
        ])
        t_a   = make_track(1, 50,  50,  150, 180)  # foot (100,180) → in A
        t_b   = make_track(2, 350, 350, 450, 490)  # foot (400,490) → in B
        t_out = make_track(3, 600, 600, 700, 700)  # foot (650,700) → outside

        checker.check_all([t_a, t_b, t_out])

        self.assertTrue(t_a.in_restricted_zone);   self.assertIn(1, t_a.zone_ids)
        self.assertTrue(t_b.in_restricted_zone);   self.assertIn(2, t_b.zone_ids)
        self.assertFalse(t_out.in_restricted_zone); self.assertEqual(t_out.zone_ids, [])

    def test_require_full_inside_mode(self):
        zone = self.ZoneDefinition(
            zone_id=3, name="Strict",
            points=np.array([[200,200],[400,200],[400,400],[200,400]], dtype=np.int32),
            require_full_inside=True,
        )
        # foot (300,390) in, centroid (300,250) in → TRIGGER
        self.assertTrue(zone.contains_person(make_track(1, 200, 100, 400, 390)))
        # foot (300,390) in, centroid (300,50) OUT → NO TRIGGER
        self.assertFalse(zone.contains_person(make_track(2, 200, 0, 400, 390)))

    def test_zone_draw_does_not_crash(self):
        checker = self.ZoneChecker()
        frame = make_blank_frame()
        try:
            checker.draw_zones(frame)
        except Exception as e:
            self.fail(f"draw_zones raised: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Alert Manager Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAlertManager(unittest.TestCase):

    def setUp(self):
        from alert_manager import AlertManager, ViolationType
        self.AlertManager  = AlertManager
        self.ViolationType = ViolationType

    def test_no_alert_before_confirm_frames(self):
        mgr   = self.AlertManager(cooldown_frames=90, confirm_frames=3)
        track = make_track(1, 100, 100, 200, 400, has_helmet=False)
        alerts = []
        for i in range(2):
            alerts.extend(mgr.process([track], i))
        self.assertEqual(len(alerts), 0)

    def test_alert_fires_at_confirm_threshold(self):
        mgr   = self.AlertManager(cooldown_frames=90, confirm_frames=3)
        track = make_track(1, 100, 100, 200, 400, has_helmet=False)
        alerts = []
        for i in range(3):
            alerts.extend(mgr.process([track], i))
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].violation_type, self.ViolationType.NO_HELMET)
        self.assertEqual(alerts[0].track_id, 1)

    def test_cooldown_prevents_duplicates(self):
        mgr   = self.AlertManager(cooldown_frames=90, confirm_frames=1)
        track = make_track(1, 100, 100, 200, 400, has_helmet=False)
        alerts = []
        for i in range(50):
            alerts.extend(mgr.process([track], i))
        nh = [a for a in alerts if a.violation_type == self.ViolationType.NO_HELMET]
        self.assertEqual(len(nh), 1)

    def test_alert_fires_again_after_cooldown(self):
        mgr   = self.AlertManager(cooldown_frames=10, confirm_frames=1)
        track = make_track(1, 100, 100, 200, 400, has_helmet=False)
        alerts = []
        for i in range(25):
            alerts.extend(mgr.process([track], i))
        nh = [a for a in alerts if a.violation_type == self.ViolationType.NO_HELMET]
        self.assertGreaterEqual(len(nh), 2)

    def test_streak_resets_on_ok_frame(self):
        mgr      = self.AlertManager(cooldown_frames=90, confirm_frames=5)
        t_bad    = make_track(1, 100, 100, 200, 400, has_helmet=False)
        t_ok     = make_track(1, 100, 100, 200, 400, has_helmet=True)
        for i in range(3):
            mgr.process([t_bad], i)
        mgr.process([t_ok], 3)
        alerts = []
        for i in range(4, 8):
            alerts.extend(mgr.process([t_bad], i))
        self.assertEqual(len(alerts), 0)

    def test_zone_breach_alert(self):
        mgr   = self.AlertManager(cooldown_frames=90, confirm_frames=1)
        track = make_track(1, 100, 100, 200, 400, in_zone=True, zone_ids=[2])
        alerts = mgr.process([track], 0)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].violation_type, self.ViolationType.ZONE_BREACH)
        self.assertIn(2, alerts[0].zone_ids)

    def test_stale_track_pruned(self):
        mgr   = self.AlertManager(cooldown_frames=90, confirm_frames=1, prune_after_frames=10)
        track = make_track(99, 100, 100, 200, 400, has_helmet=False)
        mgr.process([track], 0)
        self.assertIn(99, mgr._states)
        other = make_track(2, 500, 100, 600, 400)
        for i in range(1, 16):
            mgr.process([other], i)
        self.assertNotIn(99, mgr._states)

    def test_both_violations_same_track(self):
        mgr   = self.AlertManager(cooldown_frames=90, confirm_frames=1)
        track = make_track(5, 100, 100, 200, 400, has_helmet=False, in_zone=True, zone_ids=[1])
        alerts = mgr.process([track], 0)
        types = {a.violation_type for a in alerts}
        self.assertIn(self.ViolationType.NO_HELMET, types)
        self.assertIn(self.ViolationType.ZONE_BREACH, types)


# ─────────────────────────────────────────────────────────────────────────────
# TrackedPerson State Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTrackedPerson(unittest.TestCase):

    def test_foot_point_bottom_center(self):
        t = make_track(1, 100, 50, 300, 450)
        self.assertEqual(t.foot_point, (200, 450))

    def test_violation_label_no_helmet(self):
        t = make_track(1, 0, 0, 100, 200, has_helmet=False)
        self.assertIn("NO HELMET", t.violation_label)

    def test_violation_label_zone(self):
        t = make_track(1, 0, 0, 100, 200, has_helmet=True, in_zone=True, zone_ids=[1])
        self.assertIn("ZONE VIOLATION", t.violation_label)

    def test_violation_label_both(self):
        t = make_track(1, 0, 0, 100, 200, has_helmet=False, in_zone=True, zone_ids=[2])
        self.assertIn("NO HELMET", t.violation_label)
        self.assertIn("ZONE VIOLATION", t.violation_label)

    def test_violation_label_ok(self):
        t = make_track(1, 0, 0, 100, 200, has_helmet=True)
        self.assertEqual(t.violation_label, "OK")

    def test_has_violation_flag(self):
        self.assertFalse(make_track(1, 0,0,100,200, has_helmet=True).has_violation)
        self.assertTrue(make_track(2, 0,0,100,200, has_helmet=False).has_violation)
        self.assertTrue(make_track(3, 0,0,100,200, has_helmet=True, in_zone=True, zone_ids=[1]).has_violation)

    def test_crop_clips_to_frame_boundary(self):
        t = make_track(1, 5, 5, 100, 200)
        frame = make_blank_frame()
        crop = t.crop_frame(frame, padding=0.3)
        self.assertGreater(crop.shape[0], 0)
        self.assertGreater(crop.shape[1], 0)


# ─────────────────────────────────────────────────────────────────────────────
# Input Handler Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestInputHandler(unittest.TestCase):

    def test_invalid_source_raises(self):
        from input_handler import VideoSource, VideoSourceError
        src = VideoSource(source="definitely_not_a_file.mp4")
        with self.assertRaises(VideoSourceError):
            src.open()

    def test_context_manager_cleans_up_on_error(self):
        from input_handler import VideoSource, VideoSourceError
        src = VideoSource(source="definitely_not_a_file.mp4")
        try:
            with src:
                pass
        except VideoSourceError:
            pass
        self.assertIsNone(src._cap)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Industrial Safety Vision — Isolated Logic Tests")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestZoneLogic, TestAlertManager, TestTrackedPerson, TestInputHandler]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(f"\n{'PASSED' if result.wasSuccessful() else 'FAILED'} — "
          f"{result.testsRun} tests, {len(result.failures)} failures, {len(result.errors)} errors")
    sys.exit(0 if result.wasSuccessful() else 1)
