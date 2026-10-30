"""
alert_manager.py — Per-track alert deduplication and violation state machine.

Responsibilities:
  - Maintain per-track violation history
  - Enforce alert cooldowns so the same violation isn't printed 30× per second
  - Require N consecutive violation frames before emitting an alert
    (prevents transient false positives from flickering detections)
  - Expose structured AlertEvent objects for downstream use (logging, ESP32, etc.)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

import config
from tracking import TrackedPerson

logger = logging.getLogger(__name__)


class ViolationType(Enum):
    NO_HELMET    = auto()
    ZONE_BREACH  = auto()


@dataclass
class AlertEvent:
    track_id:       int
    violation_type: ViolationType
    zone_ids:       List[int]
    timestamp:      float = field(default_factory=time.time)
    frame_number:   int   = 0

    def __str__(self) -> str:
        if self.violation_type == ViolationType.NO_HELMET:
            return f"ID {self.track_id}: NO HELMET"
        else:
            zone_str = ",".join(str(z) for z in self.zone_ids)
            return f"ID {self.track_id}: ZONE VIOLATION [Zone {zone_str}]"


@dataclass
class _TrackState:
    """Internal mutable state for one track ID."""
    # Consecutive frame counters for violation confirmation
    no_helmet_streak:   int = 0
    zone_breach_streak: int = 0

    # Frame numbers of last emitted alert per type
    last_no_helmet_alert_frame:   int = -9999
    last_zone_breach_alert_frame: int = -9999


class AlertManager:
    """
    Stateful alert controller.

    Design:
      - State machine per track: UNKNOWN → STREAK_BUILDING → ALERT_EMITTED → COOLDOWN
      - Violation must persist for VIOLATION_CONFIRM_FRAMES consecutive frames
        before an alert fires (avoids one-frame flickers)
      - After an alert fires, ALERT_COOLDOWN_FRAMES must pass before the same
        violation fires again for the same track ID
      - Stale tracks (not seen recently) are pruned from the registry
    """

    def __init__(
        self,
        cooldown_frames:   int = config.ALERT_COOLDOWN_FRAMES,
        confirm_frames:    int = config.VIOLATION_CONFIRM_FRAMES,
        prune_after_frames: int = 300,   # Remove track state after N frames unseen
    ) -> None:
        self.cooldown_frames    = cooldown_frames
        self.confirm_frames     = confirm_frames
        self.prune_after_frames = prune_after_frames

        self._states:       Dict[int, _TrackState] = {}
        self._last_seen:    Dict[int, int]          = {}  # track_id → last frame
        self._current_frame = 0

    def process(
        self,
        tracks: List[TrackedPerson],
        frame_number: int,
    ) -> List[AlertEvent]:
        """
        Evaluate all tracks and return a list of new AlertEvents to emit.
        Call once per frame after PPE and zone checks are complete.
        """
        self._current_frame = frame_number
        events: List[AlertEvent] = []

        active_ids = set()
        for track in tracks:
            tid = track.track_id
            active_ids.add(tid)
            self._last_seen[tid] = frame_number

            if tid not in self._states:
                self._states[tid] = _TrackState()

            state = self._states[tid]
            events.extend(self._evaluate_track(track, state, frame_number))

        # Prune stale track states
        stale = [
            tid for tid, last in self._last_seen.items()
            if frame_number - last > self.prune_after_frames
        ]
        for tid in stale:
            self._states.pop(tid, None)
            self._last_seen.pop(tid, None)

        return events

    def _evaluate_track(
        self,
        track: TrackedPerson,
        state: _TrackState,
        frame_number: int,
    ) -> List[AlertEvent]:
        events: List[AlertEvent] = []

        # ── NO HELMET ─────────────────────────────────────────────────────────
        if track.has_helmet is False:
            state.no_helmet_streak += 1
        else:
            state.no_helmet_streak = 0

        if state.no_helmet_streak >= self.confirm_frames:
            frames_since = frame_number - state.last_no_helmet_alert_frame
            if frames_since >= self.cooldown_frames:
                event = AlertEvent(
                    track_id=track.track_id,
                    violation_type=ViolationType.NO_HELMET,
                    zone_ids=[],
                    frame_number=frame_number,
                )
                events.append(event)
                state.last_no_helmet_alert_frame = frame_number
                logger.warning(str(event))
                print(str(event))   # ← Required console output

        # ── ZONE BREACH ───────────────────────────────────────────────────────
        if track.in_restricted_zone:
            state.zone_breach_streak += 1
        else:
            state.zone_breach_streak = 0

        if state.zone_breach_streak >= self.confirm_frames:
            frames_since = frame_number - state.last_zone_breach_alert_frame
            if frames_since >= self.cooldown_frames:
                event = AlertEvent(
                    track_id=track.track_id,
                    violation_type=ViolationType.ZONE_BREACH,
                    zone_ids=track.zone_ids,
                    frame_number=frame_number,
                )
                events.append(event)
                state.last_zone_breach_alert_frame = frame_number
                logger.warning(str(event))
                print(str(event))   # ← Required console output

        return events

    def get_streak(self, track_id: int, vtype: ViolationType) -> int:
        """Return current violation streak for a given track and type."""
        state = self._states.get(track_id)
        if state is None:
            return 0
        if vtype == ViolationType.NO_HELMET:
            return state.no_helmet_streak
        return state.zone_breach_streak
