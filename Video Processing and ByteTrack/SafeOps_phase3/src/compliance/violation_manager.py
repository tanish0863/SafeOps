"""Violation Event Lifecycle Manager for SafeOps.

Features:
- Event-based violation lifecycle: CANDIDATE -> CONFIRMED -> RESOLVED.
- Strict event deduplication: prevents generating duplicate events every frame.
- Evidence collection: records exact frame indices exhibiting non-compliance.
- Resolution tracking: marks violations resolved when worker returns to compliance.
"""
from __future__ import annotations

import uuid
from typing import Dict, List, Optional, Tuple

from src.contracts.schemas import ComplianceStatus, ViolationEvent, ViolationStatus, WorkerComplianceState


class ViolationManager:
    """Manages active and resolved safety violation events."""

    def __init__(
        self,
        recovery_confirmation_seconds: float = 1.0,
        max_occlusion_pause_seconds: float = 3.0,
    ):
        self.recovery_seconds = recovery_confirmation_seconds
        self.max_occlusion_pause_seconds = max_occlusion_pause_seconds
        # Active confirmed violations: (track_id, violation_type) -> ViolationEvent
        self._active_events: Dict[Tuple[int, str], ViolationEvent] = {}
        # Candidate violations pending confirmation: (track_id, violation_type) -> start_timestamp
        self._candidate_starts: Dict[Tuple[int, str], float] = {}
        # Consecutive compliant recovery timer: (track_id, violation_type) -> seconds
        self._recovery_timers: Dict[Tuple[int, str], float] = {}
        # Consecutive occlusion duration while in violation: (track_id, violation_type) -> seconds
        self._occlusion_durations: Dict[Tuple[int, str], float] = {}
        # Completed historical events
        self._all_events: List[ViolationEvent] = []
        self._prev_timestamps: Dict[int, float] = {}

    def reset(self) -> None:
        """Reset state between independent video streams."""
        self._active_events.clear()
        self._candidate_starts.clear()
        self._recovery_timers.clear()
        self._occlusion_durations.clear()
        self._all_events.clear()
        self._prev_timestamps.clear()

    def process_worker_compliance(
        self,
        compliance_state: WorkerComplianceState,
        frame_number: int,
        timestamp_seconds: float,
    ) -> List[ViolationEvent]:
        """Update violation event lifecycles based on current worker compliance."""
        tid = compliance_state.track_id
        prev_t = self._prev_timestamps.get(tid, timestamp_seconds)
        dt = max(0.0, timestamp_seconds - prev_t)
        self._prev_timestamps[tid] = timestamp_seconds

        events_changed: List[ViolationEvent] = []

        # Check each violation category
        item_status_map = {
            "missing_helmet": compliance_state.helmet_status,
            "missing_vest": compliance_state.vest_status,
        }

        # 1. Update active and new violations
        for vtype in ["missing_helmet", "missing_vest"]:
            key = (tid, vtype)
            item_status = item_status_map[vtype]

            if item_status == ComplianceStatus.NON_COMPLIANT:
                # Active non-compliance observed
                self._recovery_timers[key] = 0.0
                self._occlusion_durations[key] = 0.0

                if key in self._active_events:
                    # Ongoing confirmed violation: extend end timestamp and add evidence frame
                    event = self._active_events[key]
                    event.end_timestamp = round(timestamp_seconds, 3)
                    if frame_number not in event.evidence_frames:
                        event.evidence_frames.append(frame_number)
                elif compliance_state.active_violation:
                    # Transition to confirmed event
                    start_t = self._candidate_starts.get(key, timestamp_seconds)
                    event = ViolationEvent(
                        event_id=f"VIOL-{str(uuid.uuid4())[:8].upper()}",
                        track_id=tid,
                        violation_type=vtype,
                        start_timestamp=round(start_t, 3),
                        end_timestamp=round(timestamp_seconds, 3),
                        status=ViolationStatus.CONFIRMED,
                        evidence_frames=[frame_number],
                        confidence=1.0,
                        summary=f"Worker {tid} confirmed {vtype.replace('_', ' ')}",
                    )
                    self._active_events[key] = event
                    self._all_events.append(event)
                    events_changed.append(event)
                else:
                    # Still in candidate window
                    if key not in self._candidate_starts:
                        self._candidate_starts[key] = timestamp_seconds

            elif item_status == ComplianceStatus.COMPLIANT:
                # Worker is confirmed COMPLIANT: cancel candidate
                self._candidate_starts.pop(key, None)
                self._occlusion_durations[key] = 0.0

                if key in self._active_events:
                    # Accumulate recovery time only when actively compliant
                    rec_time = self._recovery_timers.get(key, 0.0) + dt
                    self._recovery_timers[key] = rec_time

                    if rec_time >= self.recovery_seconds:
                        # Mark violation resolved
                        event = self._active_events.pop(key)
                        event.status = ViolationStatus.RESOLVED
                        event.end_timestamp = round(timestamp_seconds, 3)
                        self._recovery_timers.pop(key, None)
                        events_changed.append(event)

            else:
                # Worker item status is UNKNOWN or INSUFFICIENT_EVIDENCE (e.g. occluded)
                # 1. Do NOT append frame_number to evidence_frames
                # 2. Do NOT advance recovery timer (occlusion is NOT recovery)
                # 3. Discard candidate start to prevent stale candidate confirmation
                self._candidate_starts.pop(key, None)

                if key in self._active_events:
                    # Suspend active violation.
                    # Track consecutive occlusion time
                    occ_time = self._occlusion_durations.get(key, 0.0) + dt
                    self._occlusion_durations[key] = occ_time

                    if occ_time >= self.max_occlusion_pause_seconds:
                        # Worker remained occluded beyond suspension timeout: close event
                        event = self._active_events.pop(key)
                        self._occlusion_durations.pop(key, None)
                        self._recovery_timers.pop(key, None)
                        events_changed.append(event)

        return events_changed

    def finalize_stream(self, final_timestamp: float) -> List[ViolationEvent]:
        """Close any open violation events at stream termination."""
        for key, event in list(self._active_events.items()):
            if event.status == ViolationStatus.CONFIRMED:
                event.end_timestamp = round(final_timestamp, 3)
        self._active_events.clear()
        return list(self._all_events)

    def get_all_events(self) -> List[ViolationEvent]:
        return list(self._all_events)

