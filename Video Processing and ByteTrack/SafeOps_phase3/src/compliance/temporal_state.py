"""Temporal Consistency Engine for PPE Observations.

Maintains sliding observation history per worker track ID across sequential video frames.
Prevents transient detection dropouts from generating false safety violations.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

from src.contracts.schemas import ComplianceStatus, PPEItemType, PPEStatus, WorkerPPEState


class TrackPPEHistory:
    """Sliding window of recent PPE observations for a single tracked worker."""

    def __init__(self, max_history_frames: int = 60, window_seconds: float = 2.0):
        self.max_history_frames = max_history_frames
        self.window_seconds = window_seconds
        self.helmet_observations: deque[Tuple[float, WorkerPPEState]] = deque(maxlen=max_history_frames)
        self.vest_observations: deque[Tuple[float, WorkerPPEState]] = deque(maxlen=max_history_frames)
        self.first_seen_timestamp: Optional[float] = None
        self.last_seen_timestamp: Optional[float] = None
        self.frames_present: int = 0

    def add_observation(
        self,
        helmet: WorkerPPEState,
        vest: WorkerPPEState,
        timestamp_seconds: float,
    ) -> None:
        if self.first_seen_timestamp is None:
            self.first_seen_timestamp = timestamp_seconds
        self.last_seen_timestamp = timestamp_seconds
        self.frames_present += 1

        self.helmet_observations.append((timestamp_seconds, helmet))
        self.vest_observations.append((timestamp_seconds, vest))

    def evaluate_item_status(
        self,
        item_type: PPEItemType,
        min_valid_observations: int = 10,
        positive_threshold_ratio: float = 0.50,
        current_timestamp: Optional[float] = None,
    ) -> ComplianceStatus:
        """Evaluate smoothed compliance status for a specific PPE item."""
        obs_queue = self.helmet_observations if item_type == PPEItemType.HELMET else self.vest_observations
        if not obs_queue:
            return ComplianceStatus.UNKNOWN

        if current_timestamp is None:
            current_timestamp = obs_queue[-1][0]

        # Time-based window pruning: keep observations within [current_timestamp - window_seconds, current_timestamp]
        window_obs = [obs for t, obs in obs_queue if (current_timestamp - t) <= self.window_seconds]
        if not window_obs:
            return ComplianceStatus.UNKNOWN

        # Check if all observations in window are UNKNOWN (e.g., missing model or full occlusion)
        unknown_count = sum(1 for o in window_obs if o.status == PPEStatus.UNKNOWN)
        if unknown_count == len(window_obs):
            return ComplianceStatus.UNKNOWN

        # If latest observation is UNKNOWN (occluded/uncertain in this frame), report UNKNOWN
        if window_obs[-1].status == PPEStatus.UNKNOWN:
            return ComplianceStatus.UNKNOWN

        valid_observations = [o for o in window_obs if o.status != PPEStatus.UNKNOWN]
        if len(valid_observations) < min_valid_observations:
            return ComplianceStatus.INSUFFICIENT_EVIDENCE

        detected_count = sum(1 for o in valid_observations if o.status == PPEStatus.DETECTED)
        ratio = detected_count / len(valid_observations)

        if ratio >= positive_threshold_ratio:
            return ComplianceStatus.COMPLIANT
        else:
            return ComplianceStatus.NON_COMPLIANT


class TemporalConsistencyTracker:
    """Manages observation histories across all active worker tracks."""

    def __init__(self, window_seconds: float = 2.0, default_fps: float = 30.0):
        self.window_seconds = window_seconds
        self.default_fps = default_fps
        self.max_frames = max(15, int(window_seconds * default_fps))
        self._tracks: Dict[int, TrackPPEHistory] = {}

    def update(
        self,
        track_id: int,
        helmet: WorkerPPEState,
        vest: WorkerPPEState,
        timestamp_seconds: float,
    ) -> TrackPPEHistory:
        if track_id not in self._tracks:
            self._tracks[track_id] = TrackPPEHistory(
                max_history_frames=self.max_frames,
                window_seconds=self.window_seconds,
            )

        history = self._tracks[track_id]
        history.add_observation(helmet, vest, timestamp_seconds)
        return history

    def get_history(self, track_id: int) -> Optional[TrackPPEHistory]:
        return self._tracks.get(track_id)

    def active_track_ids(self) -> List[int]:
        return list(self._tracks.keys())

    def reset(self) -> None:
        self._tracks.clear()

