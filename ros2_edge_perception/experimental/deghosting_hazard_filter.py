#!/usr/bin/env python3
"""
Temporal Persistence and De-Ghosting Filter for False-Positive Obstacle Rejection.

Filters transient sensor artifacts (specular floor glare, dust plumes, single-frame noise):
- Tracks candidate detections over a sliding temporal window of N frames
- Computes spatial-temporal persistence score: P = (frames_observed / window_size) * stability
- Suppresses transient false positives while maintaining fast reaction to genuine hazards
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class TrackedHazardCandidate:
    """Tracks a spatial hazard over successive observation cycles."""
    track_id: int
    hazard_type: str
    centroid_history: List[Tuple[float, float, float]] = field(default_factory=list)
    first_seen_timestamp: float = field(default_factory=time.time)
    last_seen_timestamp: float = field(default_factory=time.time)
    consecutive_hits: int = 1
    total_hits: int = 1
    total_misses: int = 0
    confidence_history: List[float] = field(default_factory=list)

    @property
    def persistence_score(self) -> float:
        """Computes spatial-temporal persistence score in [0.0, 1.0]."""
        window_size = 5
        hit_ratio = min(1.0, self.total_hits / window_size)

        # Spatial stability: variance of centroid positions
        if len(self.centroid_history) >= 2:
            positions = np.array(self.centroid_history[-window_size:])
            variance = np.mean(np.var(positions, axis=0))
            stability = float(np.exp(-2.5 * variance))  # Higher stability when stationary
        else:
            stability = 0.5

        mean_conf = float(np.mean(self.confidence_history[-window_size:])) if self.confidence_history else 0.5
        return float(np.clip(0.5 * hit_ratio + 0.3 * stability + 0.2 * mean_conf, 0.0, 1.0))


@dataclass
class DeghostingFilterResult:
    """Output of the de-ghosting temporal filter."""
    verified_hazards: List[Dict]      # Confirmed physical obstacles (P >= 0.75)
    rejected_ghosts: List[Dict]       # Filtered transient reflections / dust (P < 0.75)
    phantom_stops_prevented: int
    processing_time_ms: float
    timestamp: float = field(default_factory=time.time)


class TemporalDeghostingFilter:
    """
    Real-time spatial-temporal filter eliminating false-positive AMR stops.
    Integrates directly between hazard detectors and the ISO 3691-4 safety supervisor.
    """

    def __init__(
        self,
        window_size: int = 5,
        min_persistence_threshold: float = 0.70,
        association_radius_m: float = 0.35,
        max_track_staleness_sec: float = 0.50
    ):
        self.window_size = window_size
        self.min_persistence_threshold = min_persistence_threshold
        self.association_radius_m = association_radius_m
        self.max_track_staleness_sec = max_track_staleness_sec

        self._active_tracks: Dict[int, TrackedHazardCandidate] = {}
        self._next_track_id = 1
        self.total_phantom_stops_prevented = 0

    def filter_hazards(self, candidate_hazards: List[Dict]) -> DeghostingFilterResult:
        """
        Filters input candidate hazards against historical temporal tracks.
        Returns verified physical hazards and rejected transient ghosts.
        """
        t0 = time.perf_counter()
        now = time.time()

        # 1. Age out stale tracks
        stale_ids = [
            tid for tid, track in self._active_tracks.items()
            if (now - track.last_seen_timestamp) > self.max_track_staleness_sec
        ]
        for tid in stale_ids:
            del self._active_tracks[tid]

        # 2. Match incoming candidates to existing tracks via nearest neighbor
        unmatched_candidates = []
        matched_track_ids = set()

        for cand in candidate_hazards:
            pos = cand.get("centroid_3d", [0.0, 0.0, 0.0])
            c_pos = np.array(pos[:3])

            best_match_id = None
            best_dist = self.association_radius_m

            for tid, track in self._active_tracks.items():
                if tid in matched_track_ids:
                    continue
                last_pos = np.array(track.centroid_history[-1])
                dist = float(np.linalg.norm(c_pos - last_pos))
                if dist < best_dist:
                    best_dist = dist
                    best_match_id = tid

            if best_match_id is not None:
                # Update existing track
                matched_track_ids.add(best_match_id)
                track = self._active_tracks[best_match_id]
                track.centroid_history.append((float(c_pos[0]), float(c_pos[1]), float(c_pos[2])))
                track.confidence_history.append(cand.get("confidence", 0.8))
                track.last_seen_timestamp = now
                track.consecutive_hits += 1
                track.total_hits += 1
                cand["persistence_score"] = round(track.persistence_score, 3)
                cand["track_id"] = best_match_id
            else:
                unmatched_candidates.append(cand)

        # 3. Create new tracks for unmatched candidates
        for cand in unmatched_candidates:
            pos = cand.get("centroid_3d", [0.0, 0.0, 0.0])
            tid = self._next_track_id
            self._next_track_id += 1

            new_track = TrackedHazardCandidate(
                track_id=tid,
                hazard_type=cand.get("hazard_type", "UNKNOWN"),
                centroid_history=[(float(pos[0]), float(pos[1]), float(pos[2]))],
                first_seen_timestamp=now,
                last_seen_timestamp=now,
                consecutive_hits=1,
                total_hits=1,
                confidence_history=[cand.get("confidence", 0.7)]
            )
            self._active_tracks[tid] = new_track
            cand["persistence_score"] = round(new_track.persistence_score, 3)
            cand["track_id"] = tid

        # 4. Classify candidates into Verified Physical Hazards vs Rejected Ghosts
        verified = []
        rejected = []

        for cand in candidate_hazards:
            p_score = cand.get("persistence_score", 0.0)
            # High-confidence negative drop-offs (loading docks) trigger earlier for safety
            threshold = 0.55 if cand.get("hazard_type") == "NEGATIVE_DROP_OFF" else self.min_persistence_threshold

            if p_score >= threshold:
                cand["verified"] = True
                verified.append(cand)
            else:
                cand["verified"] = False
                cand["rejection_reason"] = "TRANSIENT_NOISE_OR_GLARE"
                rejected.append(cand)
                self.total_phantom_stops_prevented += 1

        dt_ms = (time.perf_counter() - t0) * 1000.0

        return DeghostingFilterResult(
            verified_hazards=verified,
            rejected_ghosts=rejected,
            phantom_stops_prevented=self.total_phantom_stops_prevented,
            processing_time_ms=round(dt_ms, 3),
            timestamp=now
        )

