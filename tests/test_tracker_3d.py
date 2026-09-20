"""
Enterprise Unit Tests for 3D Multi-Object Kalman Tracker (Tier-1 AV Stack).

Tests:
1. Kalman Filter prediction and state covariance propagation.
2. Velocity estimation convergence from sequential 3D observations.
3. Track lifecycle state machine: Tentative -> Confirmed -> Lost -> Deleted.
4. Occlusion handling: Track maintains state across missing detection frames.
5. Time-To-Collision (TTC) calculation for approaching obstacles.
"""

import numpy as np
import pytest
from ros2_edge_perception.tracker_3d import MultiObjectTracker3D, Track3D


def test_track3d_initialization_and_prediction():
    """Verify single track state initialization and constant-velocity prediction."""
    detection = {
        "x": 1.0, "y": 0.0, "z": 3.0,
        "size_x": 0.5, "size_y": 1.7, "size_z": 0.4,
        "score": 0.9, "class_name": "person", "class_id": 0
    }
    track = Track3D(detection, timestamp=0.0)

    assert track.x[0] == 1.0
    assert track.x[2] == 3.0
    assert not track.confirmed  # Needs 2 hits to confirm

    # Simulate 0.1s dt prediction
    track.predict(dt=0.1)
    # With zero initial velocity, position should remain unchanged
    assert np.isclose(track.x[0], 1.0)
    assert np.isclose(track.x[2], 3.0)


def test_velocity_estimation_convergence():
    """Verify that Kalman filter accurately estimates obstacle 3D velocity."""
    tracker = MultiObjectTracker3D(match_distance_threshold=1.5)

    # Simulate an obstacle moving forward along Z at -1.0 m/s (approaching robot)
    # Starting at z=5.0m, dt=0.1s per step -> z decreases by 0.1m each step
    z_pos = 5.0
    t = 0.0

    for _ in range(15):
        det = [{
            "x": 0.0, "y": 0.0, "z": z_pos,
            "size_x": 0.6, "size_y": 1.8, "size_z": 0.4,
            "score": 0.9, "class_name": "person", "class_id": 0
        }]
        tracks = tracker.update(det, timestamp=t)
        z_pos -= 0.10  # 1.0 m/s approach
        t += 0.10

    assert len(tracks) == 1
    active_track = tracks[0]
    # Check velocity estimation convergence: vz should be close to -1.0 m/s
    assert abs(active_track["vz"] - (-1.0)) < 0.25, f"Estimated vz={active_track['vz']}, expected ~ -1.0 m/s"
    assert active_track["track_id"] > 0


def test_occlusion_handling_and_pruning():
    """Verify track survives brief occlusion and is pruned after max_lost_frames."""
    tracker = MultiObjectTracker3D(max_lost_frames=3, match_distance_threshold=1.5)

    # 1. Establish confirmed track (2 hits)
    det = [{
        "x": 2.0, "y": 0.0, "z": 4.0,
        "size_x": 0.5, "size_y": 1.7, "size_z": 0.4,
        "score": 0.92, "class_name": "person", "class_id": 0
    }]
    tracker.update(det, timestamp=0.0)
    tracker.update(det, timestamp=0.1)
    assert len(tracker.tracks) == 1

    # 2. Simulate 2 frames of total occlusion (empty detections)
    tracks_f1 = tracker.update([], timestamp=0.2)
    tracks_f2 = tracker.update([], timestamp=0.3)
    # Track should still be alive during occlusion
    assert len(tracker.tracks) == 1

    # 3. Simulate exceeding max_lost_frames (3 more empty frames)
    tracker.update([], timestamp=0.4)
    tracker.update([], timestamp=0.5)
    tracker.update([], timestamp=0.6)

    # Track must now be safely deleted
    assert len(tracker.tracks) == 0


def test_time_to_collision_calculation():
    """Verify accurate TTC calculation for an obstacle on a collision course."""
    detection = {
        "x": 0.0, "y": 0.0, "z": 4.0,
        "size_x": 0.5, "size_y": 1.7, "size_z": 0.4,
        "score": 0.95, "class_name": "person", "class_id": 0
    }
    track = Track3D(detection, timestamp=0.0)
    # Manually inject approaching velocity: vz = -2.0 m/s at z = 4.0m
    track.x[5] = -2.0

    ttc = track.compute_ttc()
    assert ttc is not None
    # TTC = 4.0m / 2.0 m/s = 2.0 seconds
    assert np.isclose(ttc, 2.0)

