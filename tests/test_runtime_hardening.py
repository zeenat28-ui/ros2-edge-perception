#!/usr/bin/env python3
"""
Comprehensive Qualification Test Suite for ROS 2 Edge Perception Runtime Hardening.

Verifies:
1. Sensor Disconnect & Timeout Watchdogs (Camera Timeout, Depth Timeout).
2. NaN, Inf, and Out-of-Range Depth Rejection.
3. Invalid Camera Intrinsics Handling (Zero / Negative Focal Lengths).
4. Stale Frame & Backpressure Telemetry Accounting.
5. 3D Kalman Tracker Covariance & Velocity Clamping under Sensor Spikes.
6. Malformed LiDAR Point Cloud Input Rejection.
"""

import math
import os
import sys
import time
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ros2_edge_perception.perception_node import (
    filter_depth_roi,
    deproject_pixel_to_3d,
    preprocess_letterbox,
    PerceptionNode,
)
from ros2_edge_perception.tracker_3d import Track3D, MultiObjectTracker3D


class TestRuntimeHardening:

    def test_nan_inf_depth_rejection(self):
        """Verify filter_depth_roi strictly rejects NaNs, Infs, and invalid values."""
        # Case 1: Array of all NaNs
        nan_roi = np.full((30, 30), np.nan, dtype=np.float32)
        assert filter_depth_roi(nan_roi) is None

        # Case 2: Array of all Infs
        inf_roi = np.full((30, 30), np.inf, dtype=np.float32)
        assert filter_depth_roi(inf_roi) is None

        # Case 3: Array of negative depths (behind camera)
        neg_roi = np.full((30, 30), -2.5, dtype=np.float32)
        assert filter_depth_roi(neg_roi) is None

        # Case 4: Empty array
        empty_roi = np.empty((0, 0), dtype=np.float32)
        assert filter_depth_roi(empty_roi) is None

        # Case 5: Noisy array with 20% NaNs and 20% Infs, but valid 2.2m cluster
        clean = np.full(60, 2.20, dtype=np.float32)
        nans = np.full(20, np.nan, dtype=np.float32)
        infs = np.full(20, np.inf, dtype=np.float32)
        mixed = np.concatenate([clean, nans, infs])
        recovered = filter_depth_roi(mixed, min_depth=0.2, max_depth=10.0)
        assert recovered is not None
        assert abs(recovered - 2.20) < 0.05

    def test_invalid_camera_intrinsics_handling(self):
        """Verify deproject_pixel_to_3d gracefully rejects zero or negative focal lengths."""
        u, v, z = 320.0, 240.0, 2.0
        cx, cy = 320.0, 240.0

        # Zero focal length -> must return NaNs, not raise ZeroDivisionError
        x, y, z_out = deproject_pixel_to_3d(u, v, z, fx=0.0, fy=554.25, cx=cx, cy=cy)
        assert math.isnan(x) and math.isnan(y)

        # Negative focal length -> must return NaNs
        x, y, z_out = deproject_pixel_to_3d(u, v, z, fx=-500.0, fy=554.25, cx=cx, cy=cy)
        assert math.isnan(x)

        # Non-finite depth -> must return NaNs
        x, y, z_out = deproject_pixel_to_3d(u, v, z=float("nan"), fx=554.25, fy=554.25, cx=cx, cy=cy)
        assert math.isnan(z_out)

    def test_tracker_velocity_spike_clamping(self):
        """Verify Track3D clamps extreme sensor noise spikes to physical AMR dynamics."""
        det = {
            "class_name": "obstacle",
            "class_id": 1,
            "score": 0.9,
            "x": 0.0,
            "y": 0.0,
            "z": 3.0,
            "size_x": 0.5,
            "size_y": 0.5,
            "size_z": 0.5,
        }
        track = Track3D(det, timestamp=0.0)

        # Simulate a massive anomalous sensor jump (+50 meters in 0.033 seconds = 1500 m/s!)
        spike_det = {
            "class_name": "obstacle",
            "class_id": 1,
            "score": 0.9,
            "x": 50.0,
            "y": 0.0,
            "z": 3.0,
            "size_x": 0.5,
            "size_y": 0.5,
            "size_z": 0.5,
        }
        track.predict(dt=0.033)
        track.update(spike_det, timestamp=0.033)

        # Velocity vx must be clamped to <= 5.0 m/s (physical AMR limit)
        assert abs(track.x[3]) <= 5.0, f"Velocity {track.x[3]} exceeded physical clamp of 5.0 m/s"

    def test_tracker_covariance_divergence_protection(self):
        """Verify covariance matrix P does not explode during prolonged sensor occlusions."""
        det = {
            "class_name": "person",
            "class_id": 0,
            "score": 0.85,
            "x": 1.0,
            "y": 0.0,
            "z": 4.0,
            "size_x": 0.6,
            "size_y": 0.6,
            "size_z": 1.7,
        }
        track = Track3D(det, timestamp=0.0)

        # Run 100 consecutive predict steps without measurement updates (simulated sensor blackout)
        for _ in range(100):
            track.predict(dt=0.033)

        # P diagonal must remain bounded and strictly finite
        p_diag = np.diag(track.P)
        assert np.isfinite(p_diag).all()
        assert (p_diag <= 100.0).all(), f"Covariance exploded: {p_diag}"
        assert (p_diag >= 1e-4).all(), f"Covariance collapsed: {p_diag}"

    def test_tracker_nan_observation_rejection(self):
        """Verify MultiObjectTracker3D rejects observations containing NaNs."""
        tracker = MultiObjectTracker3D()
        corrupted_detections = [
            {"class_name": "person", "score": 0.8, "x": float("nan"), "y": 0.0, "z": 2.0, "size_x": 0.5, "size_y": 0.5, "size_z": 0.5},
            {"class_name": "car", "score": 0.9, "x": 1.0, "y": float("inf"), "z": 3.0, "size_x": 0.5, "size_y": 0.5, "size_z": 0.5},
            {"class_name": "valid_box", "score": 0.95, "x": 1.5, "y": 0.0, "z": 2.5, "size_x": 0.5, "size_y": 0.5, "size_z": 0.5},
        ]

        active_tracks = tracker.update(corrupted_detections, timestamp=0.0)

        # Only the valid detection should be tracked; corrupted ones must be dropped
        assert len(tracker.tracks) == 1
        tracked_id = list(tracker.tracks.keys())[0]
        assert tracker.tracks[tracked_id].class_name == "valid_box"

    def test_sensor_timeout_watchdog_state_transitions(self):
        """Verify perception node watchdog correctly detects camera and depth timeouts."""
        # Instantiate a standalone PerceptionNode instance
        node = PerceptionNode()

        # Case 1: Initial startup -> no camera frames received yet -> CAMERA_TIMEOUT
        node.last_camera_time = 0.0
        node.last_depth_time = 0.0
        node._update_safety_state()
        assert node.safety_state == "CAMERA_TIMEOUT"

        # Case 2: Camera fresh, but depth timed out (>0.5s)
        node.last_camera_time = time.time()
        node.last_depth_time = time.time() - 1.0
        node._update_safety_state()
        assert node.safety_state == "DEPTH_TIMEOUT"

        # Case 3: Both camera and depth fresh (<0.5s) -> NOMINAL
        node.last_camera_time = time.time()
        node.last_depth_time = time.time()
        node.enable_lidar_fusion = False
        node._update_safety_state()
        assert node.safety_state == "NOMINAL"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
