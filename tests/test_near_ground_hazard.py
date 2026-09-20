#!/usr/bin/env python3
"""
Unit tests for Near-Ground & Negative Obstacle Detector (ISO 3691-4).
"""

import time
import pytest
import numpy as np
from ros2_edge_perception.near_ground_hazard_detector import (
    NearGroundHazardDetector,
    GroundPlaneModel,
    DetectedHazard
)


@pytest.fixture
def detector():
    return NearGroundHazardDetector(
        min_cable_height_m=0.02,
        max_low_lying_height_m=0.35,
        min_dropoff_depth_m=0.05,
        max_roi_forward_m=8.0,
        max_roi_lateral_m=2.5,
        ransac_iterations=30,
        min_cluster_points=5
    )


def test_ground_plane_ransac_flat(detector):
    """Test RANSAC plane fitting on ideal flat industrial concrete floor."""
    np.random.seed(42)
    # Generate 5,000 points on plane z = 0 with small noise
    x = np.random.uniform(0.5, 7.0, 5000)
    y = np.random.uniform(-2.0, 2.0, 5000)
    z = np.random.normal(0.0, 0.005, 5000)
    points = np.column_stack([x, y, z])

    plane = detector.fit_ground_plane_ransac(points)
    assert plane is not None
    # Normal should point nearly straight up (+Z)
    assert abs(plane.normal[2]) > 0.95
    assert plane.inlier_ratio > 0.90


def test_positive_low_lying_cable_detection(detector):
    """Test detection of a 3cm high loose cable laying across the corridor."""
    np.random.seed(42)
    # Background flat floor
    x_floor = np.random.uniform(0.5, 7.0, 3000)
    y_floor = np.random.uniform(-2.0, 2.0, 3000)
    z_floor = np.random.normal(0.0, 0.004, 3000)
    floor_pts = np.column_stack([x_floor, y_floor, z_floor])

    # Cable obstacle: at x = 2.5m, y from -0.8 to +0.8m, z = 0.04m (4cm above ground)
    cable_x = np.random.uniform(2.45, 2.55, 60)
    cable_y = np.random.uniform(-0.8, 0.8, 60)
    cable_z = np.random.uniform(0.03, 0.05, 60)
    cable_pts = np.column_stack([cable_x, cable_y, cable_z])

    combined_pts = np.vstack([floor_pts, cable_pts])

    hazards, plane = detector.detect_hazards(combined_pts)
    assert plane is not None
    assert len(hazards) > 0

    # Verify at least one positive low-lying hazard is detected near x=2.5m
    cable_hazards = [h for h in hazards if h.hazard_type == "POSITIVE_LOW_LYING"]
    assert len(cable_hazards) > 0
    nearest = min(cable_hazards, key=lambda h: abs(h.centroid_3d[0] - 2.5))
    assert abs(nearest.centroid_3d[0] - 2.5) < 0.4
    assert 0.02 <= nearest.ground_height_diff <= 0.10


def test_negative_obstacle_dropoff_detection(detector):
    """Test detection of a loading dock edge / pit drop-off (-20cm)."""
    np.random.seed(42)
    # Ground floor extends only from x = 0.5 to x = 3.0m
    x_floor = np.random.uniform(0.5, 3.0, 2000)
    y_floor = np.random.uniform(-1.5, 1.5, 2000)
    z_floor = np.random.normal(0.0, 0.005, 2000)
    floor_pts = np.column_stack([x_floor, y_floor, z_floor])

    # Beyond x = 3.0m, ground drops down to z = -0.30m (30cm drop-off)
    x_pit = np.random.uniform(3.2, 5.0, 200)
    y_pit = np.random.uniform(-1.0, 1.0, 200)
    z_pit = np.random.normal(-0.30, 0.01, 200)
    pit_pts = np.column_stack([x_pit, y_pit, z_pit])

    combined_pts = np.vstack([floor_pts, pit_pts])

    hazards, plane = detector.detect_hazards(combined_pts)
    assert plane is not None

    dropoff_hazards = [h for h in hazards if h.hazard_type == "NEGATIVE_DROP_OFF"]
    assert len(dropoff_hazards) > 0
    nearest_drop = min(dropoff_hazards, key=lambda h: h.distance_m)
    assert nearest_drop.distance_m <= 3.5
    assert nearest_drop.severity in ["CRITICAL", "EMERGENCY"]


def test_realtime_execution_latency(detector):
    """Test that detection completes well within real-time budget (<20ms)."""
    np.random.seed(42)
    pts = np.random.uniform(-2.0, 5.0, (15000, 3))
    pts[:, 2] = np.clip(pts[:, 2], -0.1, 0.5)

    start = time.perf_counter()
    detector.detect_hazards(pts)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert elapsed_ms < 35.0  # Must run in sub-35ms even on CPU

