#!/usr/bin/env python3
"""
Unit & Integration Tests for LiDAR-Camera Fusion & 3D Oriented Bounding Boxes.
"""

import math
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ros2_edge_perception.lidar_camera_fusion import LidarCameraFusionEngine
from ros2_edge_perception.tracker_3d import MultiObjectTracker3D, Track3D


class TestLidarCameraFusion:
    def setup_method(self):
        # 640x480 standard pin-hole camera
        self.fx = 500.0
        self.fy = 500.0
        self.cx = 320.0
        self.cy = 240.0
        self.fusion = LidarCameraFusionEngine(fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy)

    def test_point_projection_math(self):
        """Verify that 3D LiDAR point directly on optical axis projects to (cx, cy)."""
        pts = np.array([[0.0, 0.0, 5.0]])
        u, v, pts_c = self.fusion.project_points(pts)

        assert len(u) == 1
        assert len(v) == 1
        assert abs(u[0] - self.cx) < 1e-4
        assert abs(v[0] - self.cy) < 1e-4
        assert abs(pts_c[0, 2] - 5.0) < 1e-4

    def test_behind_camera_rejection(self):
        """Verify points behind camera (Z <= 0) are strictly rejected."""
        pts = np.array([
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 3.0],
        ])
        u, v, pts_c = self.fusion.project_points(pts)
        assert len(u) == 1
        assert abs(pts_c[0, 2] - 3.0) < 1e-4

    def test_bounding_box_association(self):
        """Verify LiDAR points falling inside 2D box are clustered and fused."""
        # 2D Bounding box around optical center: [270, 190, 370, 290] (100x100)
        detections_2d = [
            {
                "class_name": "car",
                "class_id": 2,
                "score": 0.92,
                "x1": 270.0,
                "y1": 190.0,
                "x2": 370.0,
                "y2": 290.0,
            }
        ]

        # Generate cluster of 10 LiDAR points at Z=4.0m near center
        lidar_points = []
        for dx in np.linspace(-0.2, 0.2, 5):
            for dy in np.linspace(-0.2, 0.2, 5):
                lidar_points.append([dx, dy, 4.0])
        # Add an outlier far outside the box
        lidar_points.append([10.0, 10.0, 4.0])
        lidar_pts = np.array(lidar_points, dtype=np.float32)

        fused = self.fusion.fuse(detections_2d, lidar_pts, min_depth=0.5, max_depth=50.0)

        assert len(fused) == 1
        obj = fused[0]
        assert obj["class_name"] == "car"
        assert obj["fused_with_lidar"] is True
        assert abs(obj["z"] - 4.0) < 0.15
        assert obj["lidar_point_count"] >= 10

    def test_heading_yaw_and_quaternion_estimation(self):
        """Verify 3D oriented bounding box yaw and quaternion calculation."""
        det = {
            "class_name": "car",
            "class_id": 2,
            "score": 0.9,
            "x": 0.0,
            "y": 0.0,
            "z": 5.0,
            "size_x": 1.8,
            "size_y": 1.5,
            "size_z": 4.0,
        }
        track = Track3D(det, timestamp=0.0)

        # Static object: speed is zero -> orientation is identity quaternion
        yaw, qx, qy, qz, qw = track.compute_orientation()
        assert abs(yaw) < 1e-4
        assert abs(qw - 1.0) < 1e-4

        # Moving object: simulate forward velocity along Z (vx=0, vz=1.0)
        track.x[3] = 0.0  # vx
        track.x[5] = 1.0  # vz
        yaw, qx, qy, qz, qw = track.compute_orientation()
        assert abs(yaw) < 1e-4  # Forward heading

        # Moving rightwards along X: (vx=1.0, vz=0.0) -> yaw = pi/2
        track.x[3] = 1.0
        track.x[5] = 0.0
        yaw, qx, qy, qz, qw = track.compute_orientation()
        assert abs(yaw - math.pi / 2.0) < 1e-3
        expected_qy = math.sin(math.pi / 4.0)
        expected_qw = math.cos(math.pi / 4.0)
        assert abs(qy - expected_qy) < 1e-3
        assert abs(qw - expected_qw) < 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

