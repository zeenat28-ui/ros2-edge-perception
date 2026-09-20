"""
LiDAR-Camera Multi-Sensor Fusion Engine for ROS 2.

Features:
1. 3D Frustum Projection: Projects LiDAR PointCloud (X_L, Y_L, Z_L) to Camera Pixel Plane (u, v).
2. 2D Bounding Box Gating: Associates LiDAR points falling inside YOLO 2D detections.
3. Statistical Depth Clustering: Median filtering & outlier rejection eliminating infrared sunlight washout.
4. Heading (Yaw) & Quaternion Estimation: Computes 3D oriented bounding boxes.
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np


class LidarCameraFusionEngine:
    """Fuses 2D visual detections with 3D LiDAR point clouds."""

    def __init__(
        self,
        fx: float = 554.25,
        fy: float = 554.25,
        cx: float = 320.0,
        cy: float = 240.0,
        extrinsic_rot: Optional[np.ndarray] = None,
        extrinsic_trans: Optional[np.ndarray] = None,
    ):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        # Default Extrinsics: Identity rotation, zero translation
        self.R = extrinsic_rot if extrinsic_rot is not None else np.eye(3, dtype=np.float32)
        self.t = extrinsic_trans if extrinsic_trans is not None else np.zeros(3, dtype=np.float32)

    def set_intrinsics(self, fx: float, fy: float, cx: float, cy: float):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def set_extrinsics(self, R: np.ndarray, t: np.ndarray):
        self.R = R.astype(np.float32)
        self.t = t.astype(np.float32)

    def project_points(self, lidar_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project N x 3 LiDAR points into 2D camera pixels (u, v) and metric depth z_c.
        Returns:
            u: array of horizontal pixel coordinates
            v: array of vertical pixel coordinates
            z_c: array of metric depths in camera optical frame
        """
        if len(lidar_points) == 0:
            return np.array([]), np.array([]), np.array([])

        # 1. Transform points: P_c = R * P_l + t
        pts_c = (self.R @ lidar_points[:, :3].T).T + self.t

        # Filter points in front of camera plane (Z > 0.1m)
        valid_mask = pts_c[:, 2] > 0.1
        if not np.any(valid_mask):
            return np.array([]), np.array([]), np.array([])

        pts_c_valid = pts_c[valid_mask]
        z_c = pts_c_valid[:, 2]

        # 2. Pin-Hole Perspective Projection
        u = (self.fx * pts_c_valid[:, 0]) / z_c + self.cx
        v = (self.fy * pts_c_valid[:, 1]) / z_c + self.cy

        return u, v, pts_c_valid

    def fuse(
        self,
        detections_2d: List[Dict],
        lidar_points: np.ndarray,
        min_depth: float = 0.5,
        max_depth: float = 80.0,
    ) -> List[Dict]:
        """
        Fuses 2D bounding boxes with 3D LiDAR point clouds.
        Returns list of 3D fused obstacle dictionaries with metric bounds and orientation quaternions.
        """
        fused_results = []
        if not detections_2d:
            return fused_results

        if len(lidar_points) > 0:
            u, v, pts_c = self.project_points(lidar_points)
        else:
            u, v, pts_c = np.array([]), np.array([]), np.array([])

        for det in detections_2d:
            fused = {
                "class_name": det["class_name"],
                "class_id": det["class_id"],
                "score": det["score"],
                "x": 0.0,
                "y": 0.0,
                "z": 3.0,
                "size_x": 0.5,
                "size_y": 0.5,
                "size_z": 0.5,
                "yaw": 0.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
                "lidar_point_count": 0,
                "fused_with_lidar": False,
            }

            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]

            # Association with LiDAR points
            if len(u) > 0:
                in_box_mask = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
                in_box_pts = pts_c[in_box_mask]

                # Distance range gating
                valid_depth_mask = (in_box_pts[:, 2] >= min_depth) & (in_box_pts[:, 2] <= max_depth)
                gated_pts = in_box_pts[valid_depth_mask]

                if len(gated_pts) >= 3:
                    depths = gated_pts[:, 2]
                    median_depth = np.median(depths)
                    depth_window = max(0.4, median_depth * 0.15)

                    # Cluster filtering around median depth
                    cluster_mask = np.abs(depths - median_depth) <= depth_window
                    clustered = gated_pts[cluster_mask]

                    if len(clustered) >= 2:
                        min_bounds = np.min(clustered, axis=0)
                        max_bounds = np.max(clustered, axis=0)

                        fused["x"] = float((min_bounds[0] + max_bounds[0]) / 2.0)
                        fused["y"] = float((min_bounds[1] + max_bounds[1]) / 2.0)
                        fused["z"] = float((min_bounds[2] + max_bounds[2]) / 2.0)

                        fused["size_x"] = float(max(0.3, max_bounds[0] - min_bounds[0]))
                        fused["size_y"] = float(max(0.3, max_bounds[1] - min_bounds[1]))
                        fused["size_z"] = float(max(0.3, max_bounds[2] - min_bounds[2]))

                        # Heading (Yaw) and Quaternion calculation
                        if len(clustered) >= 5:
                            dx = clustered[:, 0] - fused["x"]
                            dz = clustered[:, 2] - fused["z"]
                            cov_xx = np.mean(dx * dx)
                            cov_xz = np.mean(dx * dz)
                            cov_zz = np.mean(dz * dz)
                            yaw = 0.5 * math.atan2(2.0 * cov_xz, cov_xx - cov_zz)
                            fused["yaw"] = float(yaw)
                            fused["qy"] = float(math.sin(yaw / 2.0))
                            fused["qw"] = float(math.cos(yaw / 2.0))

                        fused["lidar_point_count"] = len(clustered)
                        fused["fused_with_lidar"] = True
                        fused_results.append(fused)
                        continue

            # Fallback to pin-hole box center
            cx_box = (x1 + x2) / 2.0
            cy_box = (y1 + y2) / 2.0
            fallback_depth = det.get("depth", 3.0)
            fused["x"] = float((cx_box - self.cx) * fallback_depth / self.fx)
            fused["y"] = float((cy_box - self.cy) * fallback_depth / self.fy)
            fused["z"] = float(fallback_depth)
            fused["size_x"] = float(max(0.4, (x2 - x1) * fallback_depth / self.fx))
            fused["size_y"] = float(max(0.4, (y2 - y1) * fallback_depth / self.fy))
            fused["size_z"] = 0.5
            fused_results.append(fused)

        return fused_results

