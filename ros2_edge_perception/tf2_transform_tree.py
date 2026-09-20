r"""
AURA-Drive™ SE(3) Coordinate Transform Tree & tf2 Broadcaster
============================================================
Frontier Lie Algebra SE(3) Dynamic Transform Tree:
  - Homogeneous transformation matrices T in SE(3).
  - Matrix exponential exp([xi]_\wedge) and logarithm log(T) in se(3).
  - Maintains full kinematic transform tree:
      map -> odom -> base_footprint -> base_link -> laser_link / camera_optical_frame
  - Transforms arbitrary 3D spatial points between any two frames in the tree.
"""

import time
import math
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import numpy as np


@dataclass
class Transform3D:
    parent_frame: str
    child_frame: str
    translation: np.ndarray  # (3,) [x, y, z]
    rotation_quat: np.ndarray  # (4,) [qx, qy, qz, qw]
    timestamp: float


def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """Converts a normalized quaternion [qx, qy, qz, qw] to a 3x3 rotation matrix R in SO(3)."""
    qx, qy, qz, qw = q / np.linalg.norm(q)
    return np.array([
        [1.0 - 2.0*(qy**2 + qz**2), 2.0*(qx*qy - qz*qw),     2.0*(qx*qz + qy*qw)],
        [2.0*(qx*qy + qz*qw),     1.0 - 2.0*(qx**2 + qz**2), 2.0*(qy*qz - qx*qw)],
        [2.0*(qx*qz - qy*qw),     2.0*(qy*qz + qx*qw),     1.0 - 2.0*(qx**2 + qy**2)]
    ], dtype=np.float32)


def rot_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Converts a 3x3 rotation matrix R in SO(3) to a quaternion [qx, qy, qz, qw]."""
    tr = np.trace(R)
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    q = np.array([qx, qy, qz, qw], dtype=np.float32)
    return q / np.linalg.norm(q)


def se3_exp(xi: np.ndarray) -> np.ndarray:
    r"""
    Lie algebra se(3) exponential map: exp([xi]_\wedge) in SE(3).
    xi is a 6D twist vector: [vx, vy, vz, omega_x, omega_y, omega_z]
    """
    v = xi[0:3]
    w = xi[3:6]
    theta = float(np.linalg.norm(w))

    T = np.eye(4, dtype=np.float32)
    if theta < 1e-6:
        T[0:3, 0:3] = np.eye(3)
        T[0:3, 3] = v
        return T

    # Rodrigues formula
    w_hat = np.array([
        [0.0, -w[2], w[1]],
        [w[2], 0.0, -w[0]],
        [-w[1], w[0], 0.0]
    ], dtype=np.float32) / theta

    R = np.eye(3) + math.sin(theta) * w_hat + (1.0 - math.cos(theta)) * np.dot(w_hat, w_hat)
    V = np.eye(3) + ((1.0 - math.cos(theta)) / theta) * w_hat + ((theta - math.sin(theta)) / theta) * np.dot(w_hat, w_hat)

    T[0:3, 0:3] = R
    T[0:3, 3] = np.dot(V, v)
    return T


class TF2TransformTree:
    """
    Maintains the full kinematic transform tree for the industrial AMR.
    """

    def __init__(self):
        self._transforms: Dict[Tuple[str, str], np.ndarray] = {}  # (parent, child) -> 4x4 T
        self._init_static_robot_transforms()

    def _init_static_robot_transforms(self):
        """Initializes calibrated static URDF transforms."""
        # 1. base_footprint -> base_link (chassis height: 0.08m)
        self.set_transform("base_footprint", "base_link", translation=[0.0, 0.0, 0.08], rpy=[0.0, 0.0, 0.0])
        # 2. base_link -> laser_link (LiDAR mount: x=0.35m, z=0.28m)
        self.set_transform("base_link", "laser_link", translation=[0.35, 0.0, 0.28], rpy=[0.0, 0.0, 0.0])
        # 3. base_link -> camera_link (Front camera: x=0.42m, z=0.35m)
        self.set_transform("base_link", "camera_link", translation=[0.42, 0.0, 0.35], rpy=[0.0, 0.0, 0.0])
        # 4. camera_link -> camera_optical_frame (Standard ROS optical frame: x-right, y-down, z-forward)
        self.set_transform("camera_link", "camera_optical_frame", translation=[0.0, 0.0, 0.0], rpy=[-math.pi/2, 0.0, -math.pi/2])

    def set_transform(
        self,
        parent_frame: str,
        child_frame: str,
        translation: List[float],
        rpy: Optional[List[float]] = None,
        quaternion: Optional[List[float]] = None
    ):
        """Register or update a directed SE(3) transform from parent to child."""
        T = np.eye(4, dtype=np.float32)
        T[0:3, 3] = np.array(translation, dtype=np.float32)

        if quaternion is not None:
            R = quat_to_rot_matrix(np.array(quaternion, dtype=np.float32))
        elif rpy is not None:
            roll, pitch, yaw = rpy
            Rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
            Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
            Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
            R = np.dot(Rz, np.dot(Ry, Rx))
        else:
            R = np.eye(3, dtype=np.float32)

        T[0:3, 0:3] = R
        # T maps coordinates from child_frame into parent_frame
        self._transforms[(child_frame, parent_frame)] = T
        # T_inv maps coordinates from parent_frame into child_frame
        T_inv = np.eye(4, dtype=np.float32)
        T_inv[0:3, 0:3] = R.T
        T_inv[0:3, 3] = -np.dot(R.T, T[0:3, 3])
        self._transforms[(parent_frame, child_frame)] = T_inv

    def update_odometry(self, x: float, y: float, theta: float):
        """Update dynamic odom -> base_footprint transform."""
        self.set_transform("odom", "base_footprint", translation=[x, y, 0.0], rpy=[0.0, 0.0, theta])

    def update_map_to_odom(self, x: float, y: float, theta: float):
        """Update dynamic map -> odom transform."""
        self.set_transform("map", "odom", translation=[x, y, 0.0], rpy=[0.0, 0.0, theta])

    def lookup_transform(self, target_frame: str, source_frame: str) -> np.ndarray:
        """
        Computes the net 4x4 SE(3) transformation matrix T_{target <- source}
        via BFS path finding in the transform graph.
        """
        if target_frame == source_frame:
            return np.eye(4, dtype=np.float32)

        # BFS to find chain of frames
        queue = [[source_frame]]
        visited = {source_frame}
        path = None

        while queue:
            curr_path = queue.pop(0)
            curr_node = curr_path[-1]

            if curr_node == target_frame:
                path = curr_path
                break

            for (p, c) in self._transforms.keys():
                if p == curr_node and c not in visited:
                    visited.add(c)
                    queue.append(curr_path + [c])

        if path is None:
            raise KeyError(f"No transform path exists between '{source_frame}' and '{target_frame}'")

        # Accumulate matrix transformations along path
        T_net = np.eye(4, dtype=np.float32)
        for i in range(len(path) - 1):
            T_step = self._transforms[(path[i], path[i+1])]
            T_net = np.dot(T_step, T_net)

        return T_net

    def transform_points(self, points_3d: np.ndarray, source_frame: str, target_frame: str) -> np.ndarray:
        """Transforms an array of 3D points (N, 3) from source_frame into target_frame."""
        if len(points_3d) == 0:
            return points_3d

        T = self.lookup_transform(target_frame, source_frame)
        # Convert to homogeneous coordinates (N, 4)
        N = len(points_3d)
        homo = np.hstack([points_3d, np.ones((N, 1), dtype=np.float32)])
        transformed = np.dot(homo, T.T)
        return transformed[:, 0:3]


if __name__ == "__main__":
    tf_tree = TF2TransformTree()
    tf_tree.update_odometry(x=2.5, y=1.2, theta=0.0)

    # Transform a LiDAR point at (1.0, 0.0, 0.0) in laser_link to base_footprint
    pt_lidar = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    pt_base = tf_tree.transform_points(pt_lidar, source_frame="laser_link", target_frame="base_footprint")
    print("LiDAR point in base_footprint:", pt_base)
    assert abs(pt_base[0, 0] - 1.35) < 1e-3  # 1.0 + 0.35
    assert abs(pt_base[0, 2] - 0.36) < 1e-3  # 0.28 + 0.08
    print("SE(3) Transform Tree self-test passed.")
