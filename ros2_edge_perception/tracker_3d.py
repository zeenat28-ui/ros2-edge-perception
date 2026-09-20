"""
Tier-1 3D Multi-Object Tracking Engine (3D Kalman Filter & Association).

Engineered for Autonomous Mobile Robots (AMRs) and Autonomous Vehicles.
Features:
- Constant-Velocity 3D Kalman Filter: State [x, y, z, vx, vy, vz, sx, sy, sz]
- Track Lifecycle State Machine: Tentative -> Confirmed -> Lost -> Deleted
- Occlusion Handling: Predicts obstacle trajectory across sensor dropouts
- Metric Velocity Vector Estimation: Computes 3D velocity in m/s
- Time-To-Collision (TTC) Risk Evaluation for predictive collision avoidance
- Temporal Association via 3D Euclidean & Bounding Box Overlap
"""

import math
import time
from typing import Dict, List, Optional, Tuple
import numpy as np


class Track3D:
    """Represents a single tracked dynamic 3D obstacle."""

    _next_id = 1

    def __init__(self, detection: dict, timestamp: float):
        self.track_id = Track3D._next_id
        Track3D._next_id += 1

        self.class_name = detection["class_name"]
        self.class_id = detection["class_id"]
        self.score = detection["score"]

        # State vector: [x, y, z, vx, vy, vz, sx, sy, sz]
        self.x = np.zeros(9, dtype=np.float32)
        self.x[0] = detection["x"]
        self.x[1] = detection["y"]
        self.x[2] = detection["z"]
        self.x[3] = 0.0  # vx
        self.x[4] = 0.0  # vy
        self.x[5] = 0.0  # vz
        self.x[6] = detection["size_x"]
        self.x[7] = detection["size_y"]
        self.x[8] = detection["size_z"]

        # State Covariance Matrix P (9x9)
        self.P = np.diag([
            0.1, 0.1, 0.1,    # Positional uncertainty (0.1m)
            1.0, 1.0, 1.0,    # Velocity uncertainty (1.0 m/s)
            0.05, 0.05, 0.05  # Size uncertainty
        ]).astype(np.float32)

        # Process Noise Q
        self.Q_pos = 0.05
        self.Q_vel = 0.5
        self.Q_size = 0.01

        # Measurement Noise R (Position & Size)
        self.R = np.diag([0.05, 0.05, 0.1, 0.05, 0.05, 0.05]).astype(np.float32)

        # Lifecycle management
        self.hits = 1
        self.age = 1
        self.lost_frames = 0
        self.confirmed = False
        self.last_update_time = timestamp

    def predict(self, dt: float):
        """Kalman Filter Prediction Step."""
        dt = max(0.001, min(0.5, dt))

        # State transition matrix F
        F = np.eye(9, dtype=np.float32)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        # Extrapolate state: x = F * x
        self.x = F @ self.x

        # Process noise covariance Q
        Q = np.diag([
            self.Q_pos * dt, self.Q_pos * dt, self.Q_pos * dt,
            self.Q_vel * dt, self.Q_vel * dt, self.Q_vel * dt,
            self.Q_size * dt, self.Q_size * dt, self.Q_size * dt,
        ]).astype(np.float32)

        # Extrapolate covariance: P = F * P * F^T + Q
        self.P = F @ self.P @ F.T + Q

        self.age += 1
        self.lost_frames += 1

    def update(self, detection: dict, timestamp: float):
        """Kalman Filter Measurement Update Step."""
        self.score = detection["score"]

        # Measurement vector z: [x, y, z, sx, sy, sz]
        z = np.array([
            detection["x"], detection["y"], detection["z"],
            detection["size_x"], detection["size_y"], detection["size_z"]
        ], dtype=np.float32)

        # Measurement matrix H (6x9)
        H = np.zeros((6, 9), dtype=np.float32)
        H[0, 0] = 1.0  # x
        H[1, 1] = 1.0  # y
        H[2, 2] = 1.0  # z
        H[3, 6] = 1.0  # sx
        H[4, 7] = 1.0  # sy
        H[5, 8] = 1.0  # sz

        # Innovation: y = z - H*x
        y = z - H @ self.x

        # Innovation covariance: S = H * P * H^T + R
        S = H @ self.P @ H.T + self.R

        # Optimal Kalman Gain: K = P * H^T * S^-1
        K = self.P @ H.T @ np.linalg.inv(S)

        # Updated state: x = x + K*y
        self.x = self.x + K @ y

        # Updated covariance: P = (I - K*H) * P
        I = np.eye(9, dtype=np.float32)
        self.P = (I - K @ H) @ self.P

        self.hits += 1
        self.lost_frames = 0
        if self.hits >= 2:
            self.confirmed = True
        self.last_update_time = timestamp

    def predict_future_trajectory(self, horizon_seconds: float = 2.0, steps: int = 4) -> List[Tuple[float, float, float]]:
        """Forecast future 3D positions for predictive collision avoidance."""
        future_poses = []
        dt_step = horizon_seconds / steps
        for i in range(1, steps + 1):
            t_future = i * dt_step
            fx = self.x[0] + self.x[3] * t_future
            fy = self.x[1] + self.x[4] * t_future
            fz = self.x[2] + self.x[5] * t_future
            future_poses.append((float(fx), float(fy), float(fz)))
        return future_poses

    def compute_ttc(self) -> Optional[float]:
        """
        Compute Time-To-Collision (TTC) in seconds.
        TTC = Distance / Approach_Speed (if moving towards robot, i.e., vz < 0).
        """
        z_dist = self.x[2]
        vz = self.x[5]
        if vz < -0.2 and z_dist > 0.1:  # Approaching at > 0.2 m/s
            return float(z_dist / abs(vz))
        return None

    def compute_orientation(self) -> Tuple[float, float, float, float, float]:
        """
        Compute 3D oriented bounding box heading (yaw) and quaternion.
        Aligned with velocity vector in ground plane (X-Z in camera frame).
        Returns: (yaw, qx, qy, qz, qw)
        """
        vx = self.x[3]
        vz = self.x[5]
        speed_sq = vx * vx + vz * vz
        if speed_sq > 0.04:  # > 0.2 m/s
            yaw = float(math.atan2(vx, vz))
            qx = 0.0
            qy = float(math.sin(yaw / 2.0))
            qz = 0.0
            qw = float(math.cos(yaw / 2.0))
        else:
            yaw = 0.0
            qx = 0.0
            qy = 0.0
            qz = 0.0
            qw = 1.0
        return yaw, qx, qy, qz, qw


class MultiObjectTracker3D:
    """
    Tier-1 3D Multi-Object Tracker.
    Manages track lifecycles, association, and predictive obstacle state estimation.
    """

    def __init__(
        self,
        max_lost_frames: int = 5,
        match_distance_threshold: float = 1.2,  # meters
    ):
        self.max_lost_frames = max_lost_frames
        self.match_dist_thresh = match_distance_threshold
        self.tracks: Dict[int, Track3D] = {}
        self.last_timestamp: Optional[float] = None

    def update(self, detections_3d: List[dict], timestamp: float) -> List[dict]:
        """
        Run predict -> associate -> update lifecycle.
        Returns list of active, confirmed 3D tracks with persistent IDs and velocities.
        """
        # Calculate elapsed dt
        if self.last_timestamp is None:
            dt = 0.033
        else:
            dt = max(0.001, timestamp - self.last_timestamp)
        self.last_timestamp = timestamp

        # 1. Prediction step for all existing tracks
        for track in self.tracks.values():
            track.predict(dt)

        # 2. Association step (Greedy nearest-neighbor matching in 3D Euclidean space)
        track_ids = list(self.tracks.keys())
        matched_tracks = set()
        matched_detections = set()

        if len(track_ids) > 0 and len(detections_3d) > 0:
            # Construct distance matrix
            cost_matrix = np.zeros((len(track_ids), len(detections_3d)), dtype=np.float32)
            for i, tid in enumerate(track_ids):
                track_pos = self.tracks[tid].x[:3]
                for j, det in enumerate(detections_3d):
                    det_pos = np.array([det["x"], det["y"], det["z"]], dtype=np.float32)
                    dist = np.linalg.norm(track_pos - det_pos)
                    # Penalize class mismatch
                    if det["class_name"] != self.tracks[tid].class_name:
                        dist += 10.0
                    cost_matrix[i, j] = dist

            # Try Hungarian assignment, fallback to greedy
            try:
                from scipy.optimize import linear_sum_assignment
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r, c] <= self.match_dist_thresh:
                        matched_tracks.add(track_ids[r])
                        matched_detections.add(c)
                        self.tracks[track_ids[r]].update(detections_3d[c], timestamp)
            except ImportError:
                # Greedy association
                for i, tid in enumerate(track_ids):
                    best_j = -1
                    min_cost = self.match_dist_thresh
                    for j, det in enumerate(detections_3d):
                        if j not in matched_detections and cost_matrix[i, j] < min_cost:
                            min_cost = cost_matrix[i, j]
                            best_j = j
                    if best_j >= 0:
                        matched_tracks.add(tid)
                        matched_detections.add(best_j)
                        self.tracks[tid].update(detections_3d[best_j], timestamp)

        # 3. Create new tracks for unmatched detections
        for j, det in enumerate(detections_3d):
            if j not in matched_detections:
                new_track = Track3D(det, timestamp)
                self.tracks[new_track.track_id] = new_track

        # 4. Prune dead tracks (lost for > max_lost_frames)
        dead_ids = [
            tid for tid, track in self.tracks.items()
            if track.lost_frames > self.max_lost_frames
        ]
        for tid in dead_ids:
            del self.tracks[tid]

        # 5. Format active tracked objects output
        tracked_output = []
        for track in self.tracks.values():
            if track.confirmed:
                future_trajectory = track.predict_future_trajectory(horizon_seconds=2.0, steps=4)
                ttc = track.compute_ttc()
                yaw, qx, qy, qz, qw = track.compute_orientation()

                tracked_output.append(
                    {
                        "track_id": track.track_id,
                        "class_name": track.class_name,
                        "class_id": track.class_id,
                        "score": track.score,
                        "x": float(track.x[0]),
                        "y": float(track.x[1]),
                        "z": float(track.x[2]),
                        "vx": float(track.x[3]),
                        "vy": float(track.x[4]),
                        "vz": float(track.x[5]),
                        "size_x": float(track.x[6]),
                        "size_y": float(track.x[7]),
                        "size_z": float(track.x[8]),
                        "yaw": yaw,
                        "qx": qx,
                        "qy": qy,
                        "qz": qz,
                        "qw": qw,
                        "future_trajectory": future_trajectory,
                        "ttc": ttc,
                        "lost_frames": track.lost_frames,
                    }
                )

        return tracked_output

