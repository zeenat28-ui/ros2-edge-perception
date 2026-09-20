#!/usr/bin/env python3
"""
Near-Ground and Negative Obstacle Detector for Industrial AMRs (ISO 3691-4).

Detects hazards missed by standard 2D planar LiDARs:
- Low-lying positive obstacles (cables, pallet forks, wheel chocks: 2cm - 30cm)
- Negative obstacles (loading dock cliff drop-offs, floor pits, staircase descends)
- Overhanging obstacles (protruding storage racks, truck lift gates)
- Employs vectorized RANSAC ground plane fitting and height-differential clustering
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


@dataclass
class DetectedHazard:
    """Represents a classified near-ground or negative hazard."""
    hazard_id: int
    hazard_type: str  # 'POSITIVE_LOW_LYING', 'NEGATIVE_DROP_OFF', 'OVERHANGING'
    severity: str     # 'WARNING', 'CRITICAL', 'EMERGENCY'
    centroid_3d: Tuple[float, float, float]  # (x, y, z) in robot frame (m)
    bbox_dimensions: Tuple[float, float, float]  # (width, length, height) in m
    distance_m: float
    confidence: float
    ground_height_diff: float  # Elevation relative to fitted ground plane (m)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "hazard_id": self.hazard_id,
            "hazard_type": self.hazard_type,
            "severity": self.severity,
            "centroid_3d": [round(c, 3) for c in self.centroid_3d],
            "bbox_dimensions": [round(d, 3) for d in self.bbox_dimensions],
            "distance_m": round(self.distance_m, 3),
            "confidence": round(self.confidence, 3),
            "ground_height_diff": round(self.ground_height_diff, 3),
            "timestamp": self.timestamp
        }


@dataclass
class GroundPlaneModel:
    """Fitted ground plane: a*x + b*y + c*z + d = 0."""
    a: float
    b: float
    c: float
    d: float
    inlier_ratio: float
    normal: np.ndarray

    def distance_to_points(self, points: np.ndarray) -> np.ndarray:
        """Calculate signed distance from points (N, 3) to plane.
        Positive: above ground. Negative: below ground.
        """
        # Plane normal is expected to point UPWARDS (z > 0 in robot frame)
        denom = np.sqrt(self.a**2 + self.b**2 + self.c**2)
        if denom < 1e-6:
            return np.zeros(len(points))
        return (self.a * points[:, 0] + self.b * points[:, 1] + self.c * points[:, 2] + self.d) / denom


class NearGroundHazardDetector:
    """
    High-speed vectorized 3D Near-Ground and Negative Obstacle Detector.
    Processes point clouds or depth-derived point sets in Robot Frame:
      X = Forward (+X ahead)
      Y = Lateral (+Y left, -Y right)
      Z = Vertical (+Z upwards)
    """

    def __init__(
        self,
        min_cable_height_m: float = 0.02,     # 2 cm: minimum detectable positive obstacle
        max_low_lying_height_m: float = 0.35,  # 35 cm: boundary between low-lying and standard obstacle
        min_dropoff_depth_m: float = 0.05,     # 5 cm: minimum drop-off to trigger negative obstacle
        max_roi_forward_m: float = 8.0,        # 8 m forward detection zone
        max_roi_lateral_m: float = 2.5,        # 2.5 m lateral width (+/- 1.25m from center)
        ransac_iterations: int = 40,
        ransac_distance_threshold: float = 0.03,  # 3 cm plane tolerance
        cluster_tolerance_m: float = 0.15,     # 15 cm cluster radius
        min_cluster_points: int = 5
    ):
        self.min_cable_height_m = min_cable_height_m
        self.max_low_lying_height_m = max_low_lying_height_m
        self.min_dropoff_depth_m = min_dropoff_depth_m
        self.max_roi_forward_m = max_roi_forward_m
        self.max_roi_lateral_m = max_roi_lateral_m
        self.ransac_iterations = ransac_iterations
        self.ransac_distance_threshold = ransac_distance_threshold
        self.cluster_tolerance_m = cluster_tolerance_m
        self.min_cluster_points = min_cluster_points
        self._next_hazard_id = 1

    def fit_ground_plane_ransac(self, points: np.ndarray) -> Optional[GroundPlaneModel]:
        """
        Fits a ground plane to 3D point cloud using vectorized RANSAC.
        Assumes ground plane has normal roughly aligned with +Z axis [0, 0, 1].
        """
        if len(points) < 10:
            return None

        # Filter candidate ground points (near z ~ 0)
        z_filter = (points[:, 2] > -0.5) & (points[:, 2] < 0.3)
        candidate_points = points[z_filter]
        if len(candidate_points) < 10:
            candidate_points = points

        num_points = len(candidate_points)
        best_inliers_count = 0
        best_plane = None

        # Sample random triplets
        rand_indices = np.random.randint(0, num_points, size=(self.ransac_iterations, 3))

        for idx_triplet in rand_indices:
            p1, p2, p3 = candidate_points[idx_triplet]
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm_mag = np.linalg.norm(normal)
            if norm_mag < 1e-6:
                continue

            normal = normal / norm_mag

            # Normal must point generally upwards (+Z in robot frame)
            if normal[2] < 0:
                normal = -normal

            # Check if normal is sufficiently vertical (within ~25 degrees of Z axis)
            if normal[2] < 0.85:
                continue

            d = -np.dot(normal, p1)

            # Compute distances
            distances = np.abs(np.dot(candidate_points, normal) + d)
            inliers = distances < self.ransac_distance_threshold
            inlier_count = np.sum(inliers)

            if inlier_count > best_inliers_count:
                best_inliers_count = inlier_count
                best_plane = (normal[0], normal[1], normal[2], d, inlier_count / num_points, normal)

        if best_plane is None:
            # Fallback to nominal horizontal ground plane z = 0
            return GroundPlaneModel(a=0.0, b=0.0, c=1.0, d=0.0, inlier_ratio=1.0, normal=np.array([0.0, 0.0, 1.0]))

        return GroundPlaneModel(
            a=float(best_plane[0]),
            b=float(best_plane[1]),
            c=float(best_plane[2]),
            d=float(best_plane[3]),
            inlier_ratio=float(best_plane[4]),
            normal=best_plane[5]
        )

    def detect_hazards(self, points: np.ndarray) -> Tuple[List[DetectedHazard], Optional[GroundPlaneModel]]:
        """
        Analyzes 3D point cloud for near-ground hazards and negative drop-offs.
        Returns:
            hazards: List of detected hazards classified by type and severity.
            ground_plane: Fitted ground plane model.
        """
        hazards: List[DetectedHazard] = []

        if len(points) == 0:
            return hazards, None

        # 1. Filter points to Robot Region of Interest (ROI)
        roi_mask = (
            (points[:, 0] > 0.2) & (points[:, 0] <= self.max_roi_forward_m) &
            (np.abs(points[:, 1]) <= self.max_roi_lateral_m) &
            (points[:, 2] >= -2.0) & (points[:, 2] <= 2.0)
        )
        roi_points = points[roi_mask]
        if len(roi_points) < 10:
            return hazards, None

        # 2. Fit Ground Plane via RANSAC
        ground_plane = self.fit_ground_plane_ransac(roi_points)
        if ground_plane is None:
            return hazards, None

        # 3. Calculate signed elevation relative to ground plane
        signed_heights = ground_plane.distance_to_points(roi_points)

        # 4. Identify Positive Low-Lying Hazards (Cables, Pallet Forks, Debris)
        # Height between 2cm and 35cm
        low_lying_mask = (
            (signed_heights >= self.min_cable_height_m) &
            (signed_heights <= self.max_low_lying_height_m)
        )
        low_lying_points = roi_points[low_lying_mask]

        if len(low_lying_points) >= self.min_cluster_points:
            cable_hazards = self._cluster_and_classify_hazards(
                low_lying_points,
                signed_heights[low_lying_mask],
                hazard_type="POSITIVE_LOW_LYING"
            )
            hazards.extend(cable_hazards)

        # 5. Identify Negative Drop-Offs (Dock edges, Pits)
        # Drop-offs occur when:
        # A) Explicit points measured significantly BELOW ground plane (< -5cm)
        dropoff_mask = signed_heights <= -self.min_dropoff_depth_m
        dropoff_points = roi_points[dropoff_mask]

        if len(dropoff_points) >= self.min_cluster_points:
            drop_hazards = self._cluster_and_classify_hazards(
                dropoff_points,
                signed_heights[dropoff_mask],
                hazard_type="NEGATIVE_DROP_OFF"
            )
            hazards.extend(drop_hazards)

        # B) Void / Gap analysis in expected ground ROI (Loading dock edge)
        void_hazards = self._detect_ground_voids(roi_points, ground_plane)
        hazards.extend(void_hazards)

        # 6. Identify Overhanging Obstacles (Protruding Racks/Truck tails)
        # Height above 1.2m up to 2.2m
        overhang_mask = (signed_heights > 1.2) & (signed_heights <= 2.2)
        overhang_points = roi_points[overhang_mask]
        if len(overhang_points) >= self.min_cluster_points:
            overhang_hazards = self._cluster_and_classify_hazards(
                overhang_points,
                signed_heights[overhang_mask],
                hazard_type="OVERHANGING"
            )
            hazards.extend(overhang_hazards)

        return hazards, ground_plane

    def _cluster_and_classify_hazards(
        self,
        points: np.ndarray,
        height_diffs: np.ndarray,
        hazard_type: str
    ) -> List[DetectedHazard]:
        """Simple spatial grid/euclidean clustering for real-time performance."""
        hazards = []
        if len(points) == 0:
            return hazards

        # Voxelize into 20cm spatial cells for clustering
        cell_size = self.cluster_tolerance_m
        keys = np.floor(points[:, :2] / cell_size).astype(np.int32)
        unique_keys, inverse_indices = np.unique(keys, axis=0, return_inverse=True)

        for i, _ in enumerate(unique_keys):
            cluster_mask = inverse_indices == i
            cluster_pts = points[cluster_mask]
            cluster_diffs = height_diffs[cluster_mask]

            if len(cluster_pts) < self.min_cluster_points:
                continue

            centroid = np.mean(cluster_pts, axis=0)
            dist = float(np.linalg.norm(centroid[:2]))
            min_bounds = np.min(cluster_pts, axis=0)
            max_bounds = np.max(cluster_pts, axis=0)
            dimensions = max_bounds - min_bounds
            # Ensure non-zero dimensions
            dimensions = np.maximum(dimensions, np.array([0.05, 0.05, 0.02]))

            mean_diff = float(np.mean(cluster_diffs))

            # Determine severity based on distance and hazard type
            if hazard_type == "NEGATIVE_DROP_OFF":
                severity = "EMERGENCY" if dist < 3.0 else ("CRITICAL" if dist < 5.0 else "WARNING")
            elif hazard_type == "POSITIVE_LOW_LYING":
                severity = "EMERGENCY" if dist < 2.0 else ("CRITICAL" if dist < 4.0 else "WARNING")
            else:
                severity = "CRITICAL" if dist < 2.5 else "WARNING"

            # Confidence based on cluster density
            confidence = float(np.clip(len(cluster_pts) / 25.0, 0.65, 0.99))

            hazard = DetectedHazard(
                hazard_id=self._next_hazard_id,
                hazard_type=hazard_type,
                severity=severity,
                centroid_3d=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
                bbox_dimensions=(float(dimensions[0]), float(dimensions[1]), float(dimensions[2])),
                distance_m=dist,
                confidence=confidence,
                ground_height_diff=mean_diff
            )
            self._next_hazard_id += 1
            hazards.append(hazard)

        return hazards

    def _detect_ground_voids(
        self,
        roi_points: np.ndarray,
        ground_plane: GroundPlaneModel
    ) -> List[DetectedHazard]:
        """
        Detects loading dock edges and cliff voids where ground surface terminates
        within the expected corridor.
        """
        void_hazards = []
        # Filter for points that actually conform to ground level (-5cm to +10cm)
        signed_heights = ground_plane.distance_to_points(roi_points)
        ground_mask = (signed_heights >= -self.min_dropoff_depth_m) & (signed_heights <= 0.10)
        ground_pts = roi_points[ground_mask]

        # Divide forward path into 0.5m longitudinal slices
        slice_depth = 0.5
        max_dist = min(self.max_roi_forward_m, 6.0)

        for x_start in np.arange(1.0, max_dist, slice_depth):
            x_end = x_start + slice_depth
            # Check narrow vehicle lane (|y| <= 0.6m)
            slice_mask = (ground_pts[:, 0] >= x_start) & (ground_pts[:, 0] < x_end) & (np.abs(ground_pts[:, 1]) <= 0.6)
            pts_in_slice = ground_pts[slice_mask]

            if len(pts_in_slice) == 0:
                # Sudden absence of ground surface in forward trajectory!
                dist = float(x_start)
                severity = "EMERGENCY" if dist < 2.5 else "CRITICAL"
                hazard = DetectedHazard(
                    hazard_id=self._next_hazard_id,
                    hazard_type="NEGATIVE_DROP_OFF",
                    severity=severity,
                    centroid_3d=(dist + 0.25, 0.0, -0.2),
                    bbox_dimensions=(slice_depth, 1.2, 0.3),
                    distance_m=dist,
                    confidence=0.92,
                    ground_height_diff=-0.25
                )
                self._next_hazard_id += 1
                void_hazards.append(hazard)
                break  # Stop checking further once cliff edge found

        return void_hazards
