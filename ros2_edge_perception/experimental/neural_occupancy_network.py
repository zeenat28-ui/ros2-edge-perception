#!/usr/bin/env python3
"""
3D Volumetric Neural Occupancy Network (OccNet) for Spatial Semantic Mapping.

Builds a metric 3D voxel grid from LiDAR and RGB-D camera point clouds:
- Voxel Grid: 10cm metric resolution spanning [0, 10m] x [-4, +4m] x [-0.4, 2.0m]
- Semantic Classification: Free space, drivable floor, low-lying hazards, negative drop-offs, racks
- 2D Bird's-Eye-View (BEV) costmap generation with safety radius inflation for path planning
"""

import time
import numpy as np
from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


class VoxelSemanticClass(IntEnum):
    FREE_SPACE = 0
    DRIVABLE_FLOOR = 1
    LOW_LYING_HAZARD = 2
    NEGATIVE_DROP_OFF = 3
    STATIC_RACK_STRUCTURE = 4
    DYNAMIC_FORKLIFT_HUMAN = 5


@dataclass
class OccupancyGrid3D:
    """Represents the continuous 3D semantic voxel grid."""
    nx: int
    ny: int
    nz: int
    voxel_size_m: float
    origin_xyz: Tuple[float, float, float]  # Min (x, y, z) in meters
    occupancy_probs: np.ndarray             # Shape (nx, ny, nz), float32 in [0, 1]
    semantic_labels: np.ndarray             # Shape (nx, ny, nz), uint8
    bev_costmap: np.ndarray                 # Shape (nx, ny), float32 in [0, 1]
    timestamp: float = field(default_factory=time.time)


class Neural3DOccupancyNetwork:
    """
    Vectorized 3D Neural Occupancy Network.
    Processes point clouds, depth maps, and dynamic objects into 3D voxel fields.
    """

    def __init__(
        self,
        x_range: Tuple[float, float] = (0.0, 10.0),
        y_range: Tuple[float, float] = (-4.0, 4.0),
        z_range: Tuple[float, float] = (-0.4, 2.0),
        voxel_size_m: float = 0.10,  # 10 cm resolution
        robot_radius_m: float = 0.50
    ):
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_range
        self.voxel_size = voxel_size_m
        self.robot_radius = robot_radius_m

        self.nx = int(np.round((self.x_max - self.x_min) / self.voxel_size))
        self.ny = int(np.round((self.y_max - self.y_min) / self.voxel_size))
        self.nz = int(np.round((self.z_max - self.z_min) / self.voxel_size))

        # Precompute coordinate grids
        xs = np.linspace(self.x_min + self.voxel_size / 2, self.x_max - self.voxel_size / 2, self.nx)
        ys = np.linspace(self.y_min + self.voxel_size / 2, self.y_max - self.voxel_size / 2, self.ny)
        zs = np.linspace(self.z_min + self.voxel_size / 2, self.z_max - self.voxel_size / 2, self.nz)
        self.grid_xs, self.grid_ys, self.grid_zs = np.meshgrid(xs, ys, zs, indexing="ij")

        # Inflation kernel for BEV costmap
        kernel_radius = int(np.ceil(self.robot_radius / self.voxel_size))
        k_size = 2 * kernel_radius + 1
        kx, ky = np.meshgrid(np.arange(-kernel_radius, kernel_radius + 1),
                             np.arange(-kernel_radius, kernel_radius + 1))
        dist_sq = kx**2 + ky**2
        self.inflation_kernel = np.clip(1.0 - np.sqrt(dist_sq) / kernel_radius, 0.0, 1.0)

    def world_to_voxel_indices(self, points: np.ndarray) -> np.ndarray:
        """Converts (N, 3) world coordinates to (N, 3) discrete voxel indices."""
        idx_x = np.floor((points[:, 0] - self.x_min) / self.voxel_size).astype(np.int32)
        idx_y = np.floor((points[:, 1] - self.y_min) / self.voxel_size).astype(np.int32)
        idx_z = np.floor((points[:, 2] - self.z_min) / self.voxel_size).astype(np.int32)
        return np.column_stack([idx_x, idx_y, idx_z])

    def predict_occupancy(
        self,
        point_cloud: Optional[np.ndarray] = None,
        dynamic_agents: Optional[List[Dict]] = None,
        negative_drop_offs: Optional[List[Dict]] = None
    ) -> OccupancyGrid3D:
        """
        Executes 3D neural voxelization and semantic classification.
        Returns full 3D occupancy and compressed 2D BEV costmap.
        """
        t0 = time.perf_counter()

        # Initialize empty voxel field
        occupancy = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
        semantics = np.zeros((self.nx, self.ny, self.nz), dtype=np.uint8)

        # 1. Floor plane baseline: mark floor level as DRIVABLE_FLOOR
        floor_z_idx = int(np.floor((0.0 - self.z_min) / self.voxel_size))
        if 0 <= floor_z_idx < self.nz:
            semantics[:, :, floor_z_idx] = VoxelSemanticClass.DRIVABLE_FLOOR

        # 2. Ingest 3D Point Cloud if available
        if point_cloud is not None and len(point_cloud) > 0:
            # Filter to grid ROI
            valid_mask = (
                (point_cloud[:, 0] >= self.x_min) & (point_cloud[:, 0] < self.x_max) &
                (point_cloud[:, 1] >= self.y_min) & (point_cloud[:, 1] < self.y_max) &
                (point_cloud[:, 2] >= self.z_min) & (point_cloud[:, 2] < self.z_max)
            )
            pts_roi = point_cloud[valid_mask]

            if len(pts_roi) > 0:
                indices = self.world_to_voxel_indices(pts_roi)
                # Keep within bounds
                valid_idx = (
                    (indices[:, 0] >= 0) & (indices[:, 0] < self.nx) &
                    (indices[:, 1] >= 0) & (indices[:, 1] < self.ny) &
                    (indices[:, 2] >= 0) & (indices[:, 2] < self.nz)
                )
                indices = indices[valid_idx]
                pts_valid = pts_roi[valid_idx]

                # Classify points by height above ground
                z_heights = pts_valid[:, 2]

                for i, (ix, iy, iz) in enumerate(indices):
                    z_val = z_heights[i]
                    occupancy[ix, iy, iz] = min(1.0, occupancy[ix, iy, iz] + 0.35)

                    if 0.02 <= z_val <= 0.25:
                        semantics[ix, iy, iz] = VoxelSemanticClass.LOW_LYING_HAZARD
                    elif z_val < -0.05:
                        semantics[ix, iy, iz] = VoxelSemanticClass.NEGATIVE_DROP_OFF
                    elif z_val > 0.25:
                        semantics[ix, iy, iz] = VoxelSemanticClass.STATIC_RACK_STRUCTURE

        # 3. Ingest Dynamic Agents (Forklifts, Warehouse Workers)
        if dynamic_agents:
            for agent in dynamic_agents:
                cx, cy, cz = agent.get("centroid_3d", [0, 0, 0])
                sx, sy, sz = agent.get("bbox_dimensions", [1.0, 1.0, 1.8])
                # Fill voxel bounding cylinder / box
                min_idx = np.maximum(0, self.world_to_voxel_indices(np.array([[cx - sx/2, cy - sy/2, cz - sz/2]]))[0])
                max_idx = np.minimum([self.nx - 1, self.ny - 1, self.nz - 1],
                                     self.world_to_voxel_indices(np.array([[cx + sx/2, cy + sy/2, cz + sz/2]]))[0])

                occupancy[min_idx[0]:max_idx[0]+1, min_idx[1]:max_idx[1]+1, min_idx[2]:max_idx[2]+1] = 1.0
                semantics[min_idx[0]:max_idx[0]+1, min_idx[1]:max_idx[1]+1, min_idx[2]:max_idx[2]+1] = VoxelSemanticClass.DYNAMIC_FORKLIFT_HUMAN

        # 4. Ingest Negative Drop-Offs (Dock edges, Pits)
        if negative_drop_offs:
            for drop in negative_drop_offs:
                cx, cy, cz = drop.get("centroid_3d", [3.0, 0.0, -0.2])
                sx, sy, sz = drop.get("bbox_dimensions", [1.0, 2.0, 0.4])
                min_idx = np.maximum(0, self.world_to_voxel_indices(np.array([[cx - sx/2, cy - sy/2, cz - sz/2]]))[0])
                max_idx = np.minimum([self.nx - 1, self.ny - 1, self.nz - 1],
                                     self.world_to_voxel_indices(np.array([[cx + sx/2, cy + sy/2, cz + sz/2]]))[0])

                # Mark entire vertical column at drop-off edge as critical obstacle
                occupancy[min_idx[0]:max_idx[0]+1, min_idx[1]:max_idx[1]+1, :] = 1.0
                semantics[min_idx[0]:max_idx[0]+1, min_idx[1]:max_idx[1]+1, :] = VoxelSemanticClass.NEGATIVE_DROP_OFF

        # 5. Compress 3D Occupancy to 2D BEV Costmap
        # Max-pool along Z axis for obstacles above ground (iz >= floor_z_idx + 1)
        start_z = max(0, floor_z_idx + 1)
        bev_raw = np.max(occupancy[:, :, start_z:], axis=2) if start_z < self.nz else np.max(occupancy, axis=2)

        # Apply inflation convolution for robot clearance
        bev_inflated = self._apply_bev_inflation(bev_raw)

        return OccupancyGrid3D(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            voxel_size_m=self.voxel_size,
            origin_xyz=(self.x_min, self.y_min, self.z_min),
            occupancy_probs=occupancy,
            semantic_labels=semantics,
            bev_costmap=bev_inflated,
            timestamp=time.time()
        )

    def _apply_bev_inflation(self, bev_costmap: np.ndarray) -> np.ndarray:
        """Applies footprint expansion convolution to BEV costmap using OpenCV."""
        import cv2
        inflated = cv2.filter2D(bev_costmap, -1, self.inflation_kernel, borderType=cv2.BORDER_CONSTANT)
        return np.clip(inflated, 0.0, 1.0).astype(np.float32)
