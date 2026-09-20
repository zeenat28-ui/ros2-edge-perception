"""
3D Semantic Voxel Occupancy & Traversability Engine Node.
Discretizes the physical space into 3D voxels (0.1m resolution),
estimates voxel velocity flow [vx, vy, vz], and generates 2D/3D BEV costmaps.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple

class VoxelOccupancyGrid3D:
    """Python/NumPy implementation of 3D Semantic Voxel Grid with SIMD acceleration."""

    VOXEL_FREE = 0
    VOXEL_DRIVABLE = 1
    VOXEL_STATIC_OBSTACLE = 2
    VOXEL_DYNAMIC_OBSTACLE = 3
    VOXEL_UNKNOWN = 4

    def __init__(self, size_x: float = 20.0, size_y: float = 4.0, size_z: float = 50.0, resolution: float = 0.2):
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.resolution = resolution

        self.dim_x = int(math.ceil(size_x / resolution))
        self.dim_y = int(math.ceil(size_y / resolution))
        self.dim_z = int(math.ceil(size_z / resolution))

        self.origin_x = -size_x / 2.0
        self.origin_y = -size_y / 2.0
        self.origin_z = 0.0

        # Voxel grid: (dim_z, dim_y, dim_x)
        self.grid = np.full((self.dim_z, self.dim_y, self.dim_x), self.VOXEL_FREE, dtype=np.uint8)
        self.flow_x = np.zeros((self.dim_z, self.dim_y, self.dim_x), dtype=np.float32)
        self.flow_y = np.zeros((self.dim_z, self.dim_y, self.dim_x), dtype=np.float32)
        self.flow_z = np.zeros((self.dim_z, self.dim_y, self.dim_x), dtype=np.float32)

    def world_to_grid(self, x: float, y: float, z: float) -> Optional[Tuple[int, int, int]]:
        if not (self.origin_x <= x < self.origin_x + self.size_x and
                self.origin_y <= y < self.origin_y + self.size_y and
                self.origin_z <= z < self.origin_z + self.size_z):
            return None
        ix = int((x - self.origin_x) / self.resolution)
        iy = int((y - self.origin_y) / self.resolution)
        iz = int((z - self.origin_z) / self.resolution)
        if 0 <= ix < self.dim_x and 0 <= iy < self.dim_y and 0 <= iz < self.dim_z:
            return (ix, iy, iz)
        return None

    def insert_bounding_box(self, cx: float, cy: float, cz: float,
                            w: float, h: float, l: float,
                            vx: float = 0.0, vy: float = 0.0, vz: float = 0.0,
                            voxel_type: int = VOXEL_DYNAMIC_OBSTACLE):
        """Vectorized 3D bounding box voxel insertion."""
        min_x, max_x = cx - w/2.0, cx + w/2.0
        min_y, max_y = cy - h/2.0, cy + h/2.0
        min_z, max_z = cz - l/2.0, cz + l/2.0

        ix1 = max(0, min(self.dim_x - 1, int((min_x - self.origin_x) / self.resolution)))
        ix2 = max(0, min(self.dim_x - 1, int((max_x - self.origin_x) / self.resolution)))
        iy1 = max(0, min(self.dim_y - 1, int((min_y - self.origin_y) / self.resolution)))
        iy2 = max(0, min(self.dim_y - 1, int((max_y - self.origin_y) / self.resolution)))
        iz1 = max(0, min(self.dim_z - 1, int((min_z - self.origin_z) / self.resolution)))
        iz2 = max(0, min(self.dim_z - 1, int((max_z - self.origin_z) / self.resolution)))

        if ix1 <= ix2 and iy1 <= iy2 and iz1 <= iz2:
            self.grid[iz1:iz2+1, iy1:iy2+1, ix1:ix2+1] = voxel_type
            self.flow_x[iz1:iz2+1, iy1:iy2+1, ix1:ix2+1] = vx
            self.flow_y[iz1:iz2+1, iy1:iy2+1, ix1:ix2+1] = vy
            self.flow_z[iz1:iz2+1, iy1:iy2+1, ix1:ix2+1] = vz

    def generate_bev_costmap(self) -> np.ndarray:
        """Projects 3D voxel grid down to a 2D BEV costmap (dim_z, dim_x) for MPPI."""
        # Find maximum obstacle severity across vertical height Y
        # Dynamic obstacles = 254 (lethal), Static = 200, Free = 0
        bev = np.zeros((self.dim_z, self.dim_x), dtype=np.uint8)
        
        dynamic_mask = np.any(self.grid == self.VOXEL_DYNAMIC_OBSTACLE, axis=1)
        static_mask = np.any(self.grid == self.VOXEL_STATIC_OBSTACLE, axis=1)
        
        bev[static_mask] = 200
        bev[dynamic_mask] = 254 # Lethal obstacle override
        return bev

    def reset(self):
        self.grid.fill(self.VOXEL_FREE)
        self.flow_x.fill(0.0)
        self.flow_y.fill(0.0)
        self.flow_z.fill(0.0)
