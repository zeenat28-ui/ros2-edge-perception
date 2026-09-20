"""
Unit and Integration Tests for 3D Semantic Voxel Occupancy Grid Engine.
"""

import numpy as np
import pytest
from ros2_edge_perception.occupancy_node import VoxelOccupancyGrid3D

def test_voxel_grid_initialization():
    grid = VoxelOccupancyGrid3D(size_x=20.0, size_y=4.0, size_z=50.0, resolution=0.2)
    assert grid.dim_x == 100
    assert grid.dim_y == 20
    assert grid.dim_z == 250
    assert grid.grid.shape == (250, 20, 100)
    assert np.all(grid.grid == VoxelOccupancyGrid3D.VOXEL_FREE)

def test_world_to_grid_conversion():
    grid = VoxelOccupancyGrid3D(size_x=20.0, size_y=4.0, size_z=50.0, resolution=0.2)
    # Origin is at (-10.0, -2.0, 0.0)
    coord = grid.world_to_grid(0.0, 0.0, 10.0)
    assert coord is not None
    ix, iy, iz = coord
    assert ix == 50 # centered in X
    assert iy == 10 # centered in Y
    assert iz == 50 # 10.0 / 0.2

    # Out of bounds test
    assert grid.world_to_grid(-15.0, 0.0, 10.0) is None
    assert grid.world_to_grid(0.0, 0.0, -5.0) is None

def test_bounding_box_insertion_and_flow():
    grid = VoxelOccupancyGrid3D(size_x=20.0, size_y=4.0, size_z=50.0, resolution=0.2)
    # Insert dynamic vehicle at (x=2.0, y=0.0, z=20.0) with vz = -5.0 m/s
    grid.insert_bounding_box(cx=2.0, cy=0.0, cz=20.0, w=2.0, h=1.6, l=4.0,
                             vx=0.0, vy=0.0, vz=-5.0,
                             voxel_type=VoxelOccupancyGrid3D.VOXEL_DYNAMIC_OBSTACLE)

    coord = grid.world_to_grid(2.0, 0.0, 20.0)
    assert coord is not None
    ix, iy, iz = coord
    assert grid.grid[iz, iy, ix] == VoxelOccupancyGrid3D.VOXEL_DYNAMIC_OBSTACLE
    assert grid.flow_z[iz, iy, ix] == -5.0

def test_bev_costmap_generation():
    grid = VoxelOccupancyGrid3D(size_x=20.0, size_y=4.0, size_z=50.0, resolution=0.2)
    # Insert dynamic obstacle
    grid.insert_bounding_box(cx=0.0, cy=0.0, cz=15.0, w=2.0, h=1.6, l=4.0,
                             voxel_type=VoxelOccupancyGrid3D.VOXEL_DYNAMIC_OBSTACLE)

    bev = grid.generate_bev_costmap()
    assert bev.shape == (250, 100)
    
    # Check that center at z=15m has lethal cost 254
    iz = int(15.0 / 0.2)
    ix = int((0.0 - (-10.0)) / 0.2)
    assert bev[iz, ix] == 254
    # Far away cell should be 0 (free)
    assert bev[5, 5] == 0
