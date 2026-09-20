#!/usr/bin/env python3
"""
Unit tests for Neural 3D Occupancy & Volumetric Feature Network (OccNet).
"""

import pytest
import numpy as np
from ros2_edge_perception.neural_occupancy_network import (
    Neural3DOccupancyNetwork,
    VoxelSemanticClass,
    OccupancyGrid3D
)


@pytest.fixture
def occ_net():
    return Neural3DOccupancyNetwork(
        x_range=(0.0, 10.0),
        y_range=(-4.0, 4.0),
        z_range=(-0.4, 2.0),
        voxel_size_m=0.10,
        robot_radius_m=0.40
    )


def test_grid_initialization(occ_net):
    """Test 3D voxel grid dimensions."""
    assert occ_net.nx == 100
    assert occ_net.ny == 80
    assert occ_net.nz == 24


def test_floor_plane_semantic_classification(occ_net):
    """Floor level must be classified as DRIVABLE_FLOOR."""
    grid = occ_net.predict_occupancy()
    floor_z_idx = int(np.floor((0.0 - occ_net.z_min) / occ_net.voxel_size))
    # Check that floor voxels have DRIVABLE_FLOOR label
    assert np.all(grid.semantic_labels[:, :, floor_z_idx] == VoxelSemanticClass.DRIVABLE_FLOOR)


def test_low_lying_hazard_voxelization(occ_net):
    """Points at z=0.04m (loose cable) must be classified as LOW_LYING_HAZARD."""
    # Cable at x=3.0, y=0.0, z=0.04m
    pts = np.array([
        [3.0, 0.0, 0.04],
        [3.0, 0.1, 0.04],
        [3.0, -0.1, 0.04]
    ])
    grid = occ_net.predict_occupancy(point_cloud=pts)
    indices = occ_net.world_to_voxel_indices(pts)
    for ix, iy, iz in indices:
        assert grid.semantic_labels[ix, iy, iz] == VoxelSemanticClass.LOW_LYING_HAZARD
        assert grid.occupancy_probs[ix, iy, iz] > 0.0


def test_negative_dropoff_voxelization(occ_net):
    """Points at z=-0.25m (dock edge) must be classified as NEGATIVE_DROP_OFF."""
    pts = np.array([
        [5.0, 0.0, -0.25],
        [5.0, 0.2, -0.25]
    ])
    grid = occ_net.predict_occupancy(point_cloud=pts)
    indices = occ_net.world_to_voxel_indices(pts)
    for ix, iy, iz in indices:
        assert grid.semantic_labels[ix, iy, iz] == VoxelSemanticClass.NEGATIVE_DROP_OFF


def test_bev_costmap_inflation(occ_net):
    """BEV costmap must have shape (nx, ny) with values in [0, 1]."""
    dynamic_agents = [{
        "centroid_3d": [4.0, 0.0, 1.0],
        "bbox_dimensions": [1.0, 1.0, 1.8]
    }]
    grid = occ_net.predict_occupancy(dynamic_agents=dynamic_agents)
    assert grid.bev_costmap.shape == (100, 80)
    assert np.max(grid.bev_costmap) > 0.5
    assert np.all((grid.bev_costmap >= 0.0) & (grid.bev_costmap <= 1.0))

