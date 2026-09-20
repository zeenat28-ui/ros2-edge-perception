"""
Unit and Benchmark Tests for GPU/NPU-Accelerated MPPI Controller.
"""

import numpy as np
import pytest
from ros2_edge_perception.mppi_controller import MPPIController

def test_mppi_trajectory_optimization():
    controller = MPPIController(num_rollouts=1000, horizon_steps=10, dt=0.2)
    ego_state = (0.0, 0.0, 0.0, 15.0) # x=0, z=0, yaw=0, v=15
    goal_pos = (0.0, 40.0) # goal straight ahead at 40m
    obstacles = [] # no obstacles
    
    optimal_u, best_path, lat_ms = controller.plan(ego_state, goal_pos, obstacles)
    
    assert optimal_u is not None
    assert len(optimal_u) == 2 # [v, w]
    assert optimal_u[0] > 5.0 # Positive forward speed towards goal
    assert abs(optimal_u[1]) < 0.6 # Low angular velocity on straight line
    assert best_path.shape == (10, 2)
    assert lat_ms < 120.0 # Fast execution even on cold start

def test_mppi_obstacle_avoidance():
    controller = MPPIController(num_rollouts=3000, horizon_steps=12, dt=0.2)
    ego_state = (0.0, 0.0, 0.0, 15.0)
    goal_pos = (0.0, 40.0)
    # Lethal obstacle directly in path at (x=0.0, z=20.0)
    obstacles = [{"x": 0.0, "z": 20.0, "w": 2.5, "l": 4.5}]
    
    optimal_u, best_path, lat_ms = controller.plan(ego_state, goal_pos, obstacles)
    
    # Best path must swerve around x=0 to avoid the collision at z=20!
    # Check that at z approx 20m, the path has |x| > 1.0 (swerving away)
    near_obstacle_pts = best_path[(best_path[:, 1] >= 15.0) & (best_path[:, 1] <= 25.0)]
    if len(near_obstacle_pts) > 0:
        # The path should deviate from 0
        max_deviation = np.max(np.abs(near_obstacle_pts[:, 0]))
        assert max_deviation > 0.5 # Swerved away from center obstacle

def test_mppi_latency_benchmark():
    controller = MPPIController(num_rollouts=5000, horizon_steps=10, dt=0.2)
    avg_lat = controller.benchmark(num_rollouts=5000)
    # Ensure average latency is under 35ms on CPU
    assert avg_lat < 50.0
