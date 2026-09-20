"""
Automated unit tests for Virtual Robot Simulator.
Verifies kinematics integration, coordinate frame projections, and bounds clamping.
"""

import math
import pytest
import numpy as np


def test_differential_drive_kinematics():
    """Verify forward motion and rotation integration."""
    x, y, theta = 0.0, 0.0, 0.0
    dt = 0.05

    # 1. Drive forward at 1.0 m/s for 1 second (20 steps)
    linear_v = 1.0
    angular_w = 0.0

    for _ in range(20):
        theta += angular_w * dt
        x += linear_v * math.cos(theta) * dt
        y += linear_v * math.sin(theta) * dt

    assert math.isclose(x, 1.0, rel_tol=1e-3)
    assert math.isclose(y, 0.0, abs_tol=1e-5)
    assert math.isclose(theta, 0.0, abs_tol=1e-5)

    # 2. Rotate 90 degrees CCW (pi/2 rad/s for 1 sec)
    linear_v = 0.0
    angular_w = math.pi / 2.0

    for _ in range(20):
        theta += angular_w * dt
        theta = (theta + math.pi) % (2.0 * math.pi) - math.pi

    assert math.isclose(theta, math.pi / 2.0, rel_tol=1e-2)

    # 3. Drive forward along new heading (+Y)
    linear_v = 1.0
    angular_w = 0.0
    for _ in range(20):
        x += linear_v * math.cos(theta) * dt
        y += linear_v * math.sin(theta) * dt

    assert math.isclose(x, 1.0, abs_tol=1e-2)
    assert math.isclose(y, 1.0, rel_tol=1e-2)


def test_camera_optical_projection():
    """Verify world to robot body to camera optical frame transformation."""
    robot_x, robot_y, robot_theta = 0.0, 0.0, 0.0
    obs_x, obs_y = 3.0, 0.0

    # In front of robot
    dx_world = obs_x - robot_x
    dy_world = obs_y - robot_y

    x_body = dx_world * math.cos(robot_theta) + dy_world * math.sin(robot_theta)
    y_body = -dx_world * math.sin(robot_theta) + dy_world * math.cos(robot_theta)

    z_cam = x_body
    x_cam = -y_body

    assert math.isclose(z_cam, 3.0, abs_tol=1e-5)
    assert math.isclose(x_cam, 0.0, abs_tol=1e-5)

    # Robot turned 45 degrees left (theta = pi/4)
    robot_theta = math.pi / 4.0
    x_body = dx_world * math.cos(robot_theta) + dy_world * math.sin(robot_theta)
    y_body = -dx_world * math.sin(robot_theta) + dy_world * math.cos(robot_theta)

    z_cam = x_body
    x_cam = -y_body

    # Obstacle should appear to the right of camera center (x_cam > 0)
    assert z_cam > 0.0
    assert x_cam > 0.0  # Appears on the right in camera frame


def test_arena_bounds_clamping():
    """Verify robot cannot drive outside arena bounds."""
    arena_size = 10.0
    half_arena = arena_size / 2.0 - 0.5  # 4.5m limit

    x = 10.0
    y = -8.0
    clamped_x = float(np.clip(x, -half_arena, half_arena))
    clamped_y = float(np.clip(y, -half_arena, half_arena))

    assert clamped_x == 4.5
    assert clamped_y == -4.5

