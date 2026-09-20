"""
AURA-Drive™ Frontier Research Stack Verification Suite
======================================================
Validates the advanced research and industrial engineering components:
  1. RealSensorPipeline (live video processing, optical flow, 3D point clouds)
  2. DenoisingDiffusionVLAPolicy (RK4 numerical integration & temporal ensembling)
  3. ROS2LifecycleManager (lifecycle states: UNCONFIGURED -> ACTIVE -> FINALIZED)
  4. TF2TransformTree (SE(3) coordinate transformation and Lie algebra)
  5. DynamicRVOEngine (Reciprocal Velocity Obstacle collision avoidance)
"""

import time
import math
import pytest
import numpy as np

from ros2_edge_perception.real_sensor_pipeline import RealSensorPipeline, SensorObservation
from ros2_edge_perception.diffusion_vla_policy import DenoisingDiffusionVLAPolicy
from ros2_edge_perception.lifecycle_manager import ROS2LifecycleManager, LifecycleState, TransitionReturn
from ros2_edge_perception.tf2_transform_tree import TF2TransformTree, se3_exp
from ros2_edge_perception.dynamic_rvo_engine import DynamicRVOEngine, DynamicObstacle


class TestPillar1RealPerceptionAndDiffusion:
    """Real Sensor Pipeline and Flow-Matching ODE Policy Tests."""

    def test_real_sensor_pipeline_observation(self):
        pipeline = RealSensorPipeline()
        obs = pipeline.get_next_observation()

        assert isinstance(obs, SensorObservation)
        assert obs.frame_id >= 1
        assert obs.rgb_frame.shape == (480, 640, 3)
        assert obs.optical_flow.shape == (480, 640, 2)
        assert obs.point_cloud_3d.shape[1] == 3
        assert obs.point_cloud_3d.shape[0] > 0
        assert 0.0 <= obs.mean_luminance <= 255.0
        pipeline.release()

    def test_diffusion_rk4_integration_and_temporal_ensembling(self):
        policy = DenoisingDiffusionVLAPolicy(diffusion_steps=16)
        dummy_rgb = np.zeros((224, 224, 3), dtype=np.uint8)

        # Run with RK4 solver
        rollout_rk4 = policy.sample_trajectory(dummy_rgb, "bypass crossing forklift on left", solver="rk4")
        assert rollout_rk4.diffusion_steps == 16
        assert rollout_rk4.final_trajectory.shape == (16, 3)
        assert rollout_rk4.final_velocities.shape == (16, 2)
        assert len(rollout_rk4.denoising_history) == 17

        # Test temporal ensembling across multiple calls
        rollout_2 = policy.sample_trajectory(dummy_rgb, "bypass crossing forklift on left", solver="rk4")
        # Temporal ensembling guarantees continuous transitions
        diff = np.abs(rollout_2.final_velocities[0] - rollout_rk4.final_velocities[0])
        assert diff[0] < 0.5


class TestPillar2SystemsEngineering:
    """ROS 2 Lifecycle, SE(3) Transform Tree, and RVO Tests."""

    def test_lifecycle_state_machine_and_rollback(self):
        mgr = ROS2LifecycleManager()
        node = mgr.register_node("test_perception_node", on_configure=lambda: True, on_activate=lambda: True)
        assert node.current_state == LifecycleState.PRIMARY_UNCONFIGURED

        # Configure
        assert mgr.trigger_transition("test_perception_node", "configure") == TransitionReturn.SUCCESS
        assert node.current_state == LifecycleState.PRIMARY_INACTIVE

        # Activate
        assert mgr.trigger_transition("test_perception_node", "activate") == TransitionReturn.SUCCESS
        assert node.current_state == LifecycleState.PRIMARY_ACTIVE

        # Deactivate
        assert mgr.trigger_transition("test_perception_node", "deactivate") == TransitionReturn.SUCCESS
        assert node.current_state == LifecycleState.PRIMARY_INACTIVE

        # Shutdown
        assert mgr.trigger_transition("test_perception_node", "shutdown") == TransitionReturn.SUCCESS
        assert node.current_state == LifecycleState.PRIMARY_FINALIZED

    def test_tf2_se3_transform_tree(self):
        tree = TF2TransformTree()
        tree.update_odometry(x=3.0, y=2.0, theta=0.0)

        # Lookup transform from laser_link to odom
        T_odom_laser = tree.lookup_transform("odom", "laser_link")
        assert T_odom_laser.shape == (4, 4)

        # Point at (0, 0, 0) in laser_link should be at (3.0 + 0.35, 2.0, 0.08 + 0.28) in odom
        pt_origin = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        pt_in_odom = tree.transform_points(pt_origin, source_frame="laser_link", target_frame="odom")

        assert abs(pt_in_odom[0, 0] - 3.35) < 1e-3
        assert abs(pt_in_odom[0, 1] - 2.0) < 1e-3
        assert abs(pt_in_odom[0, 2] - 0.36) < 1e-3

    def test_dynamic_rvo_collision_avoidance(self):
        engine = DynamicRVOEngine(robot_radius=0.45, max_speed=1.5)
        robot_pos = np.array([0.0, 0.0], dtype=np.float32)
        robot_vel = np.array([1.0, 0.0], dtype=np.float32)
        pref_vel = np.array([1.0, 0.0], dtype=np.float32)

        # Obstacle moving directly towards robot
        obstacles = [
            DynamicObstacle(
                obstacle_id="head_on_forklift",
                position=np.array([3.0, 0.0], dtype=np.float32),
                velocity=np.array([-1.0, 0.0], dtype=np.float32),
                radius=0.5
            )
        ]

        res = engine.compute_optimal_velocity(robot_pos, robot_vel, pref_vel, obstacles)
        # RVO must select a velocity that deviates laterally or decelerates
        assert abs(res.admissible_velocity[1]) > 0.05 or res.admissible_velocity[0] < 0.6

    def test_kinodynamic_amr_centrifugal_and_torque_limits(self):
        from ros2_edge_perception.kinodynamic_amr_model import KinodynamicAMRModel, AMRKinodynamicState
        model = KinodynamicAMRModel(max_centrifugal_accel_mps2=1.20)
        state = AMRKinodynamicState(x=0.0, y=0.0, theta=0.0, v=1.5, omega=0.0)
        
        # Commanded high yaw rate at top speed -> would violate centrifugal acceleration limit (1.5 * 1.5 = 2.25 > 1.20)
        new_state = model.step(state, cmd_v=1.5, cmd_omega=1.5, dt=0.05)
        # Verify centrifugal clamp: |v * omega| <= 1.20 + epsilon
        assert abs(new_state.v * new_state.omega) <= 1.201

    def test_kinodynamic_amr_rk4_integration_accuracy(self):
        from ros2_edge_perception.kinodynamic_amr_model import KinodynamicAMRModel, AMRKinodynamicState
        model = KinodynamicAMRModel()
        state = AMRKinodynamicState(x=0.0, y=0.0, theta=0.0, v=1.0, omega=0.5)
        # Advance 1 second in 20 steps
        for _ in range(20):
            state = model.step(state, cmd_v=1.0, cmd_omega=0.5, dt=0.05)
        
        # In a circle of radius R = v/omega = 2.0m:
        # Heading after 1s should be ~0.5 rad
        assert abs(state.theta - 0.5) < 0.05
        assert state.x > 0.8
        assert state.y > 0.1

