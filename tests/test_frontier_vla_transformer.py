#!/usr/bin/env python3
"""
=============================================================================
AURA-DRIVE™ 2026: UNIT TESTS FOR FRONTIER VLA TRANSFORMER & KINODYNAMICS
=============================================================================
"""

import math
import pytest
import numpy as np

from ros2_edge_perception.frontier_vla_transformer import (
    FrontierVLATransformer,
    MultiHeadAttention,
    TransformerBlock
)
from ros2_edge_perception.kinodynamic_amr_model import KinodynamicAMRModel, AMRKinodynamicState


class TestFrontierVLATransformer:
    """Verifies attention mechanisms, flow-matching ODE, and semantic steering."""

    def test_multihead_attention_shape_and_scale(self):
        mha = MultiHeadAttention(d_model=128, n_heads=4)
        q = np.random.randn(16, 128).astype(np.float32)
        k = np.random.randn(24, 128).astype(np.float32)
        v = np.random.randn(24, 128).astype(np.float32)

        out, attn_weights = mha.forward(q, k, v)
        assert out.shape == (16, 128)
        assert attn_weights.shape == (4, 16, 24)
        # Sum of attention weights over keys should be ~1.0
        assert np.allclose(np.sum(attn_weights, axis=-1), 1.0, atol=1e-5)

    def test_transformer_block_residual_and_norm(self):
        block = TransformerBlock(d_model=128, n_heads=4)
        x = np.random.randn(16, 128).astype(np.float32)
        out, _ = block.forward(x)
        assert out.shape == (16, 128)
        assert not np.isnan(out).any()

    def test_vla_transformer_flow_matching_crystallization(self):
        vla = FrontierVLATransformer(action_horizon=16, diffusion_steps=16)
        img = np.zeros((224, 224, 3), dtype=np.uint8)

        rollout = vla.sample_trajectory(img, "bypass obstacle on left", solver="rk4")
        assert rollout.action_horizon == 16
        assert len(rollout.diffusion_history) == 17
        assert rollout.final_trajectory.shape == (16, 3)
        assert rollout.final_velocities.shape == (16, 2)
        assert rollout.task_intent == "EVASION_BYPASS_LEFT"
        assert rollout.final_trajectory[-1, 1] > 0.0  # Left steering

    def test_vla_transformer_emergency_halt(self):
        vla = FrontierVLATransformer()
        img = np.zeros((224, 224, 3), dtype=np.uint8)

        rollout = vla.sample_trajectory(img, "emergency halt immediately")
        assert rollout.task_intent == "EMERGENCY_HALT"
        assert np.all(rollout.final_trajectory == 0.0)
        assert np.all(rollout.final_velocities == 0.0)
        assert rollout.confidence >= 0.98


class TestKinodynamicAMRModel:
    """Verifies non-holonomic kinematics, torque saturation, and payload adaptation."""

    def test_differential_drive_forward_step(self):
        model = KinodynamicAMRModel()
        state = AMRKinodynamicState(x=0.0, y=0.0, theta=0.0, v=0.0, omega=0.0, a_lin=0.0, a_ang=0.0, wheel_slip_left=0.0, wheel_slip_right=0.0)

        next_state = model.step(state, cmd_v=1.0, cmd_omega=0.0, dt=0.05)
        assert next_state.x > 0.0
        assert next_state.v > 0.0
        assert abs(next_state.y) < 1e-4

    def test_centrifugal_acceleration_clamping(self):
        model = KinodynamicAMRModel(max_centrifugal_accel_mps2=1.2)
        state = AMRKinodynamicState(x=0.0, y=0.0, theta=0.0, v=1.5, omega=1.5, a_lin=0.0, a_ang=0.0, wheel_slip_left=0.0, wheel_slip_right=0.0)

        # Commanding v=1.5 and omega=1.5 yields v*omega = 2.25 > 1.2
        next_state = model.step(state, cmd_v=1.5, cmd_omega=1.5, dt=0.05)
        # Should be clamped
        assert abs(next_state.v * next_state.omega) <= 1.25

    def test_payload_mass_adaptation_reduces_acceleration(self):
        model_empty = KinodynamicAMRModel(base_mass_kg=85.0)
        model_loaded = KinodynamicAMRModel(base_mass_kg=85.0)
        model_loaded.set_payload_mass(200.0)  # Total 285 kg

        v_empty, _, a_empty, _ = model_empty.apply_kinodynamic_limits(1.5, 0.0, 0.0, 0.0, dt=0.05)
        v_loaded, _, a_loaded, _ = model_loaded.apply_kinodynamic_limits(1.5, 0.0, 0.0, 0.0, dt=0.05)

        # Loaded robot must have lower or equal acceleration due to traction/inertia limits
        assert a_loaded <= a_empty

