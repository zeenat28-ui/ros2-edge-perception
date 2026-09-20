#!/usr/bin/env python3
"""
Unit tests for Neural Vision-Language-Action (VLA) Policy Engine.
"""

import time
import pytest
import numpy as np
from ros2_edge_perception.neural_vla_engine import NeuralVLAEngine, VLAActionChunk


@pytest.fixture
def vla():
    return NeuralVLAEngine(embed_dim=128, num_heads=4, patch_size=16, action_horizon=16)


def test_language_tokenization(vla):
    """Test vocabulary tokenization and subword handling."""
    prompt = "navigate forward, bypass obstacle on left, dock at station"
    tokens, words = vla.tokenize_language(prompt)
    assert len(tokens) > 5
    assert "navigate" in words
    assert "left" in words
    assert "dock" in words


def test_visual_patch_projection(vla):
    """Test ViT patch extraction and projection to embedding space."""
    dummy_img = np.random.uniform(0.0, 1.0, (224, 224, 3)).astype(np.float32)
    vis_tokens = vla.forward_visual_tokens(dummy_img)
    # (224/16) * (224/16) = 14 * 14 = 196 patches
    assert vis_tokens.shape == (196, 128)


def test_cross_attention_forward(vla):
    """Test multi-head cross-attention fusion layer."""
    lang_tokens = np.random.normal(0, 1, (8, 128))
    vis_tokens = np.random.normal(0, 1, (196, 128))

    context, attn_map = vla.forward_cross_attention(lang_tokens, vis_tokens)
    assert context.shape == (8, 128)
    assert attn_map.shape == (8, 196)
    # Attention weights must sum to 1.0 across visual tokens
    assert np.allclose(np.sum(attn_map, axis=-1), 1.0, atol=1e-3)


def test_action_chunking_shape_and_intent(vla):
    """Test full forward pass and action chunk generation."""
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Test emergency halt
    chunk_halt = vla.predict_action_chunk(dummy_img, "emergency halt immediately")
    assert chunk_halt.task_intent == "EMERGENCY_HALT"
    assert np.all(chunk_halt.waypoints == 0.0)
    assert np.all(chunk_halt.velocities == 0.0)

    # Test left bypass
    chunk_left = vla.predict_action_chunk(dummy_img, "bypass obstacle on left")
    assert chunk_left.task_intent == "EVASION_BYPASS_LEFT"
    assert chunk_left.waypoints[-1, 1] > 0.0  # Lateral shift positive (left)

    # Test precision docking
    chunk_dock = vla.predict_action_chunk(dummy_img, "dock at pallet bay 2")
    assert chunk_dock.task_intent == "PRECISION_DOCKING"
    assert chunk_dock.velocities[-1, 0] < chunk_dock.velocities[0, 0]  # Velocity ramps down


def test_vla_inference_latency(vla):
    """VLA forward pass must execute in sub-25ms."""
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    vla.predict_action_chunk(dummy_img, "navigate forward to loading dock")
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert dt_ms < 35.0

