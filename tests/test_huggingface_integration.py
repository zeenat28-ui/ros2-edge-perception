#!/usr/bin/env python3
"""
Unit tests for Hugging Face Model Integration & Fine-Tuning Pipeline.
"""

import os
import pytest
import numpy as np
from ros2_edge_perception.huggingface_model_manager import HuggingFaceModelManager
from ros2_edge_perception.finetune_vla_huggingface import HuggingFaceVLAFineTuner


@pytest.fixture
def hf_manager():
    return HuggingFaceModelManager(
        repo_id="salim4n/yolov8n-detect-onnx",
        filename="yolov8n-onnx-web/yolov8n.onnx",
        cache_dir="models/huggingface/yolov8n"
    )


def test_huggingface_model_loading(hf_manager):
    """Verify Hugging Face model file exists and OpenCV DNN network is loaded."""
    assert os.path.exists(hf_manager.model_path)
    assert hf_manager.net is not None


def test_huggingface_detection_inference(hf_manager):
    """Test object detection inference on synthetic input image."""
    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
    # Draw a simulated obstacle in center
    test_img[200:400, 200:400] = [200, 200, 200]

    detections = hf_manager.detect_objects(test_img)
    # Should run with zero crashes and return a list
    assert isinstance(detections, list)


def test_vla_finetuned_checkpoint_exists():
    """Verify fine-tuned LoRA checkpoint is saved and contains expected arrays."""
    ckpt_path = "models/huggingface/finetuned_vla_checkpoint.npz"
    assert os.path.exists(ckpt_path)
    data = np.load(ckpt_path)
    assert "lora_A" in data
    assert "lora_B" in data
    assert data["lora_A"].shape == (128, 8)
    assert data["lora_B"].shape == (8, 128)
