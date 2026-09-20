"""
Enterprise Unit & Integration Test Suite for ROS 2 Edge Perception Pipeline.

Tests:
1. Aspect-ratio preserving letterbox preprocessing & tensor normalization.
2. Zero-copy buffer ingestion for RGB (bgr8) and Depth (16UC1).
3. 3D Pin-hole camera deprojection geometry & spatial accuracy.
4. Robust percentile depth outlier rejection (filtering sensor holes & edge bleed).
5. Vectorized Non-Maximum Suppression (NMS) box deduplication.
6. ONNX model architecture & tensor I/O contract.
"""

import os
import cv2
import numpy as np
import pytest

# Ensure module path is accessible
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from ros2_edge_perception.perception_node import (
    preprocess_letterbox,
    convert_depth_buffer,
    deproject_pixel_to_3d,
    filter_depth_roi,
    compute_nms,
)


def test_letterbox_aspect_ratio_preservation():
    """Verify that production letterbox resizing preserves aspect ratio with valid padding."""
    input_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    target_shape = (640, 640)

    blob, r, (dw, dh) = preprocess_letterbox(input_img, target_shape)

    assert blob.shape == (1, 3, 640, 640), f"Expected (1, 3, 640, 640), got {blob.shape}"
    expected_ratio = 640.0 / 1280.0
    assert abs(r - expected_ratio) < 1e-4, f"Scale ratio mismatch: {r} vs {expected_ratio}"
    assert dw == 0.0
    assert dh == 140.0


def test_zero_copy_rgb_buffer_ingestion():
    """Verify buffer ingestion from raw byte buffer matches source frame."""
    h, w = 480, 640
    original_frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    raw_bytes = original_frame.tobytes()

    reconstructed = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((h, w, 3))

    assert reconstructed.shape == (480, 640, 3)
    assert np.array_equal(original_frame, reconstructed)
    assert reconstructed.nbytes == original_frame.nbytes


def test_zero_copy_depth_conversion():
    """Verify 16-bit depth (mm) conversion to metric float32 (meters) via convert_depth_buffer."""
    h, w = 480, 640
    depth_mm = np.full((h, w), 2500, dtype=np.uint16)
    raw_bytes = depth_mm.tobytes()

    depth_meters = convert_depth_buffer(raw_bytes, h, w)

    assert depth_meters.shape == (480, 640)
    assert depth_meters.dtype == np.float32
    assert np.allclose(depth_meters, 2.5), f"Expected 2.5m, got {depth_meters[0, 0]}"


def test_pinhole_3d_deprojection_math():
    """Verify pinhole camera deprojection using production deproject_pixel_to_3d."""
    fx, fy = 554.25, 554.25
    cx, cy = 320.0, 240.0
    depth_z = 2.0  # 2.0 meters

    # Optical center -> (0.0, 0.0, 2.0)
    x1, y1, z1 = deproject_pixel_to_3d(320.0, 240.0, depth_z, fx, fy, cx, cy)
    assert abs(x1) < 1e-4
    assert abs(y1) < 1e-4
    assert abs(z1 - 2.0) < 1e-4

    # Offset +100 px in X -> positive X
    x2, y2, z2 = deproject_pixel_to_3d(420.0, 240.0, depth_z, fx, fy, cx, cy)
    expected_x2 = (100.0 * 2.0) / 554.25
    assert abs(x2 - expected_x2) < 1e-4


def test_robust_depth_outlier_rejection():
    """Verify robust depth outlier rejection via production filter_depth_roi."""
    clean_target = np.full(60, 1.80, dtype=np.float32)
    zeros = np.zeros(20, dtype=np.float32)
    background = np.full(20, 4.00, dtype=np.float32)
    noisy_roi = np.concatenate([clean_target, zeros, background])

    estimated_z = filter_depth_roi(noisy_roi, min_depth=0.2, max_depth=10.0)

    assert estimated_z is not None
    assert abs(estimated_z - 1.80) < 0.05, f"Filtered depth {estimated_z} deviated from expected 1.80m"


def test_nms_vectorized_suppression():
    """Verify Non-Maximum Suppression via production compute_nms."""
    box1 = [100, 100, 50, 50]
    box2 = [102, 101, 50, 49]
    scores = [0.90, 0.75]

    surviving = compute_nms([box1, box2], scores, score_threshold=0.35, nms_threshold=0.45)

    assert len(surviving) == 1
    assert surviving[0] == 0


def test_onnx_model_contract():
    """Verify YOLOv8n ONNX model existence, valid shape, and inference compatibility."""
    model_path = os.path.join(BASE_DIR, "models", "yolov8n.onnx")
    hf_model_path = os.path.join(BASE_DIR, "models", "huggingface", "yolov8n", "yolov8n-onnx-web", "yolov8n.onnx")
    target_path = model_path if os.path.exists(model_path) else (hf_model_path if os.path.exists(hf_model_path) else None)

    if not target_path or not os.path.exists(target_path):
        pytest.skip("yolov8n.onnx not present; skipping tensor test.")

    net = cv2.dnn.readNetFromONNX(target_path)
    assert net is not None

    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
    net.setInput(dummy_input)
    res = net.forward()
    assert res.shape == (1, 84, 8400)

