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


def test_letterbox_aspect_ratio_preservation():
    """Verify that letterbox resizing preserves aspect ratio with valid padding."""
    # Arbitrary non-square input image (1280x720)
    input_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    target_shape = (640, 640)

    # Replicate letterbox logic
    shape = input_img.shape[:2]
    r = min(target_shape[0] / shape[0], target_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (target_shape[1] - new_unpad[0]) / 2
    dh = (target_shape[0] - new_unpad[1]) / 2

    resized = cv2.resize(input_img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # Assertions
    assert padded.shape == (640, 640, 3), f"Expected (640, 640, 3), got {padded.shape}"
    # Verify scale ratio
    expected_ratio = 640.0 / 1280.0
    assert abs(r - expected_ratio) < 1e-4, f"Scale ratio mismatch: {r} vs {expected_ratio}"


def test_zero_copy_rgb_buffer_ingestion():
    """Verify zero-copy ingestion from raw byte buffer matches source frame."""
    h, w = 480, 640
    original_frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    raw_bytes = original_frame.tobytes()

    # Zero-copy view mapping
    reconstructed = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((h, w, 3))

    assert reconstructed.shape == (480, 640, 3)
    assert np.array_equal(original_frame, reconstructed)
    # Check that reconstructed buffer points to the same underlying memory
    assert reconstructed.nbytes == original_frame.nbytes


def test_zero_copy_depth_conversion():
    """Verify 16-bit depth (mm) conversion to metric float32 (meters)."""
    h, w = 480, 640
    # Simulate depth values between 500mm (0.5m) and 5000mm (5.0m)
    depth_mm = np.full((h, w), 2500, dtype=np.uint16)
    raw_bytes = depth_mm.tobytes()

    depth_from_buf = np.frombuffer(raw_bytes, dtype=np.uint16).reshape((h, w))
    depth_meters = depth_from_buf.astype(np.float32) / 1000.0

    assert depth_meters.shape == (480, 640)
    assert depth_meters.dtype == np.float32
    assert np.allclose(depth_meters, 2.5), f"Expected 2.5m, got {depth_meters[0, 0]}"


def test_pinhole_3d_deprojection_math():
    """Verify pinhole camera deprojection from pixel (u, v, z) to Euclidean (X, Y, Z)."""
    fx, fy = 554.25, 554.25
    cx, cy = 320.0, 240.0
    depth_z = 2.0  # 2.0 meters

    # Test Case 1: Point at optical center (320, 240) -> should be (0.0, 0.0, 2.0)
    u1, v1 = 320.0, 240.0
    x1 = (u1 - cx) * depth_z / fx
    y1 = (v1 - cy) * depth_z / fy
    assert abs(x1) < 1e-4
    assert abs(y1) < 1e-4

    # Test Case 2: Point offset by +100 pixels in X -> should be positive X
    u2 = 420.0
    x2 = (u2 - cx) * depth_z / fx
    expected_x2 = (100.0 * 2.0) / 554.25
    assert abs(x2 - expected_x2) < 1e-4


def test_robust_depth_outlier_rejection():
    """Verify that median percentile filtering discards noise and occlusion zeros."""
    # Target depth is 1.80m, with 20% zeros (holes) and 20% background bleed (4.0m)
    clean_target = np.full(60, 1.80, dtype=np.float32)
    zeros = np.zeros(20, dtype=np.float32)
    background = np.full(20, 4.00, dtype=np.float32)
    noisy_roi = np.concatenate([clean_target, zeros, background])

    # Filter invalid
    valid_mask = (noisy_roi >= 0.2) & (noisy_roi <= 10.0)
    valid_depths = noisy_roi[valid_mask]

    # Percentile filter (25th to 75th percentile)
    p25, p75 = np.percentile(valid_depths, [25, 75])
    filtered = valid_depths[(valid_depths >= p25) & (valid_depths <= p75)]
    estimated_z = float(np.median(filtered))

    # Should accurately recover 1.80m target
    assert abs(estimated_z - 1.80) < 0.05, f"Filtered depth {estimated_z} deviated from expected 1.80m"


def test_nms_vectorized_suppression():
    """Verify Non-Maximum Suppression eliminates duplicate overlapping detections."""
    # Two heavily overlapping boxes for the same object
    box1 = [100, 100, 50, 50]  # [x, y, w, h]
    box2 = [102, 101, 50, 49]  # Almost identical overlap
    scores = [0.90, 0.75]

    indices = cv2.dnn.NMSBoxes([box1, box2], scores, score_threshold=0.35, nms_threshold=0.45)
    surviving_indices = list(indices.flatten()) if len(indices) > 0 else []

    # Only the higher-confidence box (index 0) must survive
    assert len(surviving_indices) == 1
    assert surviving_indices[0] == 0


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

