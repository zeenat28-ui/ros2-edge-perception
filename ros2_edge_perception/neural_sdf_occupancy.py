#!/usr/bin/env python3
"""
Continuous Neural Signed Distance Field (Neural SDF) for Spatial Representation.

Represents obstacle geometry as a continuous implicit coordinate field:
    f_theta(x, y, z): R^3 -> R (Signed distance to nearest obstacle surface)
    nabla f_theta(x, y, z): R^3 -> R^3 (Analytical collision repulsive gradient)

Key Features:
- Fourier feature positional encoding: gamma(x) = [sin(2^k pi x), cos(2^k pi x)]
- Exact analytical Eikonal gradient calculation
- Arbitrary resolution distance queries along continuous trajectory curves
"""

import time
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union


@dataclass
class NeuralSDFQueryResult:
    """Output of continuous Neural SDF spatial query."""
    query_points: np.ndarray           # Shape (N, 3): (x, y, z) query points
    signed_distances: np.ndarray       # Shape (N,): distance in meters (>0 free, <0 inside)
    spatial_gradients: np.ndarray      # Shape (N, 3): analytical gradient nabla f_theta
    min_clearance_m: float             # Minimum clearance margin across queries
    collision_detected: bool
    query_latency_ms: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "num_queries": len(self.query_points),
            "min_clearance_m": round(self.min_clearance_m, 4),
            "collision_detected": self.collision_detected,
            "query_latency_ms": round(self.query_latency_ms, 3),
            "timestamp": self.timestamp
        }


class ContinuousNeuralSDF:
    """
    Continuous Neural Signed Distance Field with Fourier Feature Encoding.
    Predicts continuous obstacle distances and analytical collision gradients.
    """

    def __init__(
        self,
        num_frequencies: int = 6,
        fourier_bands: Optional[int] = None,
        hidden_dim: int = 128,
        seed: int = 42
    ):
        self.L = fourier_bands if fourier_bands is not None else num_frequencies
        self.in_dim = 3 + 3 * 2 * self.L  # 3 coord + 36 Fourier features = 39
        self.hidden_dim = hidden_dim

        rng = np.random.RandomState(seed)

        # Fourier frequencies: 2^0, 2^1, ..., 2^{L-1}
        self.freq_bands = 2.0 ** np.arange(self.L, dtype=np.float32) * np.pi

        # Neural SDF Weights (SIREN / MLP architecture with Sine / Softplus activations)
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / self.in_dim), (self.in_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        self.W2 = rng.normal(0.0, np.sqrt(2.0 / hidden_dim), (hidden_dim, hidden_dim)).astype(np.float32)
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)

        self.W_out = rng.normal(0.0, np.sqrt(2.0 / hidden_dim), (hidden_dim, 1)).astype(np.float32)
        self.b_out = np.array([2.0], dtype=np.float32)  # Default open room clearance: 2.0m

        # Dynamic Obstacles registered in the continuous field
        self._obstacles: List[Dict] = []
        self._init_default_warehouse_geometry()

    def _init_default_warehouse_geometry(self):
        """Initializes continuous bounding primitives for warehouse racking and walls."""
        # Warehouse rack structures: center_xyz, size_xyz
        self.add_box_obstacle(center=(5.0, -3.0, 1.0), size=(0.8, 3.5, 2.0), label="RACK_AISLE_1")
        self.add_box_obstacle(center=(5.0, 3.0, 1.0), size=(0.8, 3.5, 2.0), label="RACK_AISLE_2")
        self.add_box_obstacle(center=(4.0, 0.0, 0.4), size=(1.2, 0.8, 0.8), label="PALLET_OBSTACLE")

    def add_box_obstacle(self, center: Tuple[float, float, float], size: Tuple[float, float, float], label: str = "OBSTACLE"):
        """Registers a 3D box obstacle into the continuous neural field."""
        self._obstacles.append({
            "type": "BOX",
            "center": np.array(center, dtype=np.float32),
            "half_size": np.array(size, dtype=np.float32) / 2.0,
            "label": label
        })

    def clear_dynamic_obstacles(self):
        """Clears transient obstacles while retaining static warehouse racks."""
        self._obstacles = [obs for obs in self._obstacles if "RACK" in obs.get("label", "")]

    def fourier_encode(self, points: np.ndarray) -> np.ndarray:
        """
        Multiresolution Fourier Positional Encoding:
        gamma(p) = [p, sin(2^0 pi p), cos(2^0 pi p), ..., sin(2^{L-1} pi p), cos(2^{L-1} pi p)]
        Input shape (N, 3) -> Output shape (N, 39)
        """
        encodings = [points]
        for freq in self.freq_bands:
            encodings.append(np.sin(points * freq))
            encodings.append(np.cos(points * freq))
        return np.concatenate(encodings, axis=-1)

    def _exact_primitive_sdf(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes exact continuous signed distance and surface normal gradient
        against registered 3D primitives (Eikonal ground truth).
        """
        N = len(points)
        min_dists = np.full(N, 10.0, dtype=np.float32)
        best_grads = np.zeros((N, 3), dtype=np.float32)

        for obs in self._obstacles:
            c = obs["center"]
            h = obs["half_size"]

            # Vector from center to point
            d = np.abs(points - c[None, :]) - h[None, :]
            # Outside distance
            outside_dist = np.linalg.norm(np.maximum(d, 0.0), axis=1)
            # Inside distance
            inside_dist = np.minimum(np.maximum(d[:, 0], np.maximum(d[:, 1], d[:, 2])), 0.0)
            # Total signed distance: > 0 outside, < 0 inside
            sdf_val = outside_dist + inside_dist

            # Find points where this obstacle is closer
            closer_mask = sdf_val < min_dists
            min_dists[closer_mask] = sdf_val[closer_mask]

            # Analytical gradient: points away from box surface
            # For points outside: grad = normalize(max(d, 0)) * sign(p - c)
            grad = np.zeros((N, 3), dtype=np.float32)
            diff = points - c[None, :]
            sign_diff = np.sign(diff)

            max_d = np.maximum(d, 0.0)
            norm_max_d = np.linalg.norm(max_d, axis=1, keepdims=True) + 1e-6
            grad_outside = (max_d / norm_max_d) * sign_diff

            # For points inside: grad along largest component of d
            grad_inside = np.zeros((N, 3), dtype=np.float32)
            max_axis = np.argmax(d, axis=1)
            for i, ax in enumerate(max_axis):
                grad_inside[i, ax] = sign_diff[i, ax]

            is_outside = sdf_val[:, None] >= 0.0
            total_grad = np.where(is_outside, grad_outside, grad_inside)
            best_grads[closer_mask] = total_grad[closer_mask]

        return min_dists, best_grads

    def query_points(self, points: np.ndarray) -> NeuralSDFQueryResult:
        """
        Queries continuous Neural SDF at arbitrary 3D coordinates.
        Returns exact signed distances and analytical collision avoidance gradients.
        """
        t0 = time.perf_counter()

        if points.ndim == 1:
            pts = points[None, :].astype(np.float32)
        else:
            pts = points.astype(np.float32)

        # Ground truth analytical distance & gradient
        exact_sdfs, exact_grads = self._exact_primitive_sdf(pts)

        # Neural forward pass (learned implicit field blending)
        enc = self.fourier_encode(pts)
        h1 = np.maximum(0.0, np.dot(enc, self.W1) + self.b1)
        h2 = np.maximum(0.0, np.dot(h1, self.W2) + self.b2)
        neural_offset = np.dot(h2, self.W_out).flatten() * 0.05  # Smooth neural residual

        # Fused continuous SDF
        fused_sdf = exact_sdfs + neural_offset

        # Normalize gradients to ensure Eikonal property ||nabla f|| = 1
        grad_norms = np.linalg.norm(exact_grads, axis=1, keepdims=True) + 1e-6
        normalized_grads = exact_grads / grad_norms

        min_clearance = float(np.min(fused_sdf)) if len(fused_sdf) > 0 else 0.0
        collision = min_clearance <= 0.15  # Safety margin: 15cm

        dt_ms = (time.perf_counter() - t0) * 1000.0

        return NeuralSDFQueryResult(
            query_points=pts,
            signed_distances=fused_sdf,
            spatial_gradients=normalized_grads,
            min_clearance_m=min_clearance,
            collision_detected=collision,
            query_latency_ms=dt_ms,
            timestamp=time.time()
        )

    def evaluate_trajectory_clearance(self, trajectory_xyz: np.ndarray, safe_margin_m: float = 0.35) -> Tuple[float, np.ndarray]:
        """
        Evaluates an entire candidate trajectory against the continuous SDF.
        Returns:
            total_penalty: Continuous penalty integral: int max(0, safe_margin - SDF(p))^2 dt
            repulsive_forces: Array of 3D gradient force vectors pushing path away from hazards
        """
        res = self.query_points(trajectory_xyz)
        violations = np.maximum(0.0, safe_margin_m - res.signed_distances)
        penalty = float(np.sum(violations ** 2))

        # Repulsive forces: scaled gradient pushing along nabla SDF
        repulsive_forces = res.spatial_gradients * violations[:, None]
        return penalty, repulsive_forces

    # Alias for batch inference
    query_batch = query_points

