#!/usr/bin/env python3
"""
Model Predictive Path Integral (MPPI) Trajectory Optimizer.

Generates optimal control inputs (v, omega) by importance sampling across parallel rollouts:
- Non-holonomic unicycle kinematics with centrifugal acceleration clamping
- Continuous bilinear interpolation on BEV obstacle costmaps
- Multi-horizon tracking of reference waypoints
- Numerically stable temperature-weighted Boltzmann importance sampling
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List


@dataclass
class MPPIOptimizationResult:
    """Result of MPPI parallel trajectory optimization."""
    best_trajectory: np.ndarray      # Shape (H+1, 3): [x, y, theta] best path
    optimal_controls: np.ndarray     # Shape (H, 2): [linear_v, angular_w]
    all_rollouts: np.ndarray         # Shape (K_sample, H+1, 2): sample paths for 3D visualization
    min_cost: float
    mean_cost: float
    effective_samples: float         # Sample diversity / entropy (ESS)
    latency_ms: float
    timestamp: float = field(default_factory=time.time)


class ParallelMPPIOptimizer:
    """
    Industrial-grade Model Predictive Path Integral (MPPI) Optimizer.
    Evaluates thousands of candidate paths simultaneously to synthesize smooth,
    obstacle-free, kinematically feasible control commands.
    """

    def __init__(
        self,
        num_samples: int = 10000,
        horizon: int = 20,
        dt: float = 0.05,            # 50 ms time step (1.0s total horizon)
        temperature: float = 0.08,   # Lambda temperature
        max_linear_v: float = 1.8,   # m/s
        max_angular_w: float = 1.5,  # rad/s
        max_accel_v: float = 2.5,    # m/s^2
        max_accel_w: float = 3.5,    # rad/s^2
        max_centripetal_accel: float = 1.8, # m/s^2 (prevents pallet tip-over)
        obstacle_weight: float = 95.0,
        goal_weight: float = 14.0,
        smoothness_weight: float = 4.0,
        vla_guidance_weight: float = 8.0,
        seed: int = 42
    ):
        self.num_samples = num_samples
        self.horizon = horizon
        self.dt = dt
        self.temperature = temperature
        self.max_linear_v = max_linear_v
        self.max_angular_w = max_angular_w
        self.max_accel_v = max_accel_v
        self.max_accel_w = max_accel_w
        self.max_centripetal_accel = max_centripetal_accel
        self.obstacle_weight = obstacle_weight
        self.goal_weight = goal_weight
        self.smoothness_weight = smoothness_weight
        self.vla_guidance_weight = vla_guidance_weight

        self.rng = np.random.RandomState(seed)

        # Nominal control sequence: shape (horizon, 2)
        self.u_nominal = np.zeros((horizon, 2), dtype=np.float32)
        self.u_nominal[:, 0] = 0.8  # Initial forward bias

        # Control noise covariance standard deviation [sigma_v, sigma_w]
        self.noise_sigma = np.array([0.45, 0.65], dtype=np.float32)

    def optimize(
        self,
        current_state: np.ndarray,            # [x, y, theta, v, w]
        target_waypoints: np.ndarray,         # (N_pts, 3): [x, y, theta] from VLA
        bev_costmap: np.ndarray,              # (Nx, Ny) from OccNet
        costmap_origin: Tuple[float, float] = (0.0, -4.0),
        costmap_resolution: float = 0.10,
        dynamic_obstacles: Optional[List[Dict]] = None
    ) -> MPPIOptimizationResult:
        """
        Executes parallel MPPI trajectory rollouts and synthesizes optimal controls.
        """
        t0 = time.perf_counter()
        K = self.num_samples
        H = self.horizon
        dt = self.dt

        # 1. Sample control perturbations: shape (K, H, 2)
        noise = self.rng.normal(0.0, 1.0, (K, H, 2)).astype(np.float32) * self.noise_sigma
        u_samples = self.u_nominal[None, :, :] + noise

        # Enforce kinodynamic velocity and centripetal acceleration limits
        u_samples[:, :, 0] = np.clip(u_samples[:, :, 0], -0.2, self.max_linear_v)
        u_samples[:, :, 1] = np.clip(u_samples[:, :, 1], -self.max_angular_w, self.max_angular_w)

        # Centripetal acceleration constraint: |v * omega| <= a_c_max
        v_abs = np.abs(u_samples[:, :, 0]) + 1e-6
        max_allowed_omega = self.max_centripetal_accel / v_abs
        u_samples[:, :, 1] = np.clip(u_samples[:, :, 1], -max_allowed_omega, max_allowed_omega)

        # 2. Vectorized Forward Kinematic Rollout: shape (K, H+1, 3)
        states = np.zeros((K, H + 1, 3), dtype=np.float32)
        states[:, 0, 0] = current_state[0]
        states[:, 0, 1] = current_state[1]
        states[:, 0, 2] = current_state[2]

        for t in range(H):
            v = u_samples[:, t, 0]
            w = u_samples[:, t, 1]
            theta = states[:, t, 2]

            # Non-holonomic unicycle motion integration
            states[:, t + 1, 0] = states[:, t, 0] + v * np.cos(theta) * dt
            states[:, t + 1, 1] = states[:, t, 1] + v * np.sin(theta) * dt
            states[:, t + 1, 2] = states[:, t, 2] + w * dt

        # 3. Vectorized Dynamic Cost Manifold Evaluation
        costs = np.zeros(K, dtype=np.float32)

        # Cost Component A: Terminal Goal Cost
        if len(target_waypoints) > 0:
            target_pt = target_waypoints[-1, :2]
            final_pos = states[:, -1, :2]
            goal_dists = np.linalg.norm(final_pos - target_pt[None, :], axis=1)
            costs += self.goal_weight * goal_dists

        # Cost Component B: Soft VLA Action Chunk Guidance
        # Encourages paths to align with all intermediate VLA waypoints
        if len(target_waypoints) >= 2:
            num_vla_pts = min(H, len(target_waypoints))
            vla_pts = target_waypoints[:num_vla_pts, :2]
            rollout_pts = states[:, 1:num_vla_pts + 1, :2]
            vla_dist = np.mean(np.linalg.norm(rollout_pts - vla_pts[None, :, :], axis=2), axis=1)
            costs += self.vla_guidance_weight * vla_dist

        # Cost Component C: Continuous Bilinear Costmap Sampling
        nx, ny = bev_costmap.shape
        ox, oy = costmap_origin
        res = costmap_resolution

        traj_xs = states[:, 1:, 0]  # (K, H)
        traj_ys = states[:, 1:, 1]  # (K, H)

        # Convert to continuous grid coordinates
        gx = (traj_xs - ox) / res
        gy = (traj_ys - oy) / res

        # Bilinear interpolation coordinates
        gx0 = np.clip(np.floor(gx).astype(np.int32), 0, nx - 2)
        gy0 = np.clip(np.floor(gy).astype(np.int32), 0, ny - 2)
        gx1 = gx0 + 1
        gy1 = gy0 + 1

        wx = np.clip(gx - gx0, 0.0, 1.0)
        wy = np.clip(gy - gy0, 0.0, 1.0)

        # 4-point bilinear sample
        c00 = bev_costmap[gx0, gy0]
        c10 = bev_costmap[gx1, gy0]
        c01 = bev_costmap[gx0, gy1]
        c11 = bev_costmap[gx1, gy1]

        cost_interp = (
            (1.0 - wx) * (1.0 - wy) * c00 +
            wx * (1.0 - wy) * c10 +
            (1.0 - wx) * wy * c01 +
            wx * wy * c11
        )
        obs_costs = np.sum(cost_interp, axis=1)
        costs += self.obstacle_weight * obs_costs

        # Cost Component D: Dynamic Obstacle Velocity Cone Avoidance
        if dynamic_obstacles:
            for obs in dynamic_obstacles:
                ox_pos, oy_pos = obs.get("centroid_3d", [0, 0, 0])[:2]
                vx_obs, vy_obs = obs.get("velocity", [0, 0])[:2]
                # Forecast obstacle position across horizon
                for t in range(H):
                    obs_t_x = ox_pos + vx_obs * (t + 1) * dt
                    obs_t_y = oy_pos + vy_obs * (t + 1) * dt
                    d_dyn = np.linalg.norm(states[:, t + 1, :2] - np.array([obs_t_x, obs_t_y])[None, :], axis=1)
                    # Severe exponential cost for approaching dynamic obstacle
                    costs += 45.0 * np.exp(-3.0 * np.maximum(0.0, d_dyn - 0.5))

        # Cost Component E: Kinodynamic Smoothness and Acceleration Penalties
        diff_u = np.diff(u_samples, axis=1)  # (K, H-1, 2)
        smoothness = np.sum(diff_u ** 2, axis=(1, 2))
        costs += self.smoothness_weight * smoothness

        # 4. Softmax Importance Weighting & Control Synthesis
        min_cost = float(np.min(costs))
        mean_cost = float(np.mean(costs))

        # Numerically stable temperature-scaled softmax
        beta = min_cost
        exp_weights = np.exp(-(costs - beta) / self.temperature)
        sum_weights = np.sum(exp_weights) + 1e-8
        weights = exp_weights / sum_weights  # Shape (K,)

        # Synthesize optimal control sequence
        self.u_nominal = np.sum(weights[:, None, None] * u_samples, axis=0)

        # Best trajectory from lowest-cost rollout
        best_idx = int(np.argmin(costs))
        best_traj = states[best_idx, :, :]

        # Subsample rollouts for 3D digital twin visualization (e.g. 150 paths)
        sample_indices = np.linspace(0, K - 1, min(150, K)).astype(np.int32)
        vis_rollouts = states[sample_indices, :, :2]

        dt_ms = (time.perf_counter() - t0) * 1000.0

        # Effective Sample Size (sample diversity metric)
        ess = float(1.0 / (np.sum(weights ** 2) + 1e-8))

        return MPPIOptimizationResult(
            best_trajectory=best_traj,
            optimal_controls=self.u_nominal,
            all_rollouts=vis_rollouts,
            min_cost=round(min_cost, 3),
            mean_cost=round(mean_cost, 3),
            effective_samples=round(ess, 1),
            latency_ms=round(dt_ms, 2),
            timestamp=time.time()
        )
