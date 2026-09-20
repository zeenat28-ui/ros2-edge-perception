"""
Kinodynamic State-Space Model for Differential-Drive Autonomous Mobile Robots.

Implements non-holonomic vehicle dynamics with actuator and traction constraints:
- Continuous differential drive kinematics: dx/dt = v*cos(th), dy/dt = v*sin(th), dth/dt = w
- 4th-Order Runge-Kutta (RK4) numerical state integration
- Centrifugal acceleration limits: |v * omega| <= a_lat_max (anti-rollover protection)
- Payload mass and rotational inertia scaling: I_z = I_base + m_payload * r^2
- Wheel traction and slip modeling for polished industrial warehouse floors
"""

import math
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class AMRKinodynamicState:
    """State vector of the AMR in SE(2) with 1st- and 2nd-order derivatives."""
    x: float = 0.0              # Global X position [m]
    y: float = 0.0              # Global Y position [m]
    theta: float = 0.0          # Heading orientation [rad]
    v: float = 0.0              # Forward longitudinal velocity [m/s]
    omega: float = 0.0          # Angular yaw rate [rad/s]
    a_lin: float = 0.0          # Instantaneous linear acceleration [m/s^2]
    a_ang: float = 0.0          # Instantaneous angular acceleration [rad/s^2]
    wheel_slip_left: float = 0.0   # Left wheel slip ratio in [0, 1]
    wheel_slip_right: float = 0.0  # Right wheel slip ratio in [0, 1]


class KinodynamicAMRModel:
    """
    Continuous-time kinodynamic model with numerical integration
    and physical actuator constraints.
    """

    def __init__(
        self,
        base_mass_kg: float = 85.0,
        wheelbase_m: float = 0.54,
        wheel_radius_m: float = 0.10,
        max_linear_vel_mps: float = 1.50,
        max_angular_vel_radps: float = 1.80,
        max_linear_accel_mps2: float = 1.20,
        max_angular_accel_radps2: float = 2.40,
        max_centrifugal_accel_mps2: float = 1.20,
        max_motor_torque_nm: float = 45.0,
        friction_coeff: float = 0.65
    ):
        self.m_base = base_mass_kg
        self.L = wheelbase_m
        self.r = wheel_radius_m
        self.v_max = max_linear_vel_mps
        self.omega_max = max_angular_vel_radps
        self.a_lin_max = max_linear_accel_mps2
        self.a_ang_max = max_angular_accel_radps2
        self.a_lat_max = max_centrifugal_accel_mps2
        self.T_max = max_motor_torque_nm
        self.mu = friction_coeff

        # Base rotational inertia about vertical Z-axis: I_z ~ 0.5 * m * (L/2)^2
        self.I_base = 0.5 * self.m_base * ((self.L / 2.0) ** 2)

        # Dynamic payload
        self.m_payload = 0.0
        self.I_total = self.I_base
        self.m_total = self.m_base

    def set_payload_mass(self, payload_kg: float):
        """Adapts mass and rotational inertia for carried cargo."""
        self.m_payload = max(0.0, float(payload_kg))
        self.m_total = self.m_base + self.m_payload
        # Payload modeled as distributed cylinder on robot deck (radius ~ L/2)
        I_payload = 0.5 * self.m_payload * ((self.L / 2.0) ** 2)
        self.I_total = self.I_base + I_payload

    def apply_kinodynamic_limits(
        self,
        target_v: float,
        target_omega: float,
        current_v: float,
        current_omega: float,
        dt: float
    ) -> Tuple[float, float, float, float]:
        """
        Clamps commanded velocities according to physical acceleration,
        motor torque, and centrifugal tipping limits.
        """
        # 1. Centrifugal Acceleration Clamp: |v * omega| <= a_lat_max
        if abs(target_v * target_omega) > self.a_lat_max:
            # Scale down angular velocity to preserve forward progress safely
            safe_omega = math.copysign(self.a_lat_max / max(abs(target_v), 1e-4), target_omega)
            target_omega = safe_omega

        # 2. Maximum Velocity Limits
        target_v = float(np.clip(target_v, -0.40, self.v_max))  # Reverse limited to 0.4 m/s
        target_omega = float(np.clip(target_omega, -self.omega_max, self.omega_max))

        # 3. Payload-Adaptive Linear Acceleration Clamp
        # F_brake = mu * m_total * g
        max_achievable_accel = min(self.a_lin_max, (self.mu * 9.81 * self.m_base) / self.m_total)
        desired_a_lin = (target_v - current_v) / max(dt, 1e-4)
        clamped_a_lin = float(np.clip(desired_a_lin, -max_achievable_accel, max_achievable_accel))
        achieved_v = current_v + clamped_a_lin * dt

        # 4. Payload-Adaptive Angular Acceleration Clamp: tau = I * alpha <= T_max * 2 / L
        max_alpha = min(self.a_ang_max, (self.T_max * self.L) / max(self.I_total, 1e-3))
        desired_a_ang = (target_omega - current_omega) / max(dt, 1e-4)
        clamped_a_ang = float(np.clip(desired_a_ang, -max_alpha, max_alpha))
        achieved_omega = current_omega + clamped_a_ang * dt

        # 5. Strict Centrifugal Tipping Protection
        if abs(achieved_v * achieved_omega) > self.a_lat_max:
            achieved_omega = float(math.copysign(self.a_lat_max / max(abs(achieved_v), 1e-4), achieved_omega))

        return achieved_v, achieved_omega, clamped_a_lin, clamped_a_ang

    def step(
        self,
        state: AMRKinodynamicState,
        cmd_v: float,
        cmd_omega: float,
        dt: float = 0.05
    ) -> AMRKinodynamicState:
        """
        Integrates non-holonomic state equations forward by dt:
        Using 4th-order Runge-Kutta (RK4) integration for high numerical fidelity.
        """
        achieved_v, achieved_omega, a_lin, a_ang = self.apply_kinodynamic_limits(
            cmd_v, cmd_omega, state.v, state.omega, dt
        )

        # Differential drive wheel speeds: v_L = v - (omega * L / 2), v_R = v + (omega * L / 2)
        v_l = achieved_v - (achieved_omega * self.L / 2.0)
        v_r = achieved_v + (achieved_omega * self.L / 2.0)

        # Slip estimation
        slip_l = float(np.clip(abs(a_lin) / (max_achievable_accel := self.mu * 9.81), 0.0, 1.0) * 0.05)
        slip_r = slip_l

        # Non-holonomic RK4 integration for (x, y, theta)
        def derivatives(x_t, y_t, th_t, v_t, w_t):
            return np.array([v_t * math.cos(th_t), v_t * math.sin(th_t), w_t])

        s0 = np.array([state.x, state.y, state.theta])
        k1 = derivatives(s0[0], s0[1], s0[2], state.v, state.omega)
        k2 = derivatives(s0[0] + 0.5 * dt * k1[0], s0[1] + 0.5 * dt * k1[1], s0[2] + 0.5 * dt * k1[2], 0.5 * (state.v + achieved_v), 0.5 * (state.omega + achieved_omega))
        k3 = derivatives(s0[0] + 0.5 * dt * k2[0], s0[1] + 0.5 * dt * k2[1], s0[2] + 0.5 * dt * k2[2], 0.5 * (state.v + achieved_v), 0.5 * (state.omega + achieved_omega))
        k4 = derivatives(s0[0] + dt * k3[0], s0[1] + dt * k3[1], s0[2] + dt * k3[2], achieved_v, achieved_omega)

        s_next = s0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        # Normalize theta to [-pi, pi]
        next_th = math.atan2(math.sin(s_next[2]), math.cos(s_next[2]))

        return AMRKinodynamicState(
            x=float(s_next[0]),
            y=float(s_next[1]),
            theta=next_th,
            v=achieved_v,
            omega=achieved_omega,
            a_lin=a_lin,
            a_ang=a_ang,
            wheel_slip_left=slip_l,
            wheel_slip_right=slip_r
        )
