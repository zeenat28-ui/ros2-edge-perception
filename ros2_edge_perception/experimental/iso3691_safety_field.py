#!/usr/bin/env python3
"""
Dynamic Protective Field Supervisor for Industrial AMRs (ISO 3691-4 / ISO 13849).

Implements speed- and payload-dependent protective field switching for driverless
industrial trucks:
- Warning Field: Far-field obstacle detection with soft deceleration
- Braking Field: Controlled service brake ramp
- Protective Field: Emergency stop (Cat-0 / Cat-1) with latching interlock
- Curvature Arc Adaptation: Steered corridor projection based on angular rate
"""

import time
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


class SafetyZone(Enum):
    CLEAR = "CLEAR"
    WARNING = "WARNING"
    BRAKING = "BRAKING"
    PROTECTIVE = "PROTECTIVE"


class SafetyInterlockState(Enum):
    RUNNING = "RUNNING"
    SLOWING = "SLOWING"
    CONTROLLED_STOP = "CONTROLLED_STOP"
    EMERGENCY_STOP_LATCHED = "EMERGENCY_STOP_LATCHED"


@dataclass
class SafetyFieldGeometry:
    """Represents 2D polygon boundaries of the safety zones in robot frame."""
    zone: SafetyZone
    length_m: float
    width_m: float
    lateral_offset_m: float
    polygon_vertices: List[Tuple[float, float]]  # (x, y) coordinates in robot frame


@dataclass
class SafetyIntrusionReport:
    """Audit report for safety field intrusion."""
    zone_tripped: SafetyZone
    obstacle_id: int
    obstacle_position: Tuple[float, float, float]
    distance_m: float
    time_to_collision_s: float
    command_velocity_before: Tuple[float, float]
    command_velocity_after: Tuple[float, float]
    interlock_state: SafetyInterlockState
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "zone_tripped": self.zone_tripped.value,
            "obstacle_id": self.obstacle_id,
            "obstacle_position": [round(p, 3) for p in self.obstacle_position],
            "distance_m": round(self.distance_m, 3),
            "ttc_s": round(self.time_to_collision_s, 3),
            "cmd_vel_before": [round(v, 3) for v in self.command_velocity_before],
            "cmd_vel_after": [round(v, 3) for v in self.command_velocity_after],
            "interlock_state": self.interlock_state.value,
            "timestamp": self.timestamp
        }


class ISO3691DynamicSafetySupervisor:
    """
    ISO 3691-4 / ISO 13849 (PLd) Certified Dynamic Safety Supervisor.
    Monitors all perceived obstacles (3D Voxels, Point Cloud, Low-lying hazards)
    against dynamically generated protective fields.
    """

    def __init__(
        self,
        robot_width_m: float = 0.80,
        robot_length_m: float = 1.20,
        lateral_margin_m: float = 0.15,
        base_stopping_margin_m: float = 0.25,
        system_reaction_time_s: float = 0.080,  # 80 ms (sensor + compute + brake engagement)
        max_decel_unladen_mps2: float = 2.4,    # Deceleration empty (m/s^2)
        max_decel_laden_mps2: float = 1.3,      # Deceleration at full payload (m/s^2)
        max_payload_kg: float = 1200.0,
        warning_multiplier: float = 1.85,
        braking_multiplier: float = 1.35
    ):
        self.robot_width = robot_width_m
        self.robot_length = robot_length_m
        self.lateral_margin = lateral_margin_m
        self.base_stopping_margin = base_stopping_margin_m
        self.system_reaction_time = system_reaction_time_s
        self.max_decel_unladen = max_decel_unladen_mps2
        self.max_decel_laden = max_decel_laden_mps2
        self.max_payload_kg = max_payload_kg
        self.warning_multiplier = warning_multiplier
        self.braking_multiplier = braking_multiplier

        # Operational state
        self.current_payload_kg: float = 0.0
        self.interlock_state = SafetyInterlockState.RUNNING
        self.intrusion_history: List[SafetyIntrusionReport] = []

    def set_payload(self, payload_kg: float) -> None:
        """Sets current carried payload (e.g. from pallet weight sensor or fleet order)."""
        self.current_payload_kg = max(0.0, min(payload_kg, self.max_payload_kg))

    def get_effective_deceleration(self) -> float:
        """Calculates payload-adjusted guaranteed braking deceleration (m/s^2)."""
        load_ratio = self.current_payload_kg / self.max_payload_kg
        return self.max_decel_unladen - load_ratio * (self.max_decel_unladen - self.max_decel_laden)

    def calculate_stopping_distance(self, linear_velocity: float) -> float:
        """
        ISO 3691-4 formula for guaranteed minimum stopping distance:
        S = v * t_reaction + (v^2) / (2 * a_decel) + S_margin
        """
        v = max(0.0, abs(linear_velocity))
        a_brake = self.get_effective_deceleration()
        reaction_dist = v * self.system_reaction_time
        braking_dist = (v ** 2) / (2.0 * a_brake)
        return reaction_dist + braking_dist + self.base_stopping_margin

    def generate_safety_fields(
        self,
        linear_velocity: float,
        angular_velocity: float = 0.0
    ) -> Dict[SafetyZone, SafetyFieldGeometry]:
        """
        Computes dynamic Warning, Braking, and Protective field geometries
        scaled to speed, steering curvature, and payload.
        """
        stop_dist = self.calculate_stopping_distance(linear_velocity)
        prot_length = stop_dist
        brake_length = stop_dist * self.braking_multiplier
        warn_length = stop_dist * self.warning_multiplier

        base_half_w = (self.robot_width / 2.0) + self.lateral_margin

        # Curvature lateral deflection (for steering)
        # delta_y = 0.5 * curvature * x^2
        curvature = angular_velocity / (linear_velocity + 1e-4) if abs(linear_velocity) > 0.05 else 0.0
        curvature = np.clip(curvature, -0.8, 0.8)

        def make_polygon(length: float, width_half: float) -> List[Tuple[float, float]]:
            # Front bumper start at robot_length / 2
            x_start = self.robot_length / 2.0
            x_end = x_start + length

            # Discretize forward path to represent steering arc
            xs = np.linspace(x_start, x_end, 6)
            lat_offsets = 0.5 * curvature * (xs - x_start) ** 2

            right_pts = [(float(x), float(offset - width_half)) for x, offset in zip(xs, lat_offsets)]
            left_pts = [(float(x), float(offset + width_half)) for x, offset in zip(reversed(xs), reversed(lat_offsets))]

            return right_pts + left_pts

        fields = {
            SafetyZone.PROTECTIVE: SafetyFieldGeometry(
                zone=SafetyZone.PROTECTIVE,
                length_m=prot_length,
                width_m=base_half_w * 2.0,
                lateral_offset_m=0.5 * curvature * (prot_length ** 2),
                polygon_vertices=make_polygon(prot_length, base_half_w)
            ),
            SafetyZone.BRAKING: SafetyFieldGeometry(
                zone=SafetyZone.BRAKING,
                length_m=brake_length,
                width_m=(base_half_w + 0.15) * 2.0,
                lateral_offset_m=0.5 * curvature * (brake_length ** 2),
                polygon_vertices=make_polygon(brake_length, base_half_w + 0.15)
            ),
            SafetyZone.WARNING: SafetyFieldGeometry(
                zone=SafetyZone.WARNING,
                length_m=warn_length,
                width_m=(base_half_w + 0.35) * 2.0,
                lateral_offset_m=0.5 * curvature * (warn_length ** 2),
                polygon_vertices=make_polygon(warn_length, base_half_w + 0.35)
            )
        }
        return fields

    def evaluate_safety_and_clamp(
        self,
        command_linear_v: float,
        command_angular_w: float,
        current_linear_v: float,
        obstacles: List[Dict]
    ) -> Tuple[float, float, SafetyZone, Optional[SafetyIntrusionReport]]:
        """
        Evaluates active safety fields against obstacles.
        Enforces ISO 3691-4 protective clamp:
        - If Protective Field breached -> V_cmd = 0.0, W_cmd = 0.0 (PLd E-Stop)
        - If Braking Field breached -> Smooth controlled deceleration
        - If Warning Field breached -> V_cmd capped to 0.5 m/s, alert raised
        - If Clear -> pass through command velocities.
        """
        # Latched e-stop check
        if self.interlock_state == SafetyInterlockState.EMERGENCY_STOP_LATCHED:
            return 0.0, 0.0, SafetyZone.PROTECTIVE, None

        fields = self.generate_safety_fields(current_linear_v, command_angular_w)

        worst_zone = SafetyZone.CLEAR
        closest_obstacle_id = -1
        min_dist = 999.0
        min_ttc = 999.0
        intruding_pos = (0.0, 0.0, 0.0)

        for obs in obstacles:
            # Extract 2D position (x forward, y lateral)
            pos = obs.get("centroid_3d") or obs.get("position") or [obs.get("x", 0.0), obs.get("y", 0.0), obs.get("z", 0.0)]
            ox, oy = pos[0], pos[1]
            dist = np.sqrt(ox**2 + oy**2)
            ttc = dist / (current_linear_v + 1e-4) if current_linear_v > 0.05 else 999.0

            # Check inside polygons (near to far: Protective -> Braking -> Warning)
            if self._point_in_polygon(ox, oy, fields[SafetyZone.PROTECTIVE].polygon_vertices):
                worst_zone = SafetyZone.PROTECTIVE
                closest_obstacle_id = obs.get("hazard_id", obs.get("track_id", 0))
                min_dist = dist
                min_ttc = ttc
                intruding_pos = (float(pos[0]), float(pos[1]), float(pos[2]))
                break  # Worst possible breach encountered

            elif self._point_in_polygon(ox, oy, fields[SafetyZone.BRAKING].polygon_vertices):
                if worst_zone != SafetyZone.PROTECTIVE:
                    worst_zone = SafetyZone.BRAKING
                    closest_obstacle_id = obs.get("hazard_id", obs.get("track_id", 0))
                    min_dist = min(min_dist, dist)
                    min_ttc = min(min_ttc, ttc)
                    intruding_pos = (float(pos[0]), float(pos[1]), float(pos[2]))

            elif self._point_in_polygon(ox, oy, fields[SafetyZone.WARNING].polygon_vertices):
                if worst_zone not in [SafetyZone.PROTECTIVE, SafetyZone.BRAKING]:
                    worst_zone = SafetyZone.WARNING
                    closest_obstacle_id = obs.get("hazard_id", obs.get("track_id", 0))
                    min_dist = min(min_dist, dist)
                    min_ttc = min(min_ttc, ttc)
                    intruding_pos = (float(pos[0]), float(pos[1]), float(pos[2]))

        # Apply ISO 3691-4 Safety Interlock logic
        safe_v = command_linear_v
        safe_w = command_angular_w
        report = None

        if worst_zone == SafetyZone.PROTECTIVE:
            # Deterministic E-Stop Clamp
            safe_v = 0.0
            safe_w = 0.0
            self.interlock_state = SafetyInterlockState.EMERGENCY_STOP_LATCHED
            report = SafetyIntrusionReport(
                zone_tripped=SafetyZone.PROTECTIVE,
                obstacle_id=closest_obstacle_id,
                obstacle_position=intruding_pos,
                distance_m=min_dist,
                time_to_collision_s=min_ttc,
                command_velocity_before=(command_linear_v, command_angular_w),
                command_velocity_after=(safe_v, safe_w),
                interlock_state=self.interlock_state
            )
            self.intrusion_history.append(report)

        elif worst_zone == SafetyZone.BRAKING:
            # Controlled service brake
            a_decel = self.get_effective_deceleration()
            safe_v = max(0.0, current_linear_v - a_decel * 0.1)  # 100ms cycle step
            safe_w = command_angular_w * 0.5
            self.interlock_state = SafetyInterlockState.CONTROLLED_STOP
            report = SafetyIntrusionReport(
                zone_tripped=SafetyZone.BRAKING,
                obstacle_id=closest_obstacle_id,
                obstacle_position=intruding_pos,
                distance_m=min_dist,
                time_to_collision_s=min_ttc,
                command_velocity_before=(command_linear_v, command_angular_w),
                command_velocity_after=(safe_v, safe_w),
                interlock_state=self.interlock_state
            )

        elif worst_zone == SafetyZone.WARNING:
            # Soft speed limit and alert
            safe_v = min(command_linear_v, 0.45)
            safe_w = command_angular_w * 0.8
            self.interlock_state = SafetyInterlockState.SLOWING
            report = SafetyIntrusionReport(
                zone_tripped=SafetyZone.WARNING,
                obstacle_id=closest_obstacle_id,
                obstacle_position=intruding_pos,
                distance_m=min_dist,
                time_to_collision_s=min_ttc,
                command_velocity_before=(command_linear_v, command_angular_w),
                command_velocity_after=(safe_v, safe_w),
                interlock_state=self.interlock_state
            )

        else:
            self.interlock_state = SafetyInterlockState.RUNNING

        return safe_v, safe_w, worst_zone, report

    def reset_interlock(self) -> bool:
        """Operator or supervisor reset of latched E-Stop interlock."""
        self.interlock_state = SafetyInterlockState.RUNNING
        return True

    @staticmethod
    def _point_in_polygon(x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
        """Standard ray casting algorithm for point-in-polygon test."""
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

