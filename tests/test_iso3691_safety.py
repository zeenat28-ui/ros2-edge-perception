#!/usr/bin/env python3
"""
Unit tests for ISO 3691-4 / ISO 13849 (PLd) Dynamic Safety Supervisor.
"""

import pytest
import numpy as np
from ros2_edge_perception.iso3691_safety_field import (
    ISO3691DynamicSafetySupervisor,
    SafetyZone,
    SafetyInterlockState
)


@pytest.fixture
def supervisor():
    return ISO3691DynamicSafetySupervisor(
        robot_width_m=0.80,
        robot_length_m=1.20,
        lateral_margin_m=0.15,
        base_stopping_margin_m=0.25,
        system_reaction_time_s=0.080,
        max_decel_unladen_mps2=2.4,
        max_decel_laden_mps2=1.3,
        max_payload_kg=1200.0
    )


def test_stopping_distance_scaling(supervisor):
    """Stopping distance must strictly increase with velocity."""
    dist_05 = supervisor.calculate_stopping_distance(0.5)
    dist_15 = supervisor.calculate_stopping_distance(1.5)
    dist_25 = supervisor.calculate_stopping_distance(2.5)

    assert dist_05 < dist_15 < dist_25
    # S = 0.5 * 0.08 + (0.25)/(4.8) + 0.25 = 0.04 + 0.052 + 0.25 = 0.342m
    assert 0.30 <= dist_05 <= 0.40
    # At 2.0 m/s: 2*0.08 + 4/(4.8) + 0.25 = 0.16 + 0.833 + 0.25 = 1.243m
    dist_20 = supervisor.calculate_stopping_distance(2.0)
    assert 1.10 <= dist_20 <= 1.40


def test_payload_adaptive_deceleration(supervisor):
    """Braking distance must expand when payload increases (ISO 3691-4 stability)."""
    # Unladen
    supervisor.set_payload(0.0)
    stop_dist_empty = supervisor.calculate_stopping_distance(1.5)

    # Fully laden with 1,200 kg pallet
    supervisor.set_payload(1200.0)
    stop_dist_full = supervisor.calculate_stopping_distance(1.5)

    assert stop_dist_full > stop_dist_empty
    # Deceleration should drop from 2.4 to 1.3 m/s^2
    assert abs(supervisor.get_effective_deceleration() - 1.3) < 1e-3


def test_safety_zone_protective_clamp(supervisor):
    """An obstacle in Zone 3 (Protective) must clamp command velocity to 0.0 m/s."""
    supervisor.set_payload(0.0)
    v_cmd = 1.5
    w_cmd = 0.0
    v_curr = 1.5

    # Obstacle directly ahead at x = 0.8m, y = 0.0m (Inside protective field)
    obstacles = [{"hazard_id": 101, "centroid_3d": [0.8, 0.0, 0.1]}]

    safe_v, safe_w, zone, report = supervisor.evaluate_safety_and_clamp(
        v_cmd, w_cmd, v_curr, obstacles
    )

    assert zone == SafetyZone.PROTECTIVE
    assert safe_v == 0.0
    assert safe_w == 0.0
    assert supervisor.interlock_state == SafetyInterlockState.EMERGENCY_STOP_LATCHED
    assert report is not None
    assert report.obstacle_id == 101


def test_safety_zone_warning_throttling(supervisor):
    """An obstacle in Warning Zone must throttle speed and keep interlock running."""
    supervisor.set_payload(0.0)
    v_cmd = 1.5
    w_cmd = 0.0
    v_curr = 1.5

    # Warning field extends up to ~1.85 * stop_dist (stop_dist ~ 0.84m -> Warning ~ 1.55m from bumper)
    # Total x from robot center: 0.6 + 1.4 = 2.0m
    obstacles = [{"hazard_id": 102, "centroid_3d": [2.0, 0.0, 0.1]}]

    safe_v, safe_w, zone, report = supervisor.evaluate_safety_and_clamp(
        v_cmd, w_cmd, v_curr, obstacles
    )

    assert zone == SafetyZone.WARNING
    assert safe_v <= 0.50  # Throttled below 0.50 m/s
    assert supervisor.interlock_state == SafetyInterlockState.SLOWING


def test_clear_path_full_speed(supervisor):
    """When path is completely clear, commands must pass through unaltered."""
    supervisor.set_payload(0.0)
    v_cmd = 1.8
    w_cmd = 0.2
    v_curr = 1.8
    obstacles = []

    safe_v, safe_w, zone, report = supervisor.evaluate_safety_and_clamp(
        v_cmd, w_cmd, v_curr, obstacles
    )

    assert zone == SafetyZone.CLEAR
    assert safe_v == v_cmd
    assert safe_w == w_cmd
    assert supervisor.interlock_state == SafetyInterlockState.RUNNING

