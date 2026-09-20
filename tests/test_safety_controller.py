"""
Automated unit tests for Autonomous Emergency Braking (AEB) Safety Controller.
Verifies teleoperation pass-through, forward velocity clamping during collision risk,
reverse escape permission, and watchdog emergency stop.
"""

import time
import pytest


class MockSafetyArbitrator:
    """Simulates the arbitration logic of SafetyControllerNode without ROS 2 middleware."""

    def __init__(self, brake_cooldown_sec=1.5, allow_reverse_escape=True):
        self.brake_cooldown_sec = brake_cooldown_sec
        self.allow_reverse_escape = allow_reverse_escape
        self.brake_active = False
        self.last_alert_time = 0.0
        self.system_state = "NOMINAL"
        self.interventions_count = 0

    def trigger_collision_alert(self, current_time: float):
        self.last_alert_time = current_time
        if not self.brake_active:
            self.brake_active = True
            self.interventions_count += 1

    def set_system_state(self, state: str):
        self.system_state = state

    def arbitrate(self, teleop_linear_x: float, teleop_angular_z: float, current_time: float):
        # Check cooldown
        if self.brake_active and (current_time - self.last_alert_time > self.brake_cooldown_sec):
            self.brake_active = False

        # Priority 1: ASIL-B Watchdog E-Stop
        if self.system_state == "EMERGENCY_STOP":
            return 0.0, 0.0, "ASIL_ESTOP"

        # Priority 2: Collision Risk (AEB)
        if self.brake_active:
            safe_linear_x = 0.0
            if teleop_linear_x < 0.0 and self.allow_reverse_escape:
                safe_linear_x = teleop_linear_x  # Allow backing away
            return safe_linear_x, teleop_angular_z, "BRAKE_INTERVENTION"

        # Priority 3: Nominal
        return teleop_linear_x, teleop_angular_z, "NOMINAL"


def test_nominal_passthrough():
    """Verify normal teleop commands are passed through unchanged."""
    arb = MockSafetyArbitrator()
    lin_x, ang_z, status = arb.arbitrate(1.0, 0.5, current_time=10.0)
    assert lin_x == 1.0
    assert ang_z == 0.5
    assert status == "NOMINAL"


def test_collision_alert_clamps_forward():
    """Verify forward motion is clamped to 0 during collision risk."""
    arb = MockSafetyArbitrator()
    arb.trigger_collision_alert(current_time=10.0)

    # User attempts to drive forward
    lin_x, ang_z, status = arb.arbitrate(1.2, 0.0, current_time=10.2)
    assert lin_x == 0.0
    assert status == "BRAKE_INTERVENTION"
    assert arb.interventions_count == 1


def test_reverse_escape_allowed():
    """Verify driver is allowed to reverse away from an obstacle during AEB."""
    arb = MockSafetyArbitrator(allow_reverse_escape=True)
    arb.trigger_collision_alert(current_time=10.0)

    # User attempts to reverse away
    lin_x, ang_z, status = arb.arbitrate(-0.8, 0.3, current_time=10.3)
    assert lin_x == -0.8
    assert ang_z == 0.3
    assert status == "BRAKE_INTERVENTION"


def test_brake_recovery_after_cooldown():
    """Verify normal driving is automatically restored after obstacle clears."""
    arb = MockSafetyArbitrator(brake_cooldown_sec=1.5)
    arb.trigger_collision_alert(current_time=10.0)

    # During cooldown (at 10.5s): clamped
    lin_x, _, status = arb.arbitrate(1.0, 0.0, current_time=10.5)
    assert lin_x == 0.0
    assert status == "BRAKE_INTERVENTION"

    # After cooldown (at 12.0s): recovered
    lin_x, _, status = arb.arbitrate(1.0, 0.0, current_time=12.0)
    assert lin_x == 1.0
    assert status == "NOMINAL"


def test_watchdog_emergency_stop():
    """Verify hardware/sensor failure forces total halt regardless of user input."""
    arb = MockSafetyArbitrator()
    arb.set_system_state("EMERGENCY_STOP")

    # Even reversing is halted if system is in ASIL E-stop
    lin_x, ang_z, status = arb.arbitrate(-1.0, 1.0, current_time=10.0)
    assert lin_x == 0.0
    assert ang_z == 0.0
    assert status == "ASIL_ESTOP"

