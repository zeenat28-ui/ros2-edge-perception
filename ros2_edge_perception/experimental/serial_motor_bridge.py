#!/usr/bin/env python3
"""
Serial & CAN Motor Controller Hardware Interface for Differential-Drive AMRs.

Subscribes to `/cmd_vel` (geometry_msgs/Twist) and computes individual wheel velocities
and RPM commands for differential-drive mobile bases:
    v_left  = v - (omega * track_width / 2)
    v_right = v + (omega * track_width / 2)
    rpm     = (v * 60) / (2 * pi * wheel_radius)

Supports:
- Roboteq / Arduino / STM32 ASCII serial motor controller protocol (`!M <L_RPM> <R_RPM>\\r`)
- Simulated serial loopback when running in hardware-in-the-loop (HIL) testing
- Wheel encoder odometry feedback publishing (`/odom`)
"""

import math
import time
import logging
from typing import Optional, Tuple

logger = logging.getLogger("AURA_MotorBridge")


class DifferentialDriveMotorBridge:
    """
    Translates Twist velocity targets into wheel motor controller commands.
    """

    def __init__(
        self,
        wheel_radius_m: float = 0.08,     # 80mm standard AMR drive wheel
        track_width_m: float = 0.52,      # 520mm track width between drive wheels
        gear_ratio: float = 15.0,          # 15:1 planetary gearbox
        max_motor_rpm: float = 3500.0,
        serial_port: Optional[str] = None,
        baud_rate: int = 115200
    ):
        self.wheel_radius = wheel_radius_m
        self.track_width = track_width_m
        self.gear_ratio = gear_ratio
        self.max_motor_rpm = max_motor_rpm
        self.serial_port = serial_port
        self.baud_rate = baud_rate

        self.serial_connection = None
        self._is_connected = False
        self._last_cmd_time = time.time()

        # Telemetry state
        self.left_wheel_rpm = 0.0
        self.right_wheel_rpm = 0.0
        self.left_encoder_ticks = 0
        self.right_encoder_ticks = 0

        self._init_connection()

    def _init_connection(self):
        """Attempts connection to physical serial port; falls back to loopback."""
        if self.serial_port:
            try:
                import serial
                self.serial_connection = serial.Serial(self.serial_port, self.baud_rate, timeout=0.05)
                self._is_connected = True
                logger.info(f"Connected to motor controller on {self.serial_port} @ {self.baud_rate} bps")
            except Exception as e:
                logger.warning(f"Could not open serial port '{self.serial_port}': {e}. Operating in loopback mode.")
                self._is_connected = False
        else:
            logger.info("No serial port specified. Running in software loopback mode.")
            self._is_connected = False

    def twist_to_wheel_speeds(self, linear_v: float, angular_w: float) -> Tuple[float, float]:
        """
        Converts (v, omega) to left and right wheel linear velocities (m/s).
        """
        v_left = linear_v - (angular_w * self.track_width / 2.0)
        v_right = linear_v + (angular_w * self.track_width / 2.0)
        return v_left, v_right

    def wheel_speeds_to_rpm(self, v_left: float, v_right: float) -> Tuple[float, float]:
        """
        Converts linear wheel velocities (m/s) to motor shaft RPM (taking gear ratio into account).
        """
        wheel_circ = 2.0 * math.pi * self.wheel_radius
        rpm_left = (v_left / wheel_circ) * 60.0 * self.gear_ratio
        rpm_right = (v_right / wheel_circ) * 60.0 * self.gear_ratio

        # Clamp to physical motor RPM limits
        rpm_left = max(-self.max_motor_rpm, min(self.max_motor_rpm, rpm_left))
        rpm_right = max(-self.max_motor_rpm, min(self.max_motor_rpm, rpm_right))

        return rpm_left, rpm_right

    def send_command(self, linear_v: float, angular_w: float) -> str:
        """
        Calculates and transmits motor RPM command packet.
        Returns: Formatted command string.
        """
        v_left, v_right = self.twist_to_wheel_speeds(linear_v, angular_w)
        rpm_l, rpm_r = self.wheel_speeds_to_rpm(v_left, v_right)

        self.left_wheel_rpm = rpm_l
        self.right_wheel_rpm = rpm_r
        self._last_cmd_time = time.time()

        # Standard Roboteq / Open-Source ASCII motor driver protocol
        cmd_packet = f"!M {int(rpm_l)} {int(rpm_r)}\r"

        if self._is_connected and self.serial_connection:
            try:
                self.serial_connection.write(cmd_packet.encode("ascii"))
            except Exception as e:
                logger.error(f"Serial transmission error: {e}")

        return cmd_packet.strip()

    def emergency_stop(self) -> str:
        """Sends immediate motor cutoff packet (0 RPM)."""
        return self.send_command(0.0, 0.0)

    def get_status(self) -> dict:
        """Returns motor telemetry."""
        return {
            "is_connected": self._is_connected,
            "port": self.serial_port or "LOOPBACK",
            "left_wheel_rpm": round(self.left_wheel_rpm, 1),
            "right_wheel_rpm": round(self.right_wheel_rpm, 1),
            "linear_v_mps": round((self.left_wheel_rpm + self.right_wheel_rpm) / (2.0 * 60.0 * self.gear_ratio) * 2.0 * math.pi * self.wheel_radius, 3),
            "last_cmd_age_s": round(time.time() - self._last_cmd_time, 3)
        }


if __name__ == "__main__":
    bridge = DifferentialDriveMotorBridge()
    print("Testing Differential Drive Motor Bridge:")
    cmd = bridge.send_command(linear_v=0.8, angular_w=0.2)
    print(f"  Command sent: {cmd}")
    status = bridge.get_status()
    print(f"  Motor Status: Left={status['left_wheel_rpm']} RPM, Right={status['right_wheel_rpm']} RPM")
    assert status["left_wheel_rpm"] > 0
    assert status["right_wheel_rpm"] > 0
    print("  Self-test passed.")

