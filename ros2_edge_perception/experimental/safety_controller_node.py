#!/usr/bin/env python3
"""
Autonomous Emergency Braking (AEB) Safety Controller Node.

Intercepts raw teleoperation drive commands (/cmd_vel_teleop) and overrides them
with Autonomous Emergency Braking (AEB) when collision risks or watchdog system failures
are detected by the Edge Perception Engine.

Topics:
- Subscribes:
    * /cmd_vel_teleop (geometry_msgs/Twist) - Raw user driving inputs
    * /perception/safety_alert (std_msgs/String) - Real-time collision alerts from PerceptionNode
    * /perception/system_state (std_msgs/String) - ASIL-B Watchdog system state
- Publishes:
    * /cmd_vel (geometry_msgs/Twist) - Safe driving commands to robot/actuators
    * /safety/status (std_msgs/String) - Human-readable and JSON safety status
"""

import time
import json
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String as StringMsg


class SafetyControllerNode(Node):
    """Autonomous Emergency Braking (AEB) Safety Intercept Controller."""

    def __init__(self):
        super().__init__("safety_controller_node")

        # ----------------------------------------------------------------------
        # Parameters
        # ----------------------------------------------------------------------
        self.declare_parameter("brake_cooldown_sec", 1.5)
        self.declare_parameter("allow_reverse_escape", True)

        self.brake_cooldown_sec = self.get_parameter("brake_cooldown_sec").value
        self.allow_reverse_escape = self.get_parameter("allow_reverse_escape").value

        # ----------------------------------------------------------------------
        # Internal State
        # ----------------------------------------------------------------------
        self.latest_teleop = Twist()
        self.last_teleop_time = 0.0
        self.last_alert_time = 0.0
        self.last_alert_msg = ""
        self.system_state = "NOMINAL"

        self.brake_active = False
        self.interventions_count = 0

        # ----------------------------------------------------------------------
        # Subscriptions & Publishers
        # ----------------------------------------------------------------------
        self.teleop_sub = self.create_subscription(
            Twist,
            "/cmd_vel_teleop",
            self._teleop_callback,
            10,
        )

        self.alert_sub = self.create_subscription(
            StringMsg,
            "/perception/safety_alert",
            self._safety_alert_callback,
            10,
        )

        self.state_sub = self.create_subscription(
            StringMsg,
            "/perception/system_state",
            self._system_state_callback,
            10,
        )

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.status_pub = self.create_publisher(StringMsg, "/safety/status", 10)

        # 20 Hz Control & Watchdog loop
        self.timer = self.create_timer(0.05, self._control_loop)

        self.get_logger().info(
            "SafetyControllerNode (AEB) initialized. "
            "Listening for teleop on /cmd_vel_teleop -> Safe output to /cmd_vel."
        )

    def _teleop_callback(self, msg: Twist):
        """Receive raw user teleoperation command."""
        self.latest_teleop = msg
        self.last_teleop_time = time.time()

    def _safety_alert_callback(self, msg: StringMsg):
        """Trigger Autonomous Emergency Braking when collision alert is received."""
        now = time.time()
        self.last_alert_time = now
        self.last_alert_msg = msg.data

        if not self.brake_active:
            self.brake_active = True
            self.interventions_count += 1
            self.get_logger().warn(
                f"[AEB INTERVENTION #{self.interventions_count}] Collision Risk Detected! "
                f"Emergency Braking Activated: {msg.data}"
            )

    def _system_state_callback(self, msg: StringMsg):
        """Update system state from ASIL-B Watchdog."""
        self.system_state = msg.data.strip()

    def _control_loop(self):
        """Evaluate safety state and arbitrate safe actuator command."""
        now = time.time()
        safe_cmd = Twist()
        status_code = "NOMINAL"
        status_detail = "Normal operation"

        # Check if AEB cooldown has expired
        if self.brake_active:
            if now - self.last_alert_time > self.brake_cooldown_sec:
                self.brake_active = False
                self.get_logger().info("[AEB RECOVERY] Hazard cleared. Teleop control restored.")

        # Priority 1: System-level Emergency Stop (e.g. camera dead / sensor timeout)
        if self.system_state == "EMERGENCY_STOP":
            status_code = "ASIL_ESTOP"
            status_detail = "Hardware or sensor timeout in Perception Watchdog"
            safe_cmd.linear.x = 0.0
            safe_cmd.angular.z = 0.0

        # Priority 2: Active Collision Warning (Autonomous Emergency Braking)
        elif self.brake_active:
            status_code = "BRAKE_INTERVENTION"
            status_detail = self.last_alert_msg

            # Suppress forward drive
            if self.latest_teleop.linear.x > 0.0:
                safe_cmd.linear.x = 0.0  # Force halt forward motion
            elif self.allow_reverse_escape:
                # Permit backing up or rotating away from obstacle
                safe_cmd.linear.x = self.latest_teleop.linear.x
            else:
                safe_cmd.linear.x = 0.0

            # Allow steering away
            safe_cmd.angular.z = self.latest_teleop.angular.z

        # Priority 3: Nominal Operation
        else:
            # If teleop stream is alive (within last 0.5s), forward commands
            if now - self.last_teleop_time < 0.5:
                safe_cmd = self.latest_teleop
            else:
                safe_cmd.linear.x = 0.0
                safe_cmd.angular.z = 0.0

        # Publish safe command to actuators/simulator
        self.cmd_pub.publish(safe_cmd)

        # Publish Safety Status Telemetry
        status_msg = StringMsg()
        status_payload = {
            "status": status_code,
            "detail": status_detail,
            "brake_active": self.brake_active,
            "interventions_count": self.interventions_count,
            "safe_linear_v": safe_cmd.linear.x,
            "safe_angular_w": safe_cmd.angular.z,
        }
        status_msg.data = json.dumps(status_payload)
        self.status_pub.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SafetyControllerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()

