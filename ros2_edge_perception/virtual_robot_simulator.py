#!/usr/bin/env python3
"""
Virtual Robot Simulator Node for Closed-Loop ROS 2 Autonomous Perception Testing.

Simulates an Autonomous Mobile Robot (AMR) with differential drive kinematics in a 10m x 10m
warehouse arena populated with static and dynamic obstacles.

Features:
- Closed-loop kinematics integration driven by /cmd_vel (geometry_msgs/Twist).
- World-to-Camera coordinate transformation and pinhole perspective projection.
- Synchronized sensor outputs:
    * /camera/image_raw (sensor_msgs/Image, bgr8)
    * /camera/depth/image_raw (sensor_msgs/Image, 16UC1 in mm)
    * /camera/camera_info (sensor_msgs/CameraInfo)
    * /lidar/points (sensor_msgs/PointCloud2)
    * /odom (nav_msgs/Odometry)
"""

import math
import time
import json
import threading
from typing import List, Dict, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, Point, Quaternion, Pose, TwistWithCovariance, PoseWithCovariance
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import String as StringMsg


class VirtualRobotSimulatorNode(Node):
    """Interactive differential-drive robot simulator in an obstacle arena."""

    def __init__(self):
        super().__init__("virtual_robot_simulator")

        # ----------------------------------------------------------------------
        # Parameters
        # ----------------------------------------------------------------------
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("arena_size_m", 10.0)
        self.declare_parameter("camera_width", 640)
        self.declare_parameter("camera_height", 480)
        self.declare_parameter("camera_fov_deg", 60.0)

        self.rate_hz = self.get_parameter("rate_hz").value
        self.arena_size = self.get_parameter("arena_size_m").value
        self.width = self.get_parameter("camera_width").value
        self.height = self.get_parameter("camera_height").value
        fov_deg = self.get_parameter("camera_fov_deg").value

        # Camera Intrinsics
        self.fx = (self.width / 2.0) / math.tan(math.radians(fov_deg / 2.0))
        self.fy = self.fx
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

        # ----------------------------------------------------------------------
        # Robot Kinematic State (World Frame: X East, Y North, Yaw Theta CCW)
        # ----------------------------------------------------------------------
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0  # radians (0 = pointing East along +X)
        self.target_linear_v = 0.0
        self.target_angular_w = 0.0
        self.current_linear_v = 0.0
        self.current_angular_w = 0.0

        # Robot physical limits
        self.max_linear_v = 1.5  # m/s
        self.max_angular_w = 2.0  # rad/s
        self.linear_accel = 2.0   # m/s^2
        self.angular_accel = 4.0  # rad/s^2

        # ----------------------------------------------------------------------
        # Obstacles in Arena
        # ----------------------------------------------------------------------
        # Obstacle definition: id, class_name, world_x, world_y, width, height, is_dynamic, vx, vy
        self.obstacles = [
            {
                "id": "human_1",
                "class_name": "person",
                "x": 3.0,
                "y": 0.0,
                "w": 0.6,
                "h": 1.75,
                "color": (0, 140, 255),
                "is_dynamic": False,
                "vx": 0.0,
                "vy": 0.0,
            },
            {
                "id": "crate_1",
                "class_name": "suitcase",
                "x": 2.0,
                "y": 2.2,
                "w": 0.8,
                "h": 0.8,
                "color": (200, 100, 50),
                "is_dynamic": False,
                "vx": 0.0,
                "vy": 0.0,
            },
            {
                "id": "human_patrol",
                "class_name": "person",
                "x": 4.5,
                "y": -2.0,
                "w": 0.6,
                "h": 1.75,
                "color": (0, 200, 200),
                "is_dynamic": True,
                "vx": 0.0,
                "vy": 0.8,
                "min_y": -3.0,
                "max_y": 1.0,
            },
        ]

        # ----------------------------------------------------------------------
        # ROS 2 Subscriptions & Publishers
        # ----------------------------------------------------------------------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Drive Command Subscription
        self.cmd_sub = self.create_subscription(
            Twist,
            "/cmd_vel",
            self._cmd_vel_callback,
            10,
        )

        # Sensor Stream Publishers
        self.image_pub = self.create_publisher(Image, "/camera/image_raw", sensor_qos)
        self.depth_pub = self.create_publisher(Image, "/camera/depth/image_raw", sensor_qos)
        self.info_pub = self.create_publisher(CameraInfo, "/camera/camera_info", sensor_qos)
        self.lidar_pub = self.create_publisher(PointCloud2, "/lidar/points", sensor_qos)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.arena_pub = self.create_publisher(StringMsg, "/virtual_robot/arena_state", 10)

        # Simulation Loop Timer
        dt = 1.0 / self.rate_hz
        self.last_update_time = time.time()
        self.timer = self.create_timer(dt, self._simulation_step)

        self.get_logger().info(
            f"Virtual Robot Simulator active at {self.rate_hz} Hz. "
            f"Listening to /cmd_vel. Robot at (0.0, 0.0), facing East."
        )

    def _cmd_vel_callback(self, msg: Twist):
        """Receive velocity commands."""
        self.target_linear_v = float(np.clip(msg.linear.x, -self.max_linear_v, self.max_linear_v))
        self.target_angular_w = float(np.clip(msg.angular.z, -self.max_angular_w, self.max_angular_w))

    def _simulation_step(self):
        """Main kinematic integration and sensor generation cycle."""
        now_time = time.time()
        dt = min(0.1, now_time - self.last_update_time)
        self.last_update_time = now_time

        # 1. Kinematics Update (Differential Drive with Accel Limiting)
        # Linear velocity slew
        dv = self.target_linear_v - self.current_linear_v
        max_dv = self.linear_accel * dt
        self.current_linear_v += float(np.clip(dv, -max_dv, max_dv))

        # Angular velocity slew
        dw = self.target_angular_w - self.current_angular_w
        max_dw = self.angular_accel * dt
        self.current_angular_w += float(np.clip(dw, -max_dw, max_dw))

        # Integrate pose in 2D plane
        self.robot_theta += self.current_angular_w * dt
        self.robot_theta = (self.robot_theta + math.pi) % (2.0 * math.pi) - math.pi  # Wrap [-pi, pi]

        self.robot_x += self.current_linear_v * math.cos(self.robot_theta) * dt
        self.robot_y += self.current_linear_v * math.sin(self.robot_theta) * dt

        # Arena bounds clamping
        half_arena = self.arena_size / 2.0 - 0.5
        self.robot_x = float(np.clip(self.robot_x, -half_arena, half_arena))
        self.robot_y = float(np.clip(self.robot_y, -half_arena, half_arena))

        # 2. Update Dynamic Obstacles
        for obs in self.obstacles:
            if obs["is_dynamic"]:
                obs["y"] += obs["vy"] * dt
                if obs["y"] > obs["max_y"]:
                    obs["y"] = obs["max_y"]
                    obs["vy"] = -abs(obs["vy"])
                elif obs["y"] < obs["min_y"]:
                    obs["y"] = obs["min_y"]
                    obs["vy"] = abs(obs["vy"])

        # 3. Publish Odometry
        ros_now = self.get_clock().now().to_msg()
        self._publish_odometry(ros_now)

        # 4. Generate & Publish Camera and LiDAR Streams
        self._generate_and_publish_sensors(ros_now)

        # 5. Publish Arena State Telemetry (for Web Visualizer)
        self._publish_arena_state()

    def _publish_odometry(self, ros_now):
        """Publish nav_msgs/Odometry message."""
        odom = Odometry()
        odom.header.stamp = ros_now
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = self.robot_x
        odom.pose.pose.position.y = self.robot_y
        odom.pose.pose.position.z = 0.0

        # Yaw to quaternion: q = [0, 0, sin(yaw/2), cos(yaw/2)]
        odom.pose.pose.orientation.x = 0.0
        odom.pose.pose.orientation.y = 0.0
        odom.pose.pose.orientation.z = math.sin(self.robot_theta / 2.0)
        odom.pose.pose.orientation.w = math.cos(self.robot_theta / 2.0)

        odom.twist.twist.linear.x = self.current_linear_v
        odom.twist.twist.angular.z = self.current_angular_w

        self.odom_pub.publish(odom)

    def _generate_and_publish_sensors(self, ros_now):
        """Project arena obstacles into Robot Camera POV and generate RGB, Depth, LiDAR."""
        # Canvas initialization
        rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        rgb[:, :] = (30, 30, 35)  # Dark industrial warehouse floor/walls
        depth_mm = np.full((self.height, self.width), 8000, dtype=np.uint16)
        lidar_pts = []

        # Draw ground grid lines for visual perspective
        for ground_y in range(int(self.cy), self.height, 35):
            cv2.line(rgb, (0, ground_y), (self.width, ground_y), (50, 50, 55), 1)

        # Robot position and orientation
        rx, ry, rth = self.robot_x, self.robot_y, self.robot_theta

        # Sort obstacles by distance so farther ones are drawn first (painter's algorithm)
        sorted_obs = sorted(
            self.obstacles,
            key=lambda o: math.hypot(o["x"] - rx, o["y"] - ry),
            reverse=True,
        )

        for obs in sorted_obs:
            dx_world = obs["x"] - rx
            dy_world = obs["y"] - ry

            # Transform into Robot Body Frame:
            # Forward: x_body = dx * cos(th) + dy * sin(th)
            # Left:    y_body = -dx * sin(th) + dy * cos(th)
            x_body = dx_world * math.cos(rth) + dy_world * math.sin(rth)
            y_body = -dx_world * math.sin(rth) + dy_world * math.cos(rth)

            # Camera Optical Frame:
            # X_cam = -y_body (Right)
            # Y_cam = 0.3 (Down, assuming camera is at height ~0.6m looking slightly down)
            # Z_cam = x_body (Forward Depth)
            z_cam = x_body
            x_cam = -y_body
            y_cam = 0.2  # slightly below horizon

            if z_cam < 0.4:
                continue  # Behind or too close to camera

            # Pinhole Camera Projection
            u_center = int(self.fx * (x_cam / z_cam) + self.cx)
            v_center = int(self.fy * (y_cam / z_cam) + self.cy)

            # Bounding box dimensions scale inversely with depth Z
            box_h = int(obs["h"] * self.fy / z_cam)
            box_w = int(obs["w"] * self.fx / z_cam)

            x1 = max(0, u_center - box_w // 2)
            x2 = min(self.width, u_center + box_w // 2)
            y1 = max(0, v_center - box_h // 2)
            y2 = min(self.height, v_center + box_h // 2)

            if x2 > x1 and y2 > y1 and z_cam < 8.0:
                # 1. Render Obstacle on RGB Image
                color = obs["color"]
                cv2.rectangle(rgb, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(rgb, (x1, y1), (x2, y2), (255, 255, 255), 2)

                # Head circle for person
                if obs["class_name"] == "person":
                    head_r = max(4, int(box_w * 0.25))
                    cv2.circle(rgb, (u_center, max(head_r, y1 - head_r)), head_r, (255, 220, 200), -1)

                # Text label
                cv2.putText(
                    rgb,
                    f"{obs['class_name']} ({z_cam:.1f}m)",
                    (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                # 2. Render Depth Image (16UC1 in mm)
                depth_val_mm = int(z_cam * 1000)
                depth_mm[y1:y2, x1:x2] = np.minimum(depth_mm[y1:y2, x1:x2], depth_val_mm)

                # 3. Generate 3D LiDAR Points in camera frame
                for dy in np.linspace(-obs["h"] / 2.0, obs["h"] / 2.0, 5):
                    for dx in np.linspace(-obs["w"] / 2.0, obs["w"] / 2.0, 5):
                        lidar_pts.extend([
                            float(x_cam + dx),
                            float(y_cam + dy),
                            float(z_cam),
                            1.0,  # Intensity
                        ])

        # HUD Overlay on Camera Image
        cv2.putText(
            rgb,
            f"Robot Pose: ({rx:.2f}m, {ry:.2f}m, {math.degrees(rth):.0f} deg) | V={self.current_linear_v:.2f} m/s",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 200),
            1,
        )

        # Publish RGB Image
        rgb_msg = Image()
        rgb_msg.header.stamp = ros_now
        rgb_msg.header.frame_id = "camera_optical_frame"
        rgb_msg.height, rgb_msg.width = self.height, self.width
        rgb_msg.encoding = "bgr8"
        rgb_msg.step = self.width * 3
        rgb_msg.data = rgb.tobytes()
        self.image_pub.publish(rgb_msg)

        # Publish Depth Image
        depth_msg = Image()
        depth_msg.header.stamp = ros_now
        depth_msg.header.frame_id = "camera_optical_frame"
        depth_msg.height, depth_msg.width = self.height, self.width
        depth_msg.encoding = "16UC1"
        depth_msg.step = self.width * 2
        depth_msg.data = depth_mm.tobytes()
        self.depth_pub.publish(depth_msg)

        # Publish CameraInfo
        info_msg = CameraInfo()
        info_msg.header.stamp = ros_now
        info_msg.header.frame_id = "camera_optical_frame"
        info_msg.width, info_msg.height = self.width, self.height
        info_msg.distortion_model = "plumb_bob"
        info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info_msg.k = [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]
        info_msg.p = [self.fx, 0.0, self.cx, 0.0, 0.0, self.fy, self.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.info_pub.publish(info_msg)

        # Publish PointCloud2
        pc_msg = PointCloud2()
        pc_msg.header.stamp = ros_now
        pc_msg.header.frame_id = "camera_optical_frame"
        pc_msg.height = 1
        num_pts = len(lidar_pts) // 4
        pc_msg.width = num_pts
        pc_msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        pc_msg.is_bigendian = False
        pc_msg.point_step = 16
        pc_msg.row_step = 16 * num_pts
        pc_msg.is_dense = True
        pc_msg.data = np.array(lidar_pts, dtype=np.float32).tobytes()
        self.lidar_pub.publish(pc_msg)

    def _publish_arena_state(self):
        """Publish JSON telemetry for Web Visualizer radar display."""
        payload = {
            "robot": {
                "x": self.robot_x,
                "y": self.robot_y,
                "theta": self.robot_theta,
                "linear_v": self.current_linear_v,
                "angular_w": self.current_angular_w,
            },
            "obstacles": [
                {
                    "id": o["id"],
                    "class_name": o["class_name"],
                    "x": o["x"],
                    "y": o["y"],
                    "w": o["w"],
                    "h": o["h"],
                }
                for o in self.obstacles
            ],
            "arena_size": self.arena_size,
        }
        msg = StringMsg()
        msg.data = json.dumps(payload)
        self.arena_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VirtualRobotSimulatorNode()
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

