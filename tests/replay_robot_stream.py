#!/usr/bin/env python3
"""
Virtual Robot Replay Harness: Simulates a real robot observing a dynamic human obstacle.

Simulates an Autonomous Mobile Robot (AMR) driving in a facility:
1. Frame 1-20: Human standing at Z = 4.5 meters (Stationary).
2. Frame 21-80: Human walks directly towards the robot at 1.2 m/s (Z decreases from 4.5m to 1.5m).
3. Frame 81-120: Danger Zone! Time-To-Collision (TTC) drops under 1.5s -> Triggers Autonomous Emergency Braking (AEB).

Publishes:
- /camera/image_raw (sensor_msgs/Image, bgr8)
- /camera/depth/image_raw (sensor_msgs/Image, 16UC1 depth in mm)
- /camera/camera_info (sensor_msgs/CameraInfo)
- /lidar/points (sensor_msgs/PointCloud2)
"""

import math
import sys
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField


class VirtualRobotReplayNode(Node):
    def __init__(self, fps: float = 10.0, total_frames: int = 100):
        super().__init__("virtual_robot_replay_node")
        self.fps = fps
        self.total_frames = total_frames
        self.frame_idx = 0

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.image_pub = self.create_publisher(Image, "/camera/image_raw", sensor_qos)
        self.depth_pub = self.create_publisher(Image, "/camera/depth/image_raw", sensor_qos)
        self.info_pub = self.create_publisher(CameraInfo, "/camera/camera_info", sensor_qos)
        self.lidar_pub = self.create_publisher(PointCloud2, "/lidar/points", sensor_qos)

        # Standard Intel RealSense D435 Intrinsics (640x480)
        self.width = 640
        self.height = 480
        self.fx = 554.25
        self.fy = 554.25
        self.cx = 320.0
        self.cy = 240.0

        timer_period = 1.0 / self.fps
        self.timer = self.create_timer(timer_period, self._publish_step)
        self.get_logger().info(f"Virtual Robot Replay active: Simulating approaching human obstacle over {self.total_frames} frames...")

    def _publish_step(self):
        if self.frame_idx >= self.total_frames:
            self.get_logger().info("Replay sequence completed successfully!")
            rclpy.shutdown()
            return

        self.frame_idx += 1
        now = self.get_clock().now().to_msg()

        # Dynamic human distance simulation:
        # Stationary for first 20 frames at 4.5m, then walks forward at 1.2 m/s
        if self.frame_idx <= 20:
            current_z = 4.5
            state_desc = "STATIONARY (Z=4.5m)"
        else:
            elapsed_walk = (self.frame_idx - 20) / self.fps
            current_z = max(1.2, 4.5 - 1.2 * elapsed_walk)
            state_desc = f"APPROACHING ROBOT (Z={current_z:.2f}m, speed=-1.2 m/s)"

        # 1. Generate Synthetic RGB with realistic person silhouette
        rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        rgb[:, :] = (35, 35, 35)

        # Draw ground grid lines
        for x in range(0, self.width, 60):
            cv2.line(rgb, (x, 240), (int((x - 320) * 2.5 + 320), 480), (55, 55, 55), 1)
        for y in range(240, 480, 40):
            cv2.line(rgb, (0, y), (640, y), (55, 55, 55), 1)

        # Bounding box size scales inversely with depth Z
        box_h = int(1.7 * self.fy / current_z)
        box_w = int(0.6 * self.fx / current_z)
        cx_pix = 320
        cy_pix = int(240 + 0.2 * self.fy / current_z)

        x1 = max(0, cx_pix - box_w // 2)
        x2 = min(self.width, cx_pix + box_w // 2)
        y1 = max(0, cy_pix - box_h // 2)
        y2 = min(self.height, cy_pix + box_h // 2)

        # Draw Person (Orange body + head)
        color = (0, 140, 255) if current_z > 2.0 else (0, 0, 255) # Turns Red in danger zone!
        cv2.rectangle(rgb, (x1, y1), (x2, y2), color, -1)
        head_r = max(5, int(box_w * 0.3))
        cv2.circle(rgb, (cx_pix, max(0, y1 - head_r)), head_r, (255, 200, 180), -1)

        # HUD Overlay
        cv2.putText(rgb, f"Virtual Robot POV - Obstacle: {state_desc}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        ttc_val = current_z / 1.2 if current_z < 4.5 else 99.0
        ttc_color = (0, 255, 0) if ttc_val > 2.0 else (0, 0, 255)
        cv2.putText(rgb, f"Distance: {current_z:.2f}m | Est. TTC: {ttc_val:.2f}s", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, ttc_color, 2)

        # 2. Generate Synchronized 16UC1 Depth
        depth_mm = np.full((self.height, self.width), 5000, dtype=np.uint16)
        if y2 > y1 and x2 > x1:
            depth_mm[y1:y2, x1:x2] = int(current_z * 1000)

        # 3. Generate Synchronized 3D LiDAR Points
        lidar_pts = []
        for dy in np.linspace(-0.6, 0.6, 6):
            for dx in np.linspace(-0.25, 0.25, 5):
                lidar_pts.extend([float(dx), float(dy), float(current_z), 1.0])

        # Publish RGB
        rgb_msg = Image()
        rgb_msg.header.stamp = now
        rgb_msg.header.frame_id = "camera_optical_frame"
        rgb_msg.height, rgb_msg.width = self.height, self.width
        rgb_msg.encoding = "bgr8"
        rgb_msg.step = self.width * 3
        rgb_msg.data = rgb.tobytes()
        self.image_pub.publish(rgb_msg)

        # Publish Depth
        depth_msg = Image()
        depth_msg.header.stamp = now
        depth_msg.header.frame_id = "camera_optical_frame"
        depth_msg.height, depth_msg.width = self.height, self.width
        depth_msg.encoding = "16UC1"
        depth_msg.step = self.width * 2
        depth_msg.data = depth_mm.tobytes()
        self.depth_pub.publish(depth_msg)

        # Publish CameraInfo
        info_msg = CameraInfo()
        info_msg.header.stamp = now
        info_msg.header.frame_id = "camera_optical_frame"
        info_msg.width, info_msg.height = self.width, self.height
        info_msg.distortion_model = "plumb_bob"
        info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info_msg.k = [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]
        info_msg.p = [self.fx, 0.0, self.cx, 0.0, 0.0, self.fy, self.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.info_pub.publish(info_msg)

        # Publish LiDAR PointCloud2
        pc_msg = PointCloud2()
        pc_msg.header.stamp = now
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

        if self.frame_idx % 10 == 0:
            self.get_logger().info(f"[Frame {self.frame_idx:03d}/{self.total_frames}] Robot POV: {state_desc} -> Published RGB-D + LiDAR")


def main():
    rclpy.init()
    node = VirtualRobotReplayNode(fps=10.0, total_frames=80)
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

