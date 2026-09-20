"""
Universal Camera & Depth Streamer Node for ROS 2 Edge Perception.

Supports:
1. 'synthetic' mode (default): Synchronized 30 FPS RGB + 16UC1 Depth + CameraInfo with moving 3D targets.
2. 'usb' mode: Local USB V4L2/DirectShow camera.
3. 'video' / 'rtsp' mode: Local video file or RTSP network camera.
"""

import math
import time
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField


class CameraStreamerNode(Node):
    """Universal RGB + Depth + LiDAR publisher supporting synthetic, USB, and network video streams."""

    def __init__(self):
        super().__init__("camera_streamer_node")

        # ----------------------------------------------------------------------
        # Parameters
        # ----------------------------------------------------------------------
        self.declare_parameter("source_type", "synthetic")  # 'synthetic', 'usb', 'video'
        self.declare_parameter("video_source", "0")          # Device index (e.g. 0) or RTSP/file path
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("frame_id", "camera_optical_frame")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("lidar_topic", "/lidar/points")
        self.declare_parameter("publish_synthetic_depth", True)
        self.declare_parameter("publish_synthetic_lidar", True)

        self.source_type = self.get_parameter("source_type").get_parameter_value().string_value.lower()
        self.video_source_str = self.get_parameter("video_source").get_parameter_value().string_value
        self.width = self.get_parameter("width").get_parameter_value().integer_value
        self.height = self.get_parameter("height").get_parameter_value().integer_value
        self.fps = self.get_parameter("fps").get_parameter_value().double_value
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        self.lidar_topic = self.get_parameter("lidar_topic").get_parameter_value().string_value
        self.publish_depth = self.get_parameter("publish_synthetic_depth").get_parameter_value().bool_value
        self.publish_lidar = self.get_parameter("publish_synthetic_lidar").get_parameter_value().bool_value

        self.get_logger().info(f"Starting CameraStreamerNode: source='{self.source_type}', target_fps={self.fps}")
        self.get_logger().info(f"Publishing RGB: '{self.image_topic}' | Depth: '{self.depth_topic}' | LiDAR: '{self.lidar_topic}'")

        # ----------------------------------------------------------------------
        # Video Capture Device Setup
        # ----------------------------------------------------------------------
        self.cap: Optional[cv2.VideoCapture] = None
        if self.source_type in ["usb", "video"]:
            src = int(self.video_source_str) if self.video_source_str.isdigit() else self.video_source_str
            self.cap = cv2.VideoCapture(src)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            if not self.cap.isOpened():
                self.get_logger().warn(f"Failed to open video source '{src}'. Falling back to synthetic pattern.")
                self.source_type = "synthetic"

        # ----------------------------------------------------------------------
        # ROS 2 Publishers & Timer
        # ----------------------------------------------------------------------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.image_pub = self.create_publisher(Image, self.image_topic, sensor_qos)
        self.depth_pub = self.create_publisher(Image, self.depth_topic, sensor_qos)
        self.info_pub = self.create_publisher(CameraInfo, self.camera_info_topic, sensor_qos)
        self.lidar_pub = self.create_publisher(PointCloud2, self.lidar_topic, sensor_qos)

        timer_period = 1.0 / max(1.0, self.fps)
        self.timer = self.create_timer(timer_period, self._publish_frame)

        # State for synthetic generator
        self.sim_frame_count = 0
        self.sim_start_time = time.time()

        # Camera Intrinsics (Standard 640x480 Pin-Hole Model)
        # fx=554.25, fy=554.25, cx=320.0, cy=240.0
        self.fx = 554.25
        self.fy = 554.25
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

    def _publish_frame(self):
        """Timer callback to fetch or generate synchronized RGB + Depth + LiDAR + CameraInfo."""
        now = self.get_clock().now()
        timestamp = now.to_msg()

        depth_frame: Optional[np.ndarray] = None
        lidar_points: List[float] = []

        if self.source_type == "synthetic":
            rgb_frame, depth_frame, lidar_points = self._generate_synthetic_rgbd_lidar()
        else:
            ret, rgb_frame = self.cap.read()
            if not ret or rgb_frame is None:
                if self.source_type == "video":
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, rgb_frame = self.cap.read()
                if not ret or rgb_frame is None:
                    self.get_logger().warn("Failed to read frame from video stream.")
                    return

            if rgb_frame.shape[1] != self.width or rgb_frame.shape[0] != self.height:
                rgb_frame = cv2.resize(rgb_frame, (self.width, self.height))

        # 1. Publish RGB Image
        rgb_msg = Image()
        rgb_msg.header.stamp = timestamp
        rgb_msg.header.frame_id = self.frame_id
        rgb_msg.height, rgb_msg.width, channels = rgb_frame.shape
        rgb_msg.encoding = "bgr8"
        rgb_msg.is_bigendian = 0
        rgb_msg.step = rgb_msg.width * channels
        rgb_msg.data = rgb_frame.tobytes()
        self.image_pub.publish(rgb_msg)

        # 2. Publish Depth Image (if available)
        if depth_frame is not None and self.publish_depth:
            depth_msg = Image()
            depth_msg.header.stamp = timestamp
            depth_msg.header.frame_id = self.frame_id
            depth_msg.height, depth_msg.width = depth_frame.shape
            depth_msg.encoding = "16UC1"  # 16-bit unsigned depth in millimeters
            depth_msg.is_bigendian = 0
            depth_msg.step = depth_msg.width * 2
            depth_msg.data = depth_frame.tobytes()
            self.depth_pub.publish(depth_msg)

        # 3. Publish CameraInfo (Intrinsics Matrix)
        info_msg = CameraInfo()
        info_msg.header.stamp = timestamp
        info_msg.header.frame_id = self.frame_id
        info_msg.width = self.width
        info_msg.height = self.height
        info_msg.distortion_model = "plumb_bob"
        info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info_msg.k = [
            self.fx, 0.0, self.cx,
            0.0, self.fy, self.cy,
            0.0, 0.0, 1.0,
        ]
        info_msg.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]
        info_msg.p = [
            self.fx, 0.0, self.cx, 0.0,
            0.0, self.fy, self.cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        self.info_pub.publish(info_msg)

        # 4. Publish Synthetic LiDAR PointCloud2
        if len(lidar_points) > 0 and self.publish_lidar:
            pc_msg = PointCloud2()
            pc_msg.header.stamp = timestamp
            pc_msg.header.frame_id = self.frame_id
            pc_msg.height = 1
            num_pts = len(lidar_points) // 4
            pc_msg.width = num_pts

            f_x = PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1)
            f_y = PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1)
            f_z = PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1)
            f_i = PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1)
            pc_msg.fields = [f_x, f_y, f_z, f_i]

            pc_msg.is_bigendian = False
            pc_msg.point_step = 16
            pc_msg.row_step = 16 * num_pts
            pc_msg.is_dense = True
            pc_msg.data = np.array(lidar_points, dtype=np.float32).tobytes()
            self.lidar_pub.publish(pc_msg)

    def _generate_synthetic_rgbd_lidar(self) -> tuple[np.ndarray, np.ndarray, list[float]]:
        """
        Generate synchronized RGB frame, 16-bit Depth frame (in mm), and LiDAR point cloud.
        Simulates 3D objects with distinct depths:
        - Background: 4000 mm (4.0 m)
        - Target 1 (Circle): 1800 mm (1.8 m)
        - Target 2 (Rectangle): 2600 mm (2.6 m)
        """
        self.sim_frame_count += 1
        elapsed = time.time() - self.sim_start_time
        lidar_pts = []

        # RGB Background
        rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        rgb[:, :] = (30, 30, 30)

        # Depth Background (4000 mm = 4.0 meters)
        depth = np.full((self.height, self.width), 4000, dtype=np.uint16)

        # Grid lines on RGB
        for x in range(0, self.width, 80):
            cv2.line(rgb, (x, 0), (x, self.height), (45, 45, 45), 1)
        for y in range(0, self.height, 80):
            cv2.line(rgb, (0, y), (self.width, y), (45, 45, 45), 1)

        # Target 1: Circle moving in Lissajous pattern (Depth: 1800 mm)
        cx1 = int(self.width / 2 + (self.width / 3) * math.sin(elapsed * 1.5))
        cy1 = int(self.height / 2 + (self.height / 3) * math.cos(elapsed * 2.0))
        cv2.circle(rgb, (cx1, cy1), 40, (0, 165, 255), -1)
        cv2.circle(rgb, (cx1, cy1), 20, (255, 255, 255), -1)
        cv2.circle(depth, (cx1, cy1), 40, 1800, -1)

        # Target 1 LiDAR ring points
        t1_z = 1.8
        t1_x = (cx1 - self.cx) * t1_z / self.fx
        t1_y = (cy1 - self.cy) * t1_z / self.fy
        for a in range(0, 360, 30):
            rad = a * math.pi / 180.0
            px = t1_x + 0.15 * math.cos(rad)
            py = t1_y + 0.15 * math.sin(rad)
            lidar_pts.extend([float(px), float(py), float(t1_z), 1.0])

        # Target 2: Rectangle moving horizontally (Depth: 2600 mm)
        rx = int((self.sim_frame_count * 5) % (self.width + 100) - 50)
        ry = int(self.height * 0.7)
        cv2.rectangle(rgb, (rx, ry), (rx + 80, ry + 60), (50, 205, 50), -1)

        rx_c = max(0, rx)
        ry_c = max(0, ry)
        rx_w = min(self.width, rx + 80)
        ry_h = min(self.height, ry + 60)
        if rx_w > rx_c and ry_h > ry_c:
            depth[ry_c:ry_h, rx_c:rx_w] = 2600

        # Target 2 LiDAR grid points
        t2_z = 2.6
        t2_x = (rx + 40.0 - self.cx) * t2_z / self.fx
        t2_y = (ry + 30.0 - self.cy) * t2_z / self.fy
        for dx in np.linspace(-0.2, 0.2, 5):
            for dy in np.linspace(-0.15, 0.15, 4):
                lidar_pts.extend([float(t2_x + dx), float(t2_y + dy), float(t2_z), 0.8])

        # Information Overlay
        title = "ROS 2 Edge Perception - Synced RGB-D + LiDAR Stream"
        cv2.putText(rgb, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        info_text = f"Frame: {self.sim_frame_count:06d} | RGB-D + LiDAR Synced | 640x480@30FPS"
        cv2.putText(rgb, info_text, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        return rgb, depth, lidar_pts

    def destroy_node(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraStreamerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
