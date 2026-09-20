"""
Test Image Publisher for ROS 2 Edge Perception Pipeline.

Downloads a verified real-world image (bus.jpg with persons and a bus)
and publishes synchronized RGB + Depth (2.0m) + CameraInfo to test
real-world object detection and 3D spatial deprojection.
"""

import os
import sys
import time
import urllib.request
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

TEST_IMAGE_URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg"
IMAGE_PATH = "/tmp/test_bus.jpg" if os.name != "nt" else os.path.join(os.environ.get("TEMP", "C:\\Temp"), "test_bus.jpg")


class TestImagePublisher(Node):

    def __init__(self):
        super().__init__("test_image_publisher")

        # Download test image if not present
        if not os.path.exists(IMAGE_PATH):
            self.get_logger().info(f"Downloading real test image from {TEST_IMAGE_URL}...")
            urllib.request.urlretrieve(TEST_IMAGE_URL, IMAGE_PATH)
            self.get_logger().info("Download complete.")

        # Read and resize to 640x480
        self.bgr = cv2.imread(IMAGE_PATH)
        if self.bgr is None:
            self.get_logger().error(f"Failed to read image from {IMAGE_PATH}")
            sys.exit(1)
        self.bgr = cv2.resize(self.bgr, (640, 480))
        self.h, self.w = self.bgr.shape[:2]

        # Simulated depth: 2000 mm (2.0 meters)
        self.depth_mm = np.full((self.h, self.w), 2000, dtype=np.uint16)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.rgb_pub = self.create_publisher(Image, "/camera/image_raw", sensor_qos)
        self.depth_pub = self.create_publisher(Image, "/camera/depth/image_raw", sensor_qos)
        self.info_pub = self.create_publisher(CameraInfo, "/camera/camera_info", sensor_qos)

        # Publish at 5 Hz
        self.timer = self.create_timer(0.2, self._publish)
        self.get_logger().info("Publishing real-world test image (bus + persons) at 5 Hz...")

    def _publish(self):
        stamp = self.get_clock().now().to_msg()
        frame_id = "camera_optical_frame"

        # RGB Image
        rgb_msg = Image()
        rgb_msg.header.stamp = stamp
        rgb_msg.header.frame_id = frame_id
        rgb_msg.height, rgb_msg.width, channels = self.bgr.shape
        rgb_msg.encoding = "bgr8"
        rgb_msg.step = self.w * channels
        rgb_msg.data = self.bgr.tobytes()
        self.rgb_pub.publish(rgb_msg)

        # Depth Image
        depth_msg = Image()
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = frame_id
        depth_msg.height, depth_msg.width = self.h, self.w
        depth_msg.encoding = "16UC1"
        depth_msg.step = self.w * 2
        depth_msg.data = self.depth_mm.tobytes()
        self.depth_pub.publish(depth_msg)

        # CameraInfo
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = frame_id
        info.width = self.w
        info.height = self.h
        info.distortion_model = "plumb_bob"
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.k = [554.25, 0.0, 320.0, 0.0, 554.25, 240.0, 0.0, 0.0, 1.0]
        self.info_pub.publish(info)


def main():
    rclpy.init()
    node = TestImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

