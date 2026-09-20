"""
Production ROS 2 Launch File for Real Sensor Edge Perception (RGB-D / LiDAR).

Connects directly to physical hardware drivers:
- Intel RealSense (realsense2_camera): /camera/color/image_raw, /camera/aligned_depth_to_color/image_raw
- Stereolabs ZED (zed_wrapper): /zed/zed_node/rgb/image_rect_color, /zed/zed_node/depth/depth_registered
- Luxonis OAK-D (depthai_ros_driver): /oak/rgb/image_raw, /oak/stereo/image_raw
- Standard V4L2 / USB / RTSP Cameras: /camera/image_raw

NOTE: For synthetic test streams, use `launch/perception.launch.py source_type:=synthetic`.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    try:
        pkg_dir = get_package_share_directory("ros2_edge_perception")
        default_model_path = os.path.join(pkg_dir, "models", "yolov8n.onnx")
    except Exception:
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_model_path = os.path.join(pkg_dir, "models", "yolov8n.onnx")

    # --------------------------------------------------------------------------
    # Launch Arguments
    # --------------------------------------------------------------------------
    backend_arg = DeclareLaunchArgument(
        "backend",
        default_value="cpp",
        description="Execution backend: 'cpp' (C++20 Zero-Copy production) or 'python'",
    )

    device_arg = DeclareLaunchArgument(
        "device",
        default_value="cpu",
        description="Inference accelerator: 'cpu', 'cuda', 'rocm'",
    )

    model_arg = DeclareLaunchArgument(
        "model_path",
        default_value=default_model_path,
        description="Path to the YOLOv8 ONNX model checkpoint",
    )

    camera_type_arg = DeclareLaunchArgument(
        "camera_type",
        default_value="realsense",
        description="Real camera driver type: 'realsense', 'zed', 'oakd', 'v4l2', 'custom'",
    )

    image_topic_arg = DeclareLaunchArgument(
        "image_topic",
        default_value="/camera/color/image_raw",
        description="RGB camera topic from physical sensor driver",
    )

    depth_topic_arg = DeclareLaunchArgument(
        "depth_topic",
        default_value="/camera/aligned_depth_to_color/image_raw",
        description="Depth topic aligned to RGB frame from physical sensor driver",
    )

    camera_info_topic_arg = DeclareLaunchArgument(
        "camera_info_topic",
        default_value="/camera/color/camera_info",
        description="CameraInfo topic containing factory calibration intrinsics",
    )

    conf_thresh_arg = DeclareLaunchArgument(
        "conf_threshold",
        default_value="0.40",
        description="Detection confidence threshold",
    )

    use_cpp = PythonExpression(["'", LaunchConfiguration("backend"), "' == 'cpp'"])
    use_python = PythonExpression(["'", LaunchConfiguration("backend"), "' != 'cpp'"])

    # --------------------------------------------------------------------------
    # Production Perception Nodes (C++20 Backend)
    # --------------------------------------------------------------------------
    perception_node_cpp = Node(
        condition=IfCondition(use_cpp),
        package="ros2_edge_perception",
        executable="perception_node_cpp",
        name="perception_node_cpp",
        output="screen",
        parameters=[
            {
                "model_path": LaunchConfiguration("model_path"),
                "device": LaunchConfiguration("device"),
                "conf_threshold": LaunchConfiguration("conf_threshold"),
                "enable_3d_projection": True,
                "enable_tracking": True,
                "min_depth_meters": 0.2,
                "max_depth_meters": 10.0,
                "ttc_threshold_seconds": 2.0,
                "image_topic": LaunchConfiguration("image_topic"),
                "depth_topic": LaunchConfiguration("depth_topic"),
                "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                "detections_topic": "/perception/detections",
                "detections_3d_topic": "/perception/detections_3d",
                "trajectories_topic": "/perception/trajectories",
                "safety_alert_topic": "/perception/safety_alert",
                "diagnostics_topic": "/perception/diagnostics",
            }
        ],
    )

    # --------------------------------------------------------------------------
    # Production Perception Nodes (Python Backend)
    # --------------------------------------------------------------------------
    perception_node_py = Node(
        condition=IfCondition(use_python),
        package="ros2_edge_perception",
        executable="perception_node",
        name="perception_node",
        output="screen",
        parameters=[
            {
                "model_path": LaunchConfiguration("model_path"),
                "device": LaunchConfiguration("device"),
                "conf_threshold": LaunchConfiguration("conf_threshold"),
                "enable_3d_projection": True,
                "enable_tracking": True,
                "min_depth_meters": 0.2,
                "max_depth_meters": 10.0,
                "ttc_threshold_seconds": 2.0,
                "image_topic": LaunchConfiguration("image_topic"),
                "depth_topic": LaunchConfiguration("depth_topic"),
                "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                "detections_topic": "/perception/detections",
                "detections_3d_topic": "/perception/detections_3d",
                "trajectories_topic": "/perception/trajectories",
                "safety_alert_topic": "/perception/safety_alert",
                "diagnostics_topic": "/perception/diagnostics",
            }
        ],
    )

    return LaunchDescription(
        [
            backend_arg,
            device_arg,
            model_arg,
            camera_type_arg,
            image_topic_arg,
            depth_topic_arg,
            camera_info_topic_arg,
            conf_thresh_arg,
            LogInfo(msg=["Starting Real Sensor Perception Pipeline on camera: ", LaunchConfiguration("camera_type")]),
            perception_node_cpp,
            perception_node_py,
        ]
    )
