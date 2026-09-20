"""
Enterprise ROS 2 Launch File for 2D & 3D Edge Perception Pipeline.

Orchestrates:
1. Universal Camera & Depth Streamer (RGB, Depth 16UC1, CameraInfo)
2. Asynchronous 2D/3D Perception Node (Zero-Copy, ONNX Runtime, 3D Deprojection)
Supports both Python and C++20 Zero-Copy backends via 'backend' argument.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
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
        default_value="python",
        description="Execution backend: 'python' or 'cpp' (C++20 Zero-Copy)",
    )

    device_arg = DeclareLaunchArgument(
        "device",
        default_value="cpu",
        description="Inference accelerator: 'cpu', 'rocm', 'cuda', 'migraphx'",
    )

    model_arg = DeclareLaunchArgument(
        "model_path",
        default_value=default_model_path,
        description="Path to the YOLOv8 ONNX model checkpoint",
    )

    source_type_arg = DeclareLaunchArgument(
        "source_type",
        default_value="synthetic",
        description="Camera feed source: 'synthetic', 'usb', 'video'",
    )

    conf_thresh_arg = DeclareLaunchArgument(
        "conf_threshold",
        default_value="0.35",
        description="Confidence threshold for object detection",
    )

    enable_3d_arg = DeclareLaunchArgument(
        "enable_3d_projection",
        default_value="true",
        description="Enable 3D spatial deprojection using synchronized depth",
    )

    publish_annotated_arg = DeclareLaunchArgument(
        "publish_annotated_image",
        default_value="true",
        description="Whether to publish annotated bounding box image feed",
    )

    use_cpp = PythonExpression(["'", LaunchConfiguration("backend"), "' == 'cpp'"])
    use_python = PythonExpression(["'", LaunchConfiguration("backend"), "' != 'cpp'"])

    # --------------------------------------------------------------------------
    # Python Backend Nodes
    # --------------------------------------------------------------------------
    camera_node_py = Node(
        condition=IfCondition(use_python),
        package="ros2_edge_perception",
        executable="camera_streamer_node",
        name="camera_streamer_node",
        output="screen",
        parameters=[
            {
                "source_type": LaunchConfiguration("source_type"),
                "fps": 30.0,
                "width": 640,
                "height": 480,
                "image_topic": "/camera/image_raw",
                "depth_topic": "/camera/depth/image_raw",
                "camera_info_topic": "/camera/camera_info",
                "publish_synthetic_depth": True,
            }
        ],
    )

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
                "enable_3d_projection": LaunchConfiguration("enable_3d_projection"),
                "publish_annotated_image": LaunchConfiguration("publish_annotated_image"),
                "image_topic": "/camera/image_raw",
                "depth_topic": "/camera/depth/image_raw",
                "camera_info_topic": "/camera/camera_info",
                "detections_topic": "/perception/detections",
                "detections_3d_topic": "/perception/detections_3d",
                "diagnostics_topic": "/perception/diagnostics",
            }
        ],
    )

    # --------------------------------------------------------------------------
    # C++20 Zero-Copy Backend Nodes
    # --------------------------------------------------------------------------
    camera_node_cpp = Node(
        condition=IfCondition(use_cpp),
        package="ros2_edge_perception",
        executable="camera_streamer_node_cpp",
        name="camera_streamer_node_cpp",
        output="screen",
        parameters=[
            {
                "source_type": LaunchConfiguration("source_type"),
                "fps": 30.0,
                "width": 640,
                "height": 480,
                "image_topic": "/camera/image_raw",
                "depth_topic": "/camera/depth/image_raw",
                "camera_info_topic": "/camera/camera_info",
                "publish_synthetic_depth": True,
            }
        ],
    )

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
                "image_topic": "/camera/image_raw",
                "depth_topic": "/camera/depth/image_raw",
                "camera_info_topic": "/camera/camera_info",
            }
        ],
    )

    return LaunchDescription(
        [
            backend_arg,
            device_arg,
            model_arg,
            source_type_arg,
            conf_thresh_arg,
            enable_3d_arg,
            publish_annotated_arg,
            camera_node_py,
            perception_node_py,
            camera_node_cpp,
            perception_node_cpp,
        ]
    )
