"""
Virtual Robot Autonomous Perception & Safety Bringup Launch File.

Orchestrates full closed-loop autonomy in simulation without hardware:
1. Virtual Robot Simulator (kinematics, arena obstacles, RGB-D & LiDAR generation)
2. Edge Perception Node (YOLOv8 + 3D Tracking + LiDAR Fusion + ASIL-B Watchdog)
3. Safety Controller Node (Autonomous Emergency Braking - AEB)
4. Web Visualizer Node (Live HTTP MJPEG Dashboard on port 8080)
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
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
    model_arg = DeclareLaunchArgument(
        "model_path",
        default_value=default_model_path,
        description="Path to the YOLOv8 ONNX model checkpoint",
    )

    port_arg = DeclareLaunchArgument(
        "port",
        default_value="8080",
        description="HTTP port for Web Visualizer Dashboard",
    )

    # --------------------------------------------------------------------------
    # Nodes
    # --------------------------------------------------------------------------
    # 1. Virtual Robot Simulator
    simulator_node = Node(
        package="ros2_edge_perception",
        executable="virtual_robot_simulator",
        name="virtual_robot_simulator",
        output="screen",
        parameters=[
            {"rate_hz": 20.0},
            {"arena_size_m": 10.0},
        ],
    )

    # 2. Perception Node
    perception_node = Node(
        package="ros2_edge_perception",
        executable="perception_node",
        name="perception_node",
        output="screen",
        parameters=[
            {"model_path": LaunchConfiguration("model_path")},
            {"device": "cpu"},
            {"enable_3d": True},
            {"enable_tracking": True},
            {"enable_lidar_fusion": True},
            {"publish_annotated": True},
            {"enable_diagnostics": True},
        ],
    )

    # 3. Safety Controller Node (AEB)
    safety_controller_node = Node(
        package="ros2_edge_perception",
        executable="safety_controller_node",
        name="safety_controller_node",
        output="screen",
        parameters=[
            {"brake_cooldown_sec": 1.5},
            {"allow_reverse_escape": True},
        ],
    )

    # 4. Web Visualizer Node
    web_visualizer_node = Node(
        package="ros2_edge_perception",
        executable="web_visualizer_node",
        name="web_visualizer_node",
        output="screen",
        parameters=[
            {"port": LaunchConfiguration("port")},
        ],
    )

    return LaunchDescription([
        model_arg,
        port_arg,
        simulator_node,
        perception_node,
        safety_controller_node,
        web_visualizer_node,
    ])

