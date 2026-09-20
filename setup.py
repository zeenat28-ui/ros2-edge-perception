import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'ros2_edge_perception'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test', 'tests']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Zeenat Riaz',
    maintainer_email='zeenatriaz468@gmail.com',
    description='Commercial-Grade Real-Time ROS 2 Perception Stack with Zero-Copy Ingestion, Asynchronous ONNX Runtime Inference, and Telemetry',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = ros2_edge_perception.perception_node:main',
            'camera_streamer_node = ros2_edge_perception.camera_streamer_node:main',
            'virtual_robot_simulator = ros2_edge_perception.virtual_robot_simulator:main',
            'safety_controller_node = ros2_edge_perception.safety_controller_node:main',
            'web_visualizer_node = ros2_edge_perception.web_visualizer_node:main',
        ],
    },
)


