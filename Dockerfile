# ==============================================================================
# ROS 2 Edge Perception Container (Enterprise C++20 & Python Stack)
# Base: ROS 2 Humble Base on Ubuntu 22.04 LTS
# ==============================================================================
FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 1. Install system & ROS 2 C++ / Python development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    tar \
    python3-pip \
    python3-colcon-common-extensions \
    ros-humble-rclcpp \
    ros-humble-rclcpp-components \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-vision-msgs \
    ros-humble-diagnostic-msgs \
    ros-humble-cv-bridge \
    ros-humble-ament-cmake-python \
    ros-humble-eigen3-cmake-module \
    libeigen3-dev \
    libopencv-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Native ONNX Runtime C++ Library (v1.16.3) into /usr/local
RUN curl -sSL https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz | \
    tar -xz -C /usr/local --strip-components=1 && \
    ldconfig

# 3. Install Python ML & Testing Dependencies
RUN pip3 install --no-cache-dir \
    "numpy>=1.22.0,<2.0.0" \
    "opencv-python-headless>=4.8.0" \
    "onnxruntime>=1.16.0" \
    "psutil>=5.9.0" \
    "pytest>=7.0.0"

# 4. Setup ROS 2 workspace
WORKDIR /ros2_ws/src/ros2_edge_perception

# Copy project files
COPY . /ros2_ws/src/ros2_edge_perception

# Download / verify YOLOv8n ONNX model during image build
RUN python3 models/download_model.py

# 5. Build both C++20 Zero-Copy targets and Python packages
WORKDIR /ros2_ws
RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install --packages-select ros2_edge_perception \
    --cmake-args -DCMAKE_BUILD_TYPE=Release

# 6. Entrypoint script
COPY <<'EOF' /entrypoint.sh
#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
exec "$@"
EOF

RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Default launch command: starts camera streamer (synthetic) + perception node
CMD ["ros2", "launch", "ros2_edge_perception", "perception.launch.py"]
