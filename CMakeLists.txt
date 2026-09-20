cmake_minimum_required(VERSION 3.16)
project(ros2_edge_perception)

# ------------------------------------------------------------------------------
# C++ Standard & Compiler Flags
# ------------------------------------------------------------------------------
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

# Find ONNX Runtime C++
find_path(ONNXRUNTIME_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  HINTS
    /usr/local/include
    /usr/include
    $ENV{ONNXRUNTIME_ROOT}/include
  PATH_SUFFIXES onnxruntime
)

find_library(ONNXRUNTIME_LIB
  NAMES onnxruntime
  HINTS
    /usr/local/lib
    /usr/lib
    /usr/lib/x86_64-linux-gnu
    $ENV{ONNXRUNTIME_ROOT}/lib
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
  message(WARNING "ONNX Runtime C++ library or headers not found in standard paths. C++ perception node may fail to link unless ONNX Runtime is installed.")
endif()

# ------------------------------------------------------------------------------
# Include Directories
# ------------------------------------------------------------------------------
include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIR}
)

# ------------------------------------------------------------------------------
# C++ Libraries & Nodes
# ------------------------------------------------------------------------------

# 1. 3D Kalman Tracker Library
add_library(ros2_edge_perception_tracker_3d SHARED
  src/tracker_3d.cpp
)
target_include_directories(ros2_edge_perception_tracker_3d PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(ros2_edge_perception_tracker_3d
  Eigen3::Eigen
)

# 2. LiDAR-Camera Multi-Sensor Fusion Library
add_library(ros2_edge_perception_lidar_fusion SHARED
  src/lidar_camera_fusion.cpp
)
target_include_directories(ros2_edge_perception_lidar_fusion PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(ros2_edge_perception_lidar_fusion
  Eigen3::Eigen
)

# 3. Perception Node Component Library
add_library(ros2_edge_perception_node_component SHARED
  src/perception_node.cpp
)
target_include_directories(ros2_edge_perception_node_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR}
)
ament_target_dependencies(ros2_edge_perception_node_component
  rclcpp
  rclcpp_components
  sensor_msgs
  vision_msgs
  geometry_msgs
  std_msgs
  diagnostic_msgs
)
target_link_libraries(ros2_edge_perception_node_component
  ros2_edge_perception_tracker_3d
  ros2_edge_perception_lidar_fusion
  ${OpenCV_LIBRARIES}
  ${ONNXRUNTIME_LIB}
)
rclcpp_components_register_nodes(ros2_edge_perception_node_component "ros2_edge_perception::PerceptionNode")

# 4. Perception Node Executable
add_executable(perception_node_cpp
  src/perception_node.cpp
)
target_compile_definitions(perception_node_cpp PRIVATE PERCEPTION_NODE_MAIN=1)
ament_target_dependencies(perception_node_cpp
  rclcpp
  rclcpp_components
  sensor_msgs
  vision_msgs
  geometry_msgs
  std_msgs
  diagnostic_msgs
)
target_link_libraries(perception_node_cpp
  ros2_edge_perception_tracker_3d
  ros2_edge_perception_lidar_fusion
  ${OpenCV_LIBRARIES}
  ${ONNXRUNTIME_LIB}
)

# 5. Camera Streamer Node Executable
add_executable(camera_streamer_node_cpp
  src/camera_streamer_node.cpp
)
ament_target_dependencies(camera_streamer_node_cpp
  rclcpp
  sensor_msgs
)
target_link_libraries(camera_streamer_node_cpp
  ${OpenCV_LIBRARIES}
)

# ------------------------------------------------------------------------------
# Install C++ Targets
# ------------------------------------------------------------------------------
install(TARGETS
  ros2_edge_perception_tracker_3d
  ros2_edge_perception_lidar_fusion
  ros2_edge_perception_node_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS
  perception_node_cpp
  camera_streamer_node_cpp
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

# ------------------------------------------------------------------------------
# Install Python Modules & Scripts
# ------------------------------------------------------------------------------
ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  ros2_edge_perception/perception_node.py
  ros2_edge_perception/camera_streamer_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# ------------------------------------------------------------------------------
# Install Assets (Config, Launch, Models, Resource)
# ------------------------------------------------------------------------------
install(DIRECTORY launch config models resource
  DESTINATION share/${PROJECT_NAME}
)

# ------------------------------------------------------------------------------
# Testing
# ------------------------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_lint_auto QUIET)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_perception_pipeline tests/test_perception_pipeline.py)
  ament_add_pytest_test(test_tracker_3d tests/test_tracker_3d.py)
  ament_add_pytest_test(test_lidar_fusion tests/test_lidar_fusion.py)
endif()

# ------------------------------------------------------------------------------
# Package Export
# ------------------------------------------------------------------------------
ament_export_include_directories(include)
ament_export_libraries(
  ros2_edge_perception_tracker_3d
  ros2_edge_perception_lidar_fusion
  ros2_edge_perception_node_component
)
ament_export_dependencies(
  rclcpp
  rclcpp_components
  sensor_msgs
  vision_msgs
  geometry_msgs
  std_msgs
  diagnostic_msgs
)
ament_package()
