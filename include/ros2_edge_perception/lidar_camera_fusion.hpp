#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <optional>
#include <algorithm>
#include <cmath>

namespace ros2_edge_perception {

struct LidarPoint {
    float x;
    float y;
    float z;
    float intensity{1.0f};
};

struct Extrinsics {
    Eigen::Matrix3f R{Eigen::Matrix3f::Identity()};
    Eigen::Vector3f t{Eigen::Vector3f::Zero()};
};

struct Intrinsics {
    float fx{554.25f};
    float fy{554.25f};
    float cx{320.0f};
    float cy{240.0f};
};

struct BBox2D {
    std::string class_name;
    int class_id{0};
    float score{0.0f};
    float x1{0.0f};
    float y1{0.0f};
    float x2{0.0f};
    float y2{0.0f};
};

struct FusedDetection3D {
    std::string class_name;
    int class_id{0};
    float score{0.0f};
    float x{0.0f};      // Center X in camera optical frame (meters)
    float y{0.0f};      // Center Y in camera optical frame (meters)
    float z{0.0f};      // Center Z in camera optical frame (meters)
    float size_x{0.5f}; // Metric width
    float size_y{0.5f}; // Metric height
    float size_z{0.5f}; // Metric depth
    float yaw{0.0f};    // Heading angle in radians
    float qx{0.0f};     // Quaternion X
    float qy{0.0f};     // Quaternion Y
    float qz{0.0f};     // Quaternion Z
    float qw{1.0f};     // Quaternion W
    size_t lidar_point_count{0};
    bool fused_with_lidar{false};
};

class LidarCameraFusionEngine {
public:
    explicit LidarCameraFusionEngine(const Intrinsics& intrinsics = Intrinsics(),
                                     const Extrinsics& extrinsics = Extrinsics());

    void set_intrinsics(const Intrinsics& intrinsics);
    void set_extrinsics(const Extrinsics& extrinsics);

    // Project a single 3D LiDAR point into 2D camera pixel coordinates
    bool project_point(const LidarPoint& pt_lidar, float& u, float& v, float& depth_cam) const;

    // Fuse 2D vision detections with 3D LiDAR point cloud
    std::vector<FusedDetection3D> fuse(
        const std::vector<BBox2D>& detections_2d,
        const std::vector<LidarPoint>& lidar_points,
        float min_depth = 0.5f,
        float max_depth = 80.0f) const;

private:
    Intrinsics intrinsics_;
    Extrinsics extrinsics_;
};

} // namespace ros2_edge_perception

