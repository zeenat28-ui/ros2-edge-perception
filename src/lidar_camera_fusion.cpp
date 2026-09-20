#include "ros2_edge_perception/lidar_camera_fusion.hpp"
#include <numeric>
#include <cmath>
#include <limits>
#include <string>

namespace ros2_edge_perception {

LidarCameraFusionEngine::LidarCameraFusionEngine(const Intrinsics& intrinsics,
                                                 const Extrinsics& extrinsics)
    : intrinsics_(intrinsics), extrinsics_(extrinsics) {}

void LidarCameraFusionEngine::set_intrinsics(const Intrinsics& intrinsics) {
    intrinsics_ = intrinsics;
}

void LidarCameraFusionEngine::set_extrinsics(const Extrinsics& extrinsics) {
    extrinsics_ = extrinsics;
}

bool LidarCameraFusionEngine::project_point(const LidarPoint& pt_lidar,
                                            float& u, float& v, float& depth_cam) const {
    // 1. Transform LiDAR 3D point into Camera Optical Frame: p_cam = R * p_lidar + t
    Eigen::Vector3f p_l(pt_lidar.x, pt_lidar.y, pt_lidar.z);
    Eigen::Vector3f p_c = extrinsics_.R * p_l + extrinsics_.t;

    // Reject points behind or too close to camera plane
    if (p_c.z() <= 0.1f) {
        return false;
    }

    depth_cam = p_c.z();

    // 2. Pin-Hole Perspective Projection
    u = (intrinsics_.fx * p_c.x()) / depth_cam + intrinsics_.cx;
    v = (intrinsics_.fy * p_c.y()) / depth_cam + intrinsics_.cy;

    return true;
}

std::vector<FusedDetection3D> LidarCameraFusionEngine::fuse(
    const std::vector<BBox2D>& detections_2d,
    const std::vector<LidarPoint>& lidar_points,
    float min_depth,
    float max_depth) const {

    std::vector<FusedDetection3D> fused_results;
    if (detections_2d.empty()) {
        return fused_results;
    }

    // Structure to hold projected point in camera space
    struct ProjectedPoint {
        float u, v;
        Eigen::Vector3f p_c;
    };

    std::vector<ProjectedPoint> valid_proj_points;
    valid_proj_points.reserve(lidar_points.size());

    for (const auto& pt : lidar_points) {
        float u, v, d;
        if (project_point(pt, u, v, d)) {
            if (d >= min_depth && d <= max_depth) {
                Eigen::Vector3f p_l(pt.x, pt.y, pt.z);
                Eigen::Vector3f p_c = extrinsics_.R * p_l + extrinsics_.t;
                valid_proj_points.push_back({u, v, p_c});
            }
        }
    }

    for (const auto& det : detections_2d) {
        FusedDetection3D fused;
        fused.class_name = det.class_name;
        fused.class_id = det.class_id;
        fused.score = det.score;
        fused.bbox_2d_center_x = (det.x1 + det.x2) / 2.0f;
        fused.bbox_2d_center_y = (det.y1 + det.y2) / 2.0f;
        fused.bbox_2d_width = det.x2 - det.x1;
        fused.bbox_2d_height = det.y2 - det.y1;

        // Collect all LiDAR points falling inside this 2D bounding box
        std::vector<Eigen::Vector3f> box_points;
        std::vector<float> depths;

        for (const auto& proj : valid_proj_points) {
            if (proj.u >= det.x1 && proj.u <= det.x2 &&
                proj.v >= det.y1 && proj.v <= det.y2) {
                box_points.push_back(proj.p_c);
                depths.push_back(proj.p_c.z());
            }
        }

        // Tier-1 Statistical Clustering: Need at least 3 points for confidence
        if (depths.size() >= 3) {
            std::sort(depths.begin(), depths.end());
            float median_depth = depths[depths.size() / 2];

            // Filter points within 15% distance window around median depth (eliminating background/foreground pass-through)
            float depth_window = std::max(0.4f, median_depth * 0.15f);
            std::vector<Eigen::Vector3f> clustered_points;

            for (const auto& p : box_points) {
                if (std::abs(p.z() - median_depth) <= depth_window) {
                    clustered_points.push_back(p);
                }
            }

            if (!clustered_points.empty()) {
                float min_x = clustered_points[0].x(), max_x = clustered_points[0].x();
                float min_y = clustered_points[0].y(), max_y = clustered_points[0].y();
                float min_z = clustered_points[0].z(), max_z = clustered_points[0].z();

                for (const auto& p : clustered_points) {
                    min_x = std::min(min_x, p.x());
                    max_x = std::max(max_x, p.x());
                    min_y = std::min(min_y, p.y());
                    max_y = std::max(max_y, p.y());
                    min_z = std::min(min_z, p.z());
                    max_z = std::max(max_z, p.z());
                }

                fused.x = (min_x + max_x) / 2.0f;
                fused.y = (min_y + max_y) / 2.0f;
                fused.z = (min_z + max_z) / 2.0f;

                fused.size_x = std::max(0.3f, max_x - min_x);
                fused.size_y = std::max(0.3f, max_y - min_y);
                fused.size_z = std::max(0.3f, max_z - min_z);

                // Estimate Heading (Yaw) using principal component of X-Z points
                if (clustered_points.size() >= 5) {
                    float mean_x = fused.x;
                    float mean_z = fused.z;
                    float cov_xx = 0.0f, cov_xz = 0.0f, cov_zz = 0.0f;
                    for (const auto& p : clustered_points) {
                        float dx = p.x() - mean_x;
                        float dz = p.z() - mean_z;
                        cov_xx += dx * dx;
                        cov_xz += dx * dz;
                        cov_zz += dz * dz;
                    }
                    // Heading angle in ground plane (X-Z)
                    fused.yaw = 0.5f * std::atan2(2.0f * cov_xz, cov_xx - cov_zz);
                    fused.qx = 0.0f;
                    fused.qy = std::sin(fused.yaw / 2.0f); // Rotation around camera Y (vertical)
                    fused.qz = 0.0f;
                    fused.qw = std::cos(fused.yaw / 2.0f);
                }

                fused.lidar_point_count = clustered_points.size();
                fused.fused_with_lidar = true;
                fused.is_valid_3d = true;
                fused.geometry_status = "LIDAR_FUSED";
                fused_results.push_back(fused);
                continue;
            }
        }

        // Fallback if no LiDAR points available inside this box:
        // Production safety: mark explicitly as unmeasured rather than fabricating arbitrary 3.0m geometry.
        fused.x = std::numeric_limits<float>::quiet_NaN();
        fused.y = std::numeric_limits<float>::quiet_NaN();
        fused.z = std::numeric_limits<float>::quiet_NaN();
        fused.size_x = 0.0f;
        fused.size_y = 0.0f;
        fused.size_z = 0.0f;
        fused.fused_with_lidar = false;
        fused.is_valid_3d = false;
        fused.geometry_status = "INVALID_UNMEASURED_DEPTH";
        fused_results.push_back(fused);
    }

    return fused_results;
}

} // namespace ros2_edge_perception

