#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "ros2_edge_perception/tracker_3d.hpp"
#include "ros2_edge_perception/lidar_camera_fusion.hpp"

#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>

namespace ros2_edge_perception {

enum class SystemSafetyState {
    STARTUP,
    NOMINAL,
    DEGRADED,
    EMERGENCY_STOP
};

inline const char* safety_state_to_string(SystemSafetyState s) {
    switch (s) {
        case SystemSafetyState::STARTUP: return "STARTUP";
        case SystemSafetyState::NOMINAL: return "NOMINAL";
        case SystemSafetyState::DEGRADED: return "DEGRADED";
        case SystemSafetyState::EMERGENCY_STOP: return "EMERGENCY_STOP";
        default: return "UNKNOWN";
    }
}

class PerceptionNode : public rclcpp::Node {
public:
    explicit PerceptionNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
    ~PerceptionNode() override;

private:
    // ROS 2 Initialization & Parameters
    void init_parameters();
    void init_onnx_session(const std::string& model_path, const std::string& device);

    // Subscriptions
    void on_image(const sensor_msgs::msg::Image::ConstSharedPtr msg);
    void on_depth(const sensor_msgs::msg::Image::ConstSharedPtr msg);
    void on_camera_info(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg);
    void on_lidar(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);

    // Asynchronous Worker Pipeline
    void inference_worker();
    void process_frame(const cv::Mat& rgb_bgr, const cv::Mat& depth_m, const std_msgs::msg::Header& header);

    // Pre & Post Processing
    cv::Mat preprocess_letterbox(const cv::Mat& img, float& scale, float& pad_w, float& pad_h);
    std::vector<Detection3DInput> postprocess_yolov8(
        const float* output_tensor, size_t num_boxes, size_t num_classes,
        float scale, float pad_w, float pad_h, int orig_w, int orig_h,
        const cv::Mat& depth_m);

    // Watchdog & Safety Management
    void update_system_safety_state();

    // Publishing & Telemetry
    void publish_detections_2d(const std::vector<Detection3DInput>& detections, const std_msgs::msg::Header& header);
    void publish_detections_3d(const std::vector<TrackedObject3D>& tracks, const std_msgs::msg::Header& header);
    void evaluate_safety_and_trajectories(const std::vector<TrackedObject3D>& tracks, const std_msgs::msg::Header& header);
    void publish_diagnostics(double lat_pre, double lat_inf, double lat_post, double lat_total, size_t num_tracks, bool collision_alert);

    // Parameters
    std::string model_path_;
    std::string device_;
    float conf_threshold_;
    float iou_threshold_;
    float min_depth_m_;
    float max_depth_m_;
    float ttc_threshold_;
    bool enable_3d_;
    bool enable_tracking_;
    bool enable_lidar_fusion_{true};

    // Camera Intrinsics
    std::mutex intrinsics_mutex_;
    float fx_{554.25f}, fy_{554.25f}, cx_{320.0f}, cy_{240.0f};
    bool has_intrinsics_{false};

    // Subscriptions & Publishers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;

    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detections_2d_pub_;
    rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr detections_3d_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr trajectories_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr safety_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr state_pub_;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;

    // ONNX Runtime Engine
    Ort::Env ort_env_{ORT_LOGGING_LEVEL_WARNING, "ros2_perception"};
    std::unique_ptr<Ort::Session> ort_session_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;

    // 3D Kalman Tracker & LiDAR Fusion
    std::unique_ptr<MultiObjectTracker3D> tracker_;
    std::unique_ptr<LidarCameraFusionEngine> lidar_fusion_;

    // Asynchronous Ingestion Buffer
    std::mutex buffer_mutex_;
    std::condition_variable cv_frame_;
    cv::Mat latest_rgb_;
    cv::Mat latest_depth_;
    std_msgs::msg::Header latest_header_;
    bool has_new_frame_{false};
    std::atomic<bool> is_running_{true};
    std::thread worker_thread_;

    // LiDAR Ingestion Buffer
    std::mutex lidar_mutex_;
    std::vector<LidarPoint> latest_lidar_points_;
    rclcpp::Time last_lidar_time_{0, 0, RCL_ROS_TIME};
    rclcpp::Time last_camera_time_{0, 0, RCL_ROS_TIME};

    // Watchdog & State
    std::atomic<SystemSafetyState> safety_state_{SystemSafetyState::STARTUP};
    rclcpp::TimerBase::SharedPtr watchdog_timer_;

    // Telemetry State
    std::atomic<uint64_t> processed_frames_{0};
    std::atomic<uint64_t> dropped_frames_{0};
    double fps_timer_{0.0};
    uint32_t fps_counter_{0};
    double current_fps_{0.0};
};

} // namespace ros2_edge_perception
