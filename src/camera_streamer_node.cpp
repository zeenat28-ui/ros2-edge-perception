#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

#include <opencv2/opencv.hpp>

#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <cstring>

namespace ros2_edge_perception {

class CameraStreamerNode : public rclcpp::Node {
public:
    explicit CameraStreamerNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : Node("camera_streamer_node", options),
          sim_frame_count_(0),
          sim_start_time_(std::chrono::steady_clock::now())
    {
        // ----------------------------------------------------------------------
        // Declare Parameters
        // ----------------------------------------------------------------------
        declare_parameter<std::string>("source_type", "synthetic");
        declare_parameter<std::string>("video_source", "0");
        declare_parameter<int>("width", 640);
        declare_parameter<int>("height", 480);
        declare_parameter<double>("fps", 30.0);
        declare_parameter<std::string>("frame_id", "camera_optical_frame");
        declare_parameter<std::string>("image_topic", "/camera/image_raw");
        declare_parameter<std::string>("depth_topic", "/camera/depth/image_raw");
        declare_parameter<std::string>("camera_info_topic", "/camera/camera_info");
        declare_parameter<std::string>("lidar_topic", "/lidar/points");
        declare_parameter<bool>("publish_synthetic_depth", true);
        declare_parameter<bool>("publish_synthetic_lidar", true);

        source_type_ = get_parameter("source_type").as_string();
        video_source_str_ = get_parameter("video_source").as_string();
        width_ = get_parameter("width").as_int();
        height_ = get_parameter("height").as_int();
        fps_ = get_parameter("fps").as_double();
        frame_id_ = get_parameter("frame_id").as_string();
        image_topic_ = get_parameter("image_topic").as_string();
        depth_topic_ = get_parameter("depth_topic").as_string();
        camera_info_topic_ = get_parameter("camera_info_topic").as_string();
        lidar_topic_ = get_parameter("lidar_topic").as_string();
        publish_depth_ = get_parameter("publish_synthetic_depth").as_bool();
        publish_lidar_ = get_parameter("publish_synthetic_lidar").as_bool();

        RCLCPP_INFO(get_logger(), "Starting C++20 Tier-1 CameraStreamerNode: source='%s', target_fps=%.1f",
                    source_type_.c_str(), fps_);
        RCLCPP_INFO(get_logger(), "Publishing RGB: '%s' | Depth: '%s' | LiDAR: '%s'",
                    image_topic_.c_str(), depth_topic_.c_str(), lidar_topic_.c_str());

        // ----------------------------------------------------------------------
        // Video Capture Device Setup
        // ----------------------------------------------------------------------
        if (source_type_ == "usb" || source_type_ == "video") {
            bool is_digit = !video_source_str_.empty() &&
                std::all_of(video_source_str_.begin(), video_source_str_.end(), ::isdigit);
            if (is_digit) {
                cap_.open(std::stoi(video_source_str_));
            } else {
                cap_.open(video_source_str_);
            }
            if (cap_.isOpened()) {
                cap_.set(cv::CAP_PROP_FRAME_WIDTH, width_);
                cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height_);
                cap_.set(cv::CAP_PROP_FPS, fps_);
            } else {
                RCLCPP_WARN(get_logger(), "Failed to open video source '%s'. Falling back to synthetic pattern.",
                            video_source_str_.c_str());
                source_type_ = "synthetic";
            }
        }

        // ----------------------------------------------------------------------
        // Publishers (Best Effort QoS, Depth 1)
        // ----------------------------------------------------------------------
        rclcpp::QoS sensor_qos(1);
        sensor_qos.best_effort();
        sensor_qos.keep_last(1);

        image_pub_ = create_publisher<sensor_msgs::msg::Image>(image_topic_, sensor_qos);
        depth_pub_ = create_publisher<sensor_msgs::msg::Image>(depth_topic_, sensor_qos);
        info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(camera_info_topic_, sensor_qos);
        lidar_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(lidar_topic_, sensor_qos);

        // Intrinsics (Standard Pin-Hole Model)
        fx_ = 554.25f;
        fy_ = 554.25f;
        cx_ = static_cast<float>(width_) / 2.0f;
        cy_ = static_cast<float>(height_) / 2.0f;

        // Timer
        auto timer_period = std::chrono::duration<double>(1.0 / std::max(1.0, fps_));
        timer_ = create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(timer_period),
            [this]() { publish_frame(); });
    }

    ~CameraStreamerNode() override {
        if (cap_.isOpened()) {
            cap_.release();
        }
    }

private:
    void publish_frame() {
        auto now = this->now();
        cv::Mat rgb_frame;
        cv::Mat depth_frame;
        std::vector<float> lidar_xyz_i; // x, y, z, intensity

        if (source_type_ == "synthetic") {
            generate_synthetic_rgbd_lidar(rgb_frame, depth_frame, lidar_xyz_i);
        } else {
            cap_ >> rgb_frame;
            if (rgb_frame.empty()) {
                if (source_type_ == "video") {
                    cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
                    cap_ >> rgb_frame;
                }
                if (rgb_frame.empty()) {
                    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Failed to capture frame.");
                    return;
                }
            }
            if (rgb_frame.cols != width_ || rgb_frame.rows != height_) {
                cv::resize(rgb_frame, rgb_frame, cv::Size(width_, height_));
            }
        }

        // 1. Publish RGB Image
        auto rgb_msg = std::make_unique<sensor_msgs::msg::Image>();
        rgb_msg->header.stamp = now;
        rgb_msg->header.frame_id = frame_id_;
        rgb_msg->height = rgb_frame.rows;
        rgb_msg->width = rgb_frame.cols;
        rgb_msg->encoding = "bgr8";
        rgb_msg->is_bigendian = 0;
        rgb_msg->step = static_cast<uint32_t>(rgb_frame.step[0]);
        size_t rgb_size = rgb_frame.total() * rgb_frame.elemSize();
        rgb_msg->data.resize(rgb_size);
        std::memcpy(rgb_msg->data.data(), rgb_frame.data, rgb_size);
        image_pub_->publish(std::move(rgb_msg));

        // 2. Publish Depth Image (if available)
        if (!depth_frame.empty() && publish_depth_) {
            auto depth_msg = std::make_unique<sensor_msgs::msg::Image>();
            depth_msg->header.stamp = now;
            depth_msg->header.frame_id = frame_id_;
            depth_msg->height = depth_frame.rows;
            depth_msg->width = depth_frame.cols;
            depth_msg->encoding = "16UC1";  // 16-bit unsigned depth in millimeters
            depth_msg->is_bigendian = 0;
            depth_msg->step = static_cast<uint32_t>(depth_frame.step[0]);
            size_t depth_size = depth_frame.total() * depth_frame.elemSize();
            depth_msg->data.resize(depth_size);
            std::memcpy(depth_msg->data.data(), depth_frame.data, depth_size);
            depth_pub_->publish(std::move(depth_msg));
        }

        // 3. Publish CameraInfo (Intrinsics Matrix)
        auto info_msg = std::make_unique<sensor_msgs::msg::CameraInfo>();
        info_msg->header.stamp = now;
        info_msg->header.frame_id = frame_id_;
        info_msg->width = width_;
        info_msg->height = height_;
        info_msg->distortion_model = "plumb_bob";
        info_msg->d = {0.0, 0.0, 0.0, 0.0, 0.0};
        info_msg->k = {
            fx_, 0.0, cx_,
            0.0, fy_, cy_,
            0.0, 0.0, 1.0
        };
        info_msg->r = {
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        };
        info_msg->p = {
            fx_, 0.0, cx_, 0.0,
            0.0, fy_, cy_, 0.0,
            0.0, 0.0, 1.0, 0.0
        };
        info_pub_->publish(std::move(info_msg));

        // 4. Publish Synthetic LiDAR PointCloud2
        if (!lidar_xyz_i.empty() && publish_lidar_) {
            auto pc_msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
            pc_msg->header.stamp = now;
            pc_msg->header.frame_id = frame_id_;
            pc_msg->height = 1;
            uint32_t num_pts = static_cast<uint32_t>(lidar_xyz_i.size() / 4);
            pc_msg->width = num_pts;

            sensor_msgs::msg::PointField fx, fy, fz, fi;
            fx.name = "x"; fx.offset = 0; fx.datatype = sensor_msgs::msg::PointField::FLOAT32; fx.count = 1;
            fy.name = "y"; fy.offset = 4; fy.datatype = sensor_msgs::msg::PointField::FLOAT32; fy.count = 1;
            fz.name = "z"; fz.offset = 8; fz.datatype = sensor_msgs::msg::PointField::FLOAT32; fz.count = 1;
            fi.name = "intensity"; fi.offset = 12; fi.datatype = sensor_msgs::msg::PointField::FLOAT32; fi.count = 1;
            pc_msg->fields = {fx, fy, fz, fi};

            pc_msg->is_bigendian = false;
            pc_msg->point_step = 16;
            pc_msg->row_step = pc_msg->point_step * pc_msg->width;
            pc_msg->is_dense = true;

            pc_msg->data.resize(lidar_xyz_i.size() * sizeof(float));
            std::memcpy(pc_msg->data.data(), lidar_xyz_i.data(), pc_msg->data.size());

            lidar_pub_->publish(std::move(pc_msg));
        }
    }

    void generate_synthetic_rgbd_lidar(cv::Mat& rgb, cv::Mat& depth, std::vector<float>& lidar_pts) {
        sim_frame_count_++;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - sim_start_time_).count();

        // RGB Background (dark gray)
        rgb = cv::Mat(height_, width_, CV_8UC3, cv::Scalar(30, 30, 30));

        // Depth Background (4000 mm = 4.0 meters)
        depth = cv::Mat(height_, width_, CV_16UC1, cv::Scalar(4000));

        // Grid lines on RGB
        for (int x = 0; x < width_; x += 80) {
            cv::line(rgb, cv::Point(x, 0), cv::Point(x, height_), cv::Scalar(45, 45, 45), 1);
        }
        for (int y = 0; y < height_; y += 80) {
            cv::line(rgb, cv::Point(0, y), cv::Point(width_, y), cv::Scalar(45, 45, 45), 1);
        }

        // Target 1: Circle moving in Lissajous pattern (Depth: 1.8m)
        int cx1 = static_cast<int>(width_ / 2.0 + (width_ / 3.0) * std::sin(elapsed * 1.5));
        int cy1 = static_cast<int>(height_ / 2.0 + (height_ / 3.0) * std::cos(elapsed * 2.0));
        cv::circle(rgb, cv::Point(cx1, cy1), 40, cv::Scalar(0, 165, 255), -1);
        cv::circle(rgb, cv::Point(cx1, cy1), 20, cv::Scalar(255, 255, 255), -1);
        cv::circle(depth, cv::Point(cx1, cy1), 40, cv::Scalar(1800), -1);

        // Generate LiDAR ring points on Target 1
        float t1_z = 1.8f;
        float t1_x = (cx1 - cx_) * t1_z / fx_;
        float t1_y = (cy1 - cy_) * t1_z / fy_;
        for (int a = 0; a < 360; a += 30) {
            float rad = a * 3.14159f / 180.0f;
            float px = t1_x + 0.15f * std::cos(rad);
            float py = t1_y + 0.15f * std::sin(rad);
            lidar_pts.push_back(px);
            lidar_pts.push_back(py);
            lidar_pts.push_back(t1_z);
            lidar_pts.push_back(1.0f); // intensity
        }

        // Target 2: Rectangle moving horizontally (Depth: 2.6m)
        int rx = static_cast<int>((sim_frame_count_ * 5) % (width_ + 100) - 50);
        int ry = static_cast<int>(height_ * 0.7);
        cv::rectangle(rgb, cv::Rect(rx, ry, 80, 60), cv::Scalar(50, 205, 50), -1);

        int rx_c = std::max(0, rx);
        int ry_c = std::max(0, ry);
        int rx_w = std::min(width_, rx + 80);
        int ry_h = std::min(height_, ry + 60);
        if (rx_w > rx_c && ry_h > ry_c) {
            depth(cv::Range(ry_c, ry_h), cv::Range(rx_c, rx_w)).setTo(cv::Scalar(2600));
        }

        // Generate LiDAR ring points on Target 2
        float t2_z = 2.6f;
        float t2_x = (rx + 40.0f - cx_) * t2_z / fx_;
        float t2_y = (ry + 30.0f - cy_) * t2_z / fy_;
        for (float dx = -0.2f; dx <= 0.2f; dx += 0.1f) {
            for (float dy = -0.15f; dy <= 0.15f; dy += 0.1f) {
                lidar_pts.push_back(t2_x + dx);
                lidar_pts.push_back(t2_y + dy);
                lidar_pts.push_back(t2_z);
                lidar_pts.push_back(0.8f);
            }
        }

        // Information Overlay
        cv::putText(rgb, "ROS 2 Edge Perception - Synced RGB-D + LiDAR Stream (C++20)",
                    cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 2);

        char info_text[128];
        std::snprintf(info_text, sizeof(info_text), "Frame: %06lu | RGB-D + LiDAR Synced | %dx%d@%.0fFPS",
                      sim_frame_count_, width_, height_, fps_);
        cv::putText(rgb, info_text, cv::Point(15, 60), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(180, 180, 180), 1);
    }

    std::string source_type_;
    std::string video_source_str_;
    int width_{640};
    int height_{480};
    double fps_{30.0};
    std::string frame_id_;
    std::string image_topic_;
    std::string depth_topic_;
    std::string camera_info_topic_;
    std::string lidar_topic_;
    bool publish_depth_{true};
    bool publish_lidar_{true};

    float fx_{554.25f};
    float fy_{554.25f};
    float cx_{320.0f};
    float cy_{240.0f};

    cv::VideoCapture cap_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr info_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    uint64_t sim_frame_count_;
    std::chrono::steady_clock::time_point sim_start_time_;
};

} // namespace ros2_edge_perception

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ros2_edge_perception::CameraStreamerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
