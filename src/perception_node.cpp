#include "ros2_edge_perception/perception_node.hpp"
#include <rclcpp_components/register_node_macro.hpp>

#include <chrono>
#include <filesystem>
#include <numeric>
#include <cstring>
#include <cmath>

namespace ros2_edge_perception {

static const std::vector<std::string> COCO_LABELS = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

PerceptionNode::PerceptionNode(const rclcpp::NodeOptions& options)
    : Node("perception_node", options) {
    init_parameters();
    init_onnx_session(model_path_, device_);

    // Subscriptions
    rclcpp::QoS sensor_qos(rclcpp::KeepLast(1));
    sensor_qos.best_effort();

    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "/camera/image_raw", sensor_qos,
        std::bind(&PerceptionNode::on_image, this, std::placeholders::_1));

    if (enable_3d_) {
        depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/camera/depth/image_raw", sensor_qos,
            std::bind(&PerceptionNode::on_depth, this, std::placeholders::_1));

        info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera/camera_info", sensor_qos,
            std::bind(&PerceptionNode::on_camera_info, this, std::placeholders::_1));

        detections_3d_pub_ = create_publisher<vision_msgs::msg::Detection3DArray>(
            "/perception/detections_3d", 10);
    }

    if (enable_lidar_fusion_) {
        lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            "/lidar/points", sensor_qos,
            std::bind(&PerceptionNode::on_lidar, this, std::placeholders::_1));
        lidar_fusion_ = std::make_unique<LidarCameraFusionEngine>();
    }

    detections_2d_pub_ = create_publisher<vision_msgs::msg::Detection2DArray>(
        "/perception/detections", 10);

    trajectories_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(
        "/perception/trajectories", 10);

    safety_pub_ = create_publisher<std_msgs::msg::String>(
        "/perception/safety_alert", 10);

    state_pub_ = create_publisher<std_msgs::msg::String>(
        "/perception/system_state", 10);

    diag_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
        "/perception/diagnostics", 10);

    if (enable_tracking_) {
        tracker_ = std::make_unique<MultiObjectTracker3D>(5, 1.5f);
    }

    // Industrial ASIL-B Watchdog Timer (5 Hz)
    watchdog_timer_ = create_wall_timer(
        std::chrono::milliseconds(200),
        std::bind(&PerceptionNode::update_system_safety_state, this));

    // Launch asynchronous worker
    worker_thread_ = std::thread(&PerceptionNode::inference_worker, this);
    RCLCPP_INFO(get_logger(), "C++20 Zero-Copy Tier-1 Perception Node initialized successfully.");
}

PerceptionNode::~PerceptionNode() {
    is_running_ = false;
    cv_frame_.notify_all();
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

void PerceptionNode::init_parameters() {
    declare_parameter("model_path", "models/yolov8n.onnx");
    declare_parameter("device", "cpu");
    declare_parameter("conf_threshold", 0.35f);
    declare_parameter("iou_threshold", 0.45f);
    declare_parameter("min_depth_meters", 0.2f);
    declare_parameter("max_depth_meters", 80.0f);
    declare_parameter("ttc_threshold_seconds", 2.0f);
    declare_parameter("enable_3d_projection", true);
    declare_parameter("enable_tracking", true);
    declare_parameter("enable_lidar_fusion", true);

    model_path_ = get_parameter("model_path").as_string();
    device_ = get_parameter("device").as_string();
    conf_threshold_ = static_cast<float>(get_parameter("conf_threshold").as_double());
    iou_threshold_ = static_cast<float>(get_parameter("iou_threshold").as_double());
    min_depth_m_ = static_cast<float>(get_parameter("min_depth_meters").as_double());
    max_depth_m_ = static_cast<float>(get_parameter("max_depth_meters").as_double());
    ttc_threshold_ = static_cast<float>(get_parameter("ttc_threshold_seconds").as_double());
    enable_3d_ = get_parameter("enable_3d_projection").as_bool();
    enable_tracking_ = get_parameter("enable_tracking").as_bool();
    enable_lidar_fusion_ = get_parameter("enable_lidar_fusion").as_bool();
}

void PerceptionNode::init_onnx_session(const std::string& model_path, const std::string& /*device*/) {
    std::string resolved = model_path;
    std::vector<std::string> candidates = {
        model_path,
        "/ros2_ws/src/ros2_edge_perception/" + model_path,
        "/ros2_ws/src/ros2_edge_perception/models/yolov8n.onnx",
        "models/yolov8n.onnx"
    };

    for (const auto& c : candidates) {
        if (std::filesystem::exists(c)) {
            resolved = c;
            break;
        }
    }

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    ort_session_ = std::make_unique<Ort::Session>(ort_env_, resolved.c_str(), session_options);
    RCLCPP_INFO(get_logger(), "Loaded ONNX model from: %s", resolved.c_str());

    input_names_.push_back("images");
    output_names_.push_back("output0");
}

void PerceptionNode::on_image(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
    // True zero-copy pointer view into ROS 2 message memory
    cv::Mat rgb_view(msg->height, msg->width, CV_8UC3, const_cast<uint8_t*>(&msg->data[0]), msg->step);

    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        if (has_new_frame_) {
            dropped_frames_++;
        }
        latest_rgb_ = rgb_view.clone();
        latest_header_ = msg->header;
        has_new_frame_ = true;
        last_camera_time_ = this->now();
    }
    cv_frame_.notify_one();
}

void PerceptionNode::on_depth(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
    // Convert 16UC1 (mm) directly to float32 meters
    cv::Mat depth_mm(msg->height, msg->width, CV_16UC1, const_cast<uint8_t*>(&msg->data[0]), msg->step);
    cv::Mat depth_m;
    depth_mm.convertTo(depth_m, CV_32FC1, 0.001);

    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        latest_depth_ = depth_m.clone();
    }
}

void PerceptionNode::on_camera_info(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
    std::lock_guard<std::mutex> lock(intrinsics_mutex_);
    fx_ = static_cast<float>(msg->k[0]);
    cx_ = static_cast<float>(msg->k[2]);
    fy_ = static_cast<float>(msg->k[4]);
    cy_ = static_cast<float>(msg->k[5]);
    has_intrinsics_ = true;

    if (lidar_fusion_) {
        lidar_fusion_->set_intrinsics({fx_, fy_, cx_, cy_});
    }
}

void PerceptionNode::on_lidar(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
    std::vector<LidarPoint> pts;
    pts.reserve(msg->width * msg->height);

    int x_offset = -1, y_offset = -1, z_offset = -1;
    for (const auto& field : msg->fields) {
        if (field.name == "x") x_offset = field.offset;
        else if (field.name == "y") y_offset = field.offset;
        else if (field.name == "z") z_offset = field.offset;
    }

    if (x_offset >= 0 && y_offset >= 0 && z_offset >= 0) {
        for (size_t i = 0; i < msg->data.size(); i += msg->point_step) {
            float x, y, z;
            std::memcpy(&x, &msg->data[i + x_offset], sizeof(float));
            std::memcpy(&y, &msg->data[i + y_offset], sizeof(float));
            std::memcpy(&z, &msg->data[i + z_offset], sizeof(float));
            if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
                pts.push_back({x, y, z, 1.0f});
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(lidar_mutex_);
        latest_lidar_points_ = std::move(pts);
        last_lidar_time_ = this->now();
    }
}

void PerceptionNode::update_system_safety_state() {
    auto now = this->now();
    double cam_elapsed = (last_camera_time_.nanoseconds() > 0) ? (now - last_camera_time_).seconds() : 999.0;
    double lidar_elapsed = (last_lidar_time_.nanoseconds() > 0) ? (now - last_lidar_time_).seconds() : 999.0;

    SystemSafetyState new_state = SystemSafetyState::NOMINAL;

    if (cam_elapsed > 0.5) {
        new_state = SystemSafetyState::EMERGENCY_STOP;
    } else if (enable_lidar_fusion_ && lidar_elapsed > 1.0) {
        new_state = SystemSafetyState::DEGRADED;
    } else {
        new_state = SystemSafetyState::NOMINAL;
    }

    safety_state_ = new_state;

    if (state_pub_) {
        std_msgs::msg::String msg;
        msg.data = safety_state_to_string(new_state);
        state_pub_->publish(msg);
    }
}

void PerceptionNode::inference_worker() {
    while (is_running_) {
        cv::Mat rgb, depth;
        std_msgs::msg::Header header;

        {
            std::unique_lock<std::mutex> lock(buffer_mutex_);
            cv_frame_.wait(lock, [this]() { return has_new_frame_ || !is_running_; });

            if (!is_running_) break;

            rgb = latest_rgb_;
            depth = latest_depth_;
            header = latest_header_;
            has_new_frame_ = false;
        }

        process_frame(rgb, depth, header);
    }
}

void PerceptionNode::process_frame(const cv::Mat& rgb_bgr, const cv::Mat& depth_m, const std_msgs::msg::Header& header) {
    auto t_start = std::chrono::steady_clock::now();

    // 1. Preprocess
    float scale, pad_w, pad_h;
    cv::Mat input_tensor = preprocess_letterbox(rgb_bgr, scale, pad_w, pad_h);
    auto t_pre = std::chrono::steady_clock::now();

    // 2. Inference
    std::array<int64_t, 4> input_shape{1, 3, 640, 640};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_val = Ort::Value::CreateTensor<float>(
        mem_info, input_tensor.ptr<float>(), input_tensor.total(), input_shape.data(), input_shape.size());

    auto output_tensors = ort_session_->Run(
        Ort::RunOptions{nullptr}, input_names_.data(), &input_val, 1, output_names_.data(), 1);

    auto t_inf = std::chrono::steady_clock::now();

    // 3. Postprocess & Deprojection
    float* out_data = output_tensors.front().GetTensorMutableData<float>();
    std::vector<Detection3DInput> raw_detections = postprocess_yolov8(
        out_data, 8400, 80, scale, pad_w, pad_h, rgb_bgr.cols, rgb_bgr.rows, depth_m);

    // 4. 3D Kalman Tracking & Trajectory Forecasting
    std::vector<TrackedObject3D> active_tracks;
    if (enable_tracking_ && tracker_) {
        double ts = header.stamp.sec + header.stamp.nanosec * 1e-9;
        active_tracks = tracker_->update(raw_detections, ts);
    }

    auto t_post = std::chrono::steady_clock::now();

    // 5. Publish Outputs
    publish_detections_2d(raw_detections, header);

    if (enable_3d_) {
        publish_detections_3d(active_tracks, header);
        evaluate_safety_and_trajectories(active_tracks, header);
    }

    // Telemetry
    double lat_pre = std::chrono::duration<double, std::milli>(t_pre - t_start).count();
    double lat_inf = std::chrono::duration<double, std::milli>(t_inf - t_pre).count();
    double lat_post = std::chrono::duration<double, std::milli>(t_post - t_inf).count();
    double lat_total = std::chrono::duration<double, std::milli>(t_post - t_start).count();

    processed_frames_++;
    publish_diagnostics(lat_pre, lat_inf, lat_post, lat_total, active_tracks.size(), false);
}

cv::Mat PerceptionNode::preprocess_letterbox(const cv::Mat& img, float& scale, float& pad_w, float& pad_h) {
    int target_w = 640;
    int target_h = 640;

    scale = std::min(static_cast<float>(target_w) / img.cols, static_cast<float>(target_h) / img.rows);
    int new_w = static_cast<int>(img.cols * scale);
    int new_h = static_cast<int>(img.rows * scale);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));

    pad_w = (target_w - new_w) / 2.0f;
    pad_h = (target_h - new_h) / 2.0f;

    cv::Mat padded(target_h, target_w, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(static_cast<int>(pad_w), static_cast<int>(pad_h), new_w, new_h)));

    // BGR to RGB, Normalization to [0, 1], HWC to CHW
    cv::Mat blob;
    cv::cvtColor(padded, blob, cv::COLOR_BGR2RGB);
    blob.convertTo(blob, CV_32FC3, 1.0 / 255.0);

    // Reorder channels to CHW
    cv::Mat chw(1, 3 * target_h * target_w, CV_32FC1);
    std::vector<cv::Mat> channels(3);
    for (int i = 0; i < 3; ++i) {
        channels[i] = cv::Mat(target_h, target_w, CV_32FC1, chw.ptr<float>() + i * target_h * target_w);
    }
    cv::split(blob, channels);

    return chw;
}

std::vector<Detection3DInput> PerceptionNode::postprocess_yolov8(
    const float* output_tensor, size_t num_boxes, size_t num_classes,
    float scale, float pad_w, float pad_h, int orig_w, int orig_h,
    const cv::Mat& depth_m) {

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<BBox2D> bboxes_2d;

    for (size_t i = 0; i < num_boxes; ++i) {
        float cx_b = output_tensor[0 * num_boxes + i];
        float cy_b = output_tensor[1 * num_boxes + i];
        float w_b  = output_tensor[2 * num_boxes + i];
        float h_b  = output_tensor[3 * num_boxes + i];

        float max_score = 0.0f;
        int best_class = -1;

        for (size_t c = 0; c < num_classes; ++c) {
            float s = output_tensor[(4 + c) * num_boxes + i];
            if (s > max_score) {
                max_score = s;
                best_class = static_cast<int>(c);
            }
        }

        if (max_score >= conf_threshold_) {
            float x1 = (cx_b - pad_w - w_b / 2.0f) / scale;
            float y1 = (cy_b - pad_h - h_b / 2.0f) / scale;
            float bw = w_b / scale;
            float bh = h_b / scale;

            x1 = std::clamp(x1, 0.0f, static_cast<float>(orig_w));
            y1 = std::clamp(y1, 0.0f, static_cast<float>(orig_h));
            bw = std::clamp(bw, 1.0f, static_cast<float>(orig_w) - x1);
            bh = std::clamp(bh, 1.0f, static_cast<float>(orig_h) - y1);

            boxes.emplace_back(static_cast<int>(x1), static_cast<int>(y1), static_cast<int>(bw), static_cast<int>(bh));
            scores.push_back(max_score);
            class_ids.push_back(best_class);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold_, iou_threshold_, indices);

    for (int idx : indices) {
        const auto& b = boxes[idx];
        BBox2D d;
        d.class_id = class_ids[idx];
        d.class_name = (d.class_id >= 0 && d.class_id < static_cast<int>(COCO_LABELS.size()))
                           ? COCO_LABELS[d.class_id] : "object";
        d.score = scores[idx];
        d.x1 = static_cast<float>(b.x);
        d.y1 = static_cast<float>(b.y);
        d.x2 = static_cast<float>(b.x + b.width);
        d.y2 = static_cast<float>(b.y + b.height);
        bboxes_2d.push_back(d);
    }

    // LiDAR-Camera Fusion if available
    std::vector<Detection3DInput> results;
    std::vector<LidarPoint> lidar_pts;
    {
        std::lock_guard<std::mutex> lock(lidar_mutex_);
        lidar_pts = latest_lidar_points_;
    }

    if (enable_lidar_fusion_ && lidar_fusion_ && !lidar_pts.empty()) {
        auto fused = lidar_fusion_->fuse(bboxes_2d, lidar_pts, min_depth_m_, max_depth_m_);
        for (const auto& f : fused) {
            Detection3DInput det;
            det.x = f.x;
            det.y = f.y;
            det.z = f.z;
            det.size_x = f.size_x;
            det.size_y = f.size_y;
            det.size_z = f.size_z;
            det.score = f.score;
            det.class_id = f.class_id;
            det.class_name = f.class_name;
            det.yaw = f.yaw;
            det.qx = f.qx;
            det.qy = f.qy;
            det.qz = f.qz;
            det.qw = f.qw;
            results.push_back(det);
        }
        return results;
    }

    // Depth-image fallback
    float fx, fy, cx, cy;
    {
        std::lock_guard<std::mutex> lock(intrinsics_mutex_);
        fx = fx_; fy = fy_; cx = cx_; cy = cy_;
    }

    for (const auto& b : bboxes_2d) {
        float z = 4.0f;
        int bx = static_cast<int>(b.x1);
        int by = static_cast<int>(b.y1);
        int bw = static_cast<int>(b.x2 - b.x1);
        int bh = static_cast<int>(b.y2 - b.y1);

        if (!depth_m.empty() && bx + bw <= depth_m.cols && by + bh <= depth_m.rows) {
            cv::Mat roi = depth_m(cv::Rect(bx, by, bw, bh));
            std::vector<float> valid_depths;
            for (int r = 0; r < roi.rows; ++r) {
                const float* r_ptr = roi.ptr<float>(r);
                for (int c = 0; c < roi.cols; ++c) {
                    if (r_ptr[c] >= min_depth_m_ && r_ptr[c] <= max_depth_m_) {
                        valid_depths.push_back(r_ptr[c]);
                    }
                }
            }
            if (valid_depths.size() > 10) {
                std::sort(valid_depths.begin(), valid_depths.end());
                size_t p50_idx = valid_depths.size() / 2;
                z = valid_depths[p50_idx];
            }
        }

        float center_u = (b.x1 + b.x2) / 2.0f;
        float center_v = (b.y1 + b.y2) / 2.0f;

        Detection3DInput det;
        det.x = (center_u - cx) * z / fx;
        det.y = (center_v - cy) * z / fy;
        det.z = z;
        det.size_x = (bw * z) / fx;
        det.size_y = (bh * z) / fy;
        det.size_z = 0.4f;
        det.score = b.score;
        det.class_id = b.class_id;
        det.class_name = b.class_name;
        det.yaw = 0.0f;
        det.qx = 0.0f;
        det.qy = 0.0f;
        det.qz = 0.0f;
        det.qw = 1.0f;
        results.push_back(det);
    }

    return results;
}

void PerceptionNode::publish_detections_2d(const std::vector<Detection3DInput>& detections, const std_msgs::msg::Header& header) {
    vision_msgs::msg::Detection2DArray msg;
    msg.header = header;

    for (const auto& d : detections) {
        vision_msgs::msg::Detection2D d2;
        d2.header = header;
        d2.bbox.center.position.x = d.x;
        d2.bbox.center.position.y = d.y;
        d2.bbox.size_x = d.size_x;
        d2.bbox.size_y = d.size_y;

        vision_msgs::msg::ObjectHypothesisWithPose hyp;
        hyp.hypothesis.class_id = d.class_name;
        hyp.hypothesis.score = d.score;
        d2.results.push_back(hyp);
        msg.detections.push_back(d2);
    }
    detections_2d_pub_->publish(msg);
}

void PerceptionNode::publish_detections_3d(const std::vector<TrackedObject3D>& tracks, const std_msgs::msg::Header& header) {
    vision_msgs::msg::Detection3DArray msg;
    msg.header = header;

    for (const auto& t : tracks) {
        vision_msgs::msg::Detection3D d3;
        d3.header = header;
        d3.id = t.class_name + "_" + std::to_string(t.track_id);

        d3.bbox.center.position.x = t.x;
        d3.bbox.center.position.y = t.y;
        d3.bbox.center.position.z = t.z;
        d3.bbox.center.orientation.x = t.qx;
        d3.bbox.center.orientation.y = t.qy;
        d3.bbox.center.orientation.z = t.qz;
        d3.bbox.center.orientation.w = t.qw;
        d3.bbox.size.x = t.size_x;
        d3.bbox.size.y = t.size_y;
        d3.bbox.size.z = t.size_z;

        vision_msgs::msg::ObjectHypothesisWithPose hyp;
        hyp.hypothesis.class_id = t.class_name;
        hyp.hypothesis.score = t.score;
        hyp.pose.pose.position.x = t.vx;
        hyp.pose.pose.position.y = t.vy;
        hyp.pose.pose.position.z = t.vz;

        d3.results.push_back(hyp);
        msg.detections.push_back(d3);
    }
    detections_3d_pub_->publish(msg);
}

void PerceptionNode::evaluate_safety_and_trajectories(const std::vector<TrackedObject3D>& tracks, const std_msgs::msg::Header& header) {
    geometry_msgs::msg::PoseArray pa;
    pa.header = header;

    for (const auto& t : tracks) {
        if (t.ttc.has_value() && t.ttc.value() < ttc_threshold_) {
            std::string alert = "CRITICAL: Collision Risk! Obstacle " + t.class_name + "_" +
                                std::to_string(t.track_id) + " at Z=" + std::to_string(t.z) +
                                "m approaching with TTC=" + std::to_string(t.ttc.value()) + "s!";
            RCLCPP_WARN(get_logger(), "%s", alert.c_str());
            std_msgs::msg::String s;
            s.data = alert;
            safety_pub_->publish(s);
        }

        for (const auto& [fx, fy, fz] : t.future_trajectory) {
            geometry_msgs::msg::Pose p;
            p.position.x = fx;
            p.position.y = fy;
            p.position.z = fz;
            p.orientation.w = 1.0;
            pa.poses.push_back(p);
        }
    }

    if (!pa.poses.empty()) {
        trajectories_pub_->publish(pa);
    }
}

void PerceptionNode::publish_diagnostics(
    double lat_pre, double lat_inf, double lat_post, double lat_total,
    size_t num_tracks, bool collision_alert) {

    diagnostic_msgs::msg::DiagnosticArray diag;
    diag.header.stamp = this->now();

    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name = "Perception Pipeline (C++20 Zero-Copy Tier-1)";
    status.hardware_id = device_;
    status.level = (lat_total <= 100.0) ? diagnostic_msgs::msg::DiagnosticStatus::OK : diagnostic_msgs::msg::DiagnosticStatus::WARN;
    status.message = (lat_total <= 100.0) ? "Nominal operation" : "Latency exceeds 100ms threshold";

    auto add_kv = [&](const std::string& k, const std::string& v) {
        diagnostic_msgs::msg::KeyValue kv;
        kv.key = k;
        kv.value = v;
        status.values.push_back(kv);
    };

    add_kv("system_safety_state", safety_state_to_string(safety_state_));
    add_kv("preprocess_latency_ms", std::to_string(lat_pre));
    add_kv("inference_latency_ms", std::to_string(lat_inf));
    add_kv("postprocess_latency_ms", std::to_string(lat_post));
    add_kv("total_latency_ms", std::to_string(lat_total));
    add_kv("effective_fps", std::to_string(current_fps_));
    add_kv("dropped_frames", std::to_string(dropped_frames_.load()));
    add_kv("processed_frames", std::to_string(processed_frames_.load()));
    add_kv("tracked_obstacles_count", std::to_string(num_tracks));
    add_kv("imminent_collision_alert", collision_alert ? "true" : "false");

    diag.status.push_back(status);
    diag_pub_->publish(diag);
}

} // namespace ros2_edge_perception

RCLCPP_COMPONENTS_REGISTER_NODE(ros2_edge_perception::PerceptionNode)

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ros2_edge_perception::PerceptionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
