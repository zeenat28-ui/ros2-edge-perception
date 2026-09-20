#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <optional>
#include <cmath>

namespace ros2_edge_perception {

struct Detection3DInput {
    float x;
    float y;
    float z;
    float size_x;
    float size_y;
    float size_z;
    float score;
    int class_id;
    std::string class_name;
    float yaw{0.0f};
    float qx{0.0f}, qy{0.0f}, qz{0.0f}, qw{1.0f};
};

struct TrackedObject3D {
    int track_id;
    std::string class_name;
    int class_id;
    float score;
    float x, y, z;
    float vx, vy, vz;
    float size_x, size_y, size_z;
    float yaw{0.0f};
    float qx{0.0f}, qy{0.0f}, qz{0.0f}, qw{1.0f};
    std::vector<std::tuple<float, float, float>> future_trajectory;
    std::optional<float> ttc;
    int lost_frames;
};

class Track3D {
public:
    Track3D(const Detection3DInput& det, double timestamp);

    void predict(double dt);
    void update(const Detection3DInput& det, double timestamp);
    std::vector<std::tuple<float, float, float>> predict_future_trajectory(float horizon = 2.0f, int steps = 4) const;
    std::optional<float> compute_ttc() const;
    void compute_orientation(float& yaw, float& qx, float& qy, float& qz, float& qw) const;

    int get_id() const { return track_id_; }
    const std::string& get_class_name() const { return class_name_; }
    int get_class_id() const { return class_id_; }
    float get_score() const { return score_; }
    bool is_confirmed() const { return confirmed_; }
    int get_lost_frames() const { return lost_frames_; }

    Eigen::Vector<float, 9> get_state() const { return x_; }

private:
    static int next_id_;
    int track_id_;
    std::string class_name_;
    int class_id_;
    float score_;

    // State vector: [x, y, z, vx, vy, vz, sx, sy, sz]
    Eigen::Matrix<float, 9, 1> x_;
    // State covariance matrix
    Eigen::Matrix<float, 9, 9> P_;

    float q_pos_ = 0.05f;
    float q_vel_ = 0.5f;
    float q_size_ = 0.01f;

    int hits_ = 1;
    int age_ = 1;
    int lost_frames_ = 0;
    bool confirmed_ = false;
    double last_update_time_;
};

class MultiObjectTracker3D {
public:
    explicit MultiObjectTracker3D(int max_lost_frames = 5, float match_distance_thresh = 1.5f);

    std::vector<TrackedObject3D> update(const std::vector<Detection3DInput>& detections, double timestamp);

private:
    int max_lost_frames_;
    float match_distance_thresh_;
    std::unordered_map<int, std::unique_ptr<Track3D>> tracks_;
    std::optional<double> last_timestamp_;
};

} // namespace ros2_edge_perception
