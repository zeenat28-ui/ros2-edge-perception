#include "ros2_edge_perception/tracker_3d.hpp"
#include <cmath>
#include <algorithm>

namespace ros2_edge_perception {

int Track3D::next_id_ = 1;

Track3D::Track3D(const Detection3DInput& det, double timestamp)
    : track_id_(next_id_++),
      class_name_(det.class_name),
      class_id_(det.class_id),
      score_(det.score),
      last_update_time_(timestamp) {
    x_.setZero();
    x_(0) = det.x;
    x_(1) = det.y;
    x_(2) = det.z;
    x_(3) = 0.0f; // vx
    x_(4) = 0.0f; // vy
    x_(5) = 0.0f; // vz
    x_(6) = det.size_x;
    x_(7) = det.size_y;
    x_(8) = det.size_z;

    P_.setZero();
    P_.diagonal() << 0.1f, 0.1f, 0.1f, 1.0f, 1.0f, 1.0f, 0.05f, 0.05f, 0.05f;
}

void Track3D::predict(double dt) {
    float dt_f = static_cast<float>(std::clamp(dt, 0.001, 0.5));

    // State transition matrix F
    Eigen::Matrix<float, 9, 9> F = Eigen::Matrix<float, 9, 9>::Identity();
    F(0, 3) = dt_f;
    F(1, 4) = dt_f;
    F(2, 5) = dt_f;

    // Extrapolate state: x = F * x
    x_ = F * x_;

    // Process noise covariance Q
    Eigen::Matrix<float, 9, 9> Q = Eigen::Matrix<float, 9, 9>::Zero();
    Q.diagonal() << q_pos_ * dt_f, q_pos_ * dt_f, q_pos_ * dt_f,
                    q_vel_ * dt_f, q_vel_ * dt_f, q_vel_ * dt_f,
                    q_size_ * dt_f, q_size_ * dt_f, q_size_ * dt_f;

    // Extrapolate covariance: P = F * P * F^T + Q
    P_ = F * P_ * F.transpose() + Q;

    age_++;
    lost_frames_++;
}

void Track3D::update(const Detection3DInput& det, double timestamp) {
    score_ = det.score;

    // Measurement vector z: [x, y, z, sx, sy, sz]
    Eigen::Matrix<float, 6, 1> z;
    z << det.x, det.y, det.z, det.size_x, det.size_y, det.size_z;

    // Measurement matrix H (6x9)
    Eigen::Matrix<float, 6, 9> H = Eigen::Matrix<float, 6, 9>::Zero();
    H(0, 0) = 1.0f;
    H(1, 1) = 1.0f;
    H(2, 2) = 1.0f;
    H(3, 6) = 1.0f;
    H(4, 7) = 1.0f;
    H(5, 8) = 1.0f;

    // Measurement noise R
    Eigen::Matrix<float, 6, 6> R = Eigen::Matrix<float, 6, 6>::Zero();
    R.diagonal() << 0.05f, 0.05f, 0.1f, 0.05f, 0.05f, 0.05f;

    // Innovation: y = z - H*x
    Eigen::Matrix<float, 6, 1> y = z - H * x_;

    // Innovation covariance: S = H * P * H^T + R
    Eigen::Matrix<float, 6, 6> S = H * P_ * H.transpose() + R;

    // Kalman Gain: K = P * H^T * S^-1
    Eigen::Matrix<float, 9, 6> K = P_ * H.transpose() * S.inverse();

    // Updated state: x = x + K*y
    x_ = x_ + K * y;

    // Updated covariance: P = (I - K*H) * P
    Eigen::Matrix<float, 9, 9> I = Eigen::Matrix<float, 9, 9>::Identity();
    P_ = (I - K * H) * P_;

    hits_++;
    lost_frames_ = 0;
    if (hits_ >= 2) {
        confirmed_ = true;
    }
    last_update_time_ = timestamp;
}

std::vector<std::tuple<float, float, float>> Track3D::predict_future_trajectory(float horizon, int steps) const {
    std::vector<std::tuple<float, float, float>> trajectory;
    float dt_step = horizon / static_cast<float>(steps);
    for (int i = 1; i <= steps; ++i) {
        float t_future = i * dt_step;
        float fx = x_(0) + x_(3) * t_future;
        float fy = x_(1) + x_(4) * t_future;
        float fz = x_(2) + x_(5) * t_future;
        trajectory.emplace_back(fx, fy, fz);
    }
    return trajectory;
}

std::optional<float> Track3D::compute_ttc() const {
    float z_dist = x_(2);
    float vz = x_(5);
    if (vz < -0.2f && z_dist > 0.1f) {
        return std::abs(z_dist / vz);
    }
    return std::nullopt;
}

void Track3D::compute_orientation(float& yaw, float& qx, float& qy, float& qz, float& qw) const {
    float vx = x_(3);
    float vz = x_(5);
    float speed_sq = vx * vx + vz * vz;

    // If speed > 0.2 m/s, align heading with velocity vector in X-Z ground plane
    if (speed_sq > 0.04f) {
        yaw = std::atan2(vx, vz); // Angle in camera coordinates relative to forward Z
        qx = 0.0f;
        qy = std::sin(yaw / 2.0f);
        qz = 0.0f;
        qw = std::cos(yaw / 2.0f);
    } else {
        yaw = 0.0f;
        qx = 0.0f;
        qy = 0.0f;
        qz = 0.0f;
        qw = 1.0f;
    }
}

MultiObjectTracker3D::MultiObjectTracker3D(int max_lost_frames, float match_distance_thresh)
    : max_lost_frames_(max_lost_frames),
      match_distance_thresh_(match_distance_thresh) {}

std::vector<TrackedObject3D> MultiObjectTracker3D::update(
    const std::vector<Detection3DInput>& detections, double timestamp) {
    double dt = last_timestamp_.has_value() ? std::max(0.001, timestamp - last_timestamp_.value()) : 0.033;
    last_timestamp_ = timestamp;

    // 1. Prediction step
    for (auto& [id, track] : tracks_) {
        track->predict(dt);
    }

    // 2. Greedy 3D Euclidean Association
    std::vector<int> track_ids;
    for (const auto& [id, track] : tracks_) {
        track_ids.push_back(id);
    }

    std::vector<bool> matched_detections(detections.size(), false);
    std::vector<bool> matched_tracks(track_ids.size(), false);

    for (size_t i = 0; i < track_ids.size(); ++i) {
        int tid = track_ids[i];
        const auto& track_state = tracks_[tid]->get_state();
        Eigen::Vector3f track_pos(track_state(0), track_state(1), track_state(2));

        int best_j = -1;
        float min_dist = match_distance_thresh_;

        for (size_t j = 0; j < detections.size(); ++j) {
            if (matched_detections[j]) continue;

            Eigen::Vector3f det_pos(detections[j].x, detections[j].y, detections[j].z);
            float dist = (track_pos - det_pos).norm();

            if (detections[j].class_name != tracks_[tid]->get_class_name()) {
                dist += 10.0f; // Class mismatch penalty
            }

            if (dist < min_dist) {
                min_dist = dist;
                best_j = static_cast<int>(j);
            }
        }

        if (best_j >= 0) {
            matched_tracks[i] = true;
            matched_detections[best_j] = true;
            tracks_[tid]->update(detections[best_j], timestamp);
        }
    }

    // 3. Create new tracks for unmatched detections
    for (size_t j = 0; j < detections.size(); ++j) {
        if (!matched_detections[j]) {
            auto new_track = std::make_unique<Track3D>(detections[j], timestamp);
            tracks_[new_track->get_id()] = std::move(new_track);
        }
    }

    // 4. Prune stale tracks
    for (auto it = tracks_.begin(); it != tracks_.end(); ) {
        if (it->second->get_lost_frames() > max_lost_frames_) {
            it = tracks_.erase(it);
        } else {
            ++it;
        }
    }

    // 5. Output confirmed tracks
    std::vector<TrackedObject3D> active_tracks;
    for (const auto& [id, track] : tracks_) {
        if (track->is_confirmed()) {
            const auto& s = track->get_state();
            TrackedObject3D obj;
            obj.track_id = track->get_id();
            obj.class_name = track->get_class_name();
            obj.class_id = track->get_class_id();
            obj.score = track->get_score();
            obj.x = s(0);
            obj.y = s(1);
            obj.z = s(2);
            obj.vx = s(3);
            obj.vy = s(4);
            obj.vz = s(5);
            obj.size_x = s(6);
            obj.size_y = s(7);
            obj.size_z = s(8);
            track->compute_orientation(obj.yaw, obj.qx, obj.qy, obj.qz, obj.qw);
            obj.future_trajectory = track->predict_future_trajectory(2.0f, 4);
            obj.ttc = track->compute_ttc();
            obj.lost_frames = track->get_lost_frames();
            active_tracks.push_back(obj);
        }
    }

    return active_tracks;
}

} // namespace ros2_edge_perception
