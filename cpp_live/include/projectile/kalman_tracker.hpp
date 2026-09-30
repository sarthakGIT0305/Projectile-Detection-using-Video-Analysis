#pragma once
#include "config.hpp"
#include "types.hpp"
#include <opencv2/video/tracking.hpp>
#include <vector>
#include <optional>

namespace projectile {
class KalmanTracker {
public: explicit KalmanTracker(const Config& config); std::vector<Track> update(const std::vector<Detection>& detections); void reset();
private:
	struct State { int id; cv::KalmanFilter filter; int age=0; int missing=0; bool predicted=false; double innovation=0; std::optional<Detection> detection; };
	Config config_; std::vector<State> states_; int next_id_=0;
};
}
