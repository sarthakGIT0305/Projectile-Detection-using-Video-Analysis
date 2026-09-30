#pragma once
#include "config.hpp"
#include "motion_detector.hpp"
#include "types.hpp"
#include <vector>

namespace projectile {
class BlobDetector {
public: explicit BlobDetector(const Config& config) : config_(config) {}
	std::vector<Detection> detect(const cv::Mat& mask) const;
private: Config config_;
};
}
