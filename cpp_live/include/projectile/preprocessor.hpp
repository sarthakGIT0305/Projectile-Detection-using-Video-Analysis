#pragma once
#include "config.hpp"
#include <opencv2/core.hpp>
#include <utility>
#include <vector>

namespace projectile {
class Preprocessor {
public: explicit Preprocessor(const Config& config) : config_(config) {}
	std::pair<cv::Mat, cv::Point> roi(const cv::Mat& frame) const;
	cv::Mat gray(const cv::Mat& bgr) const;
	cv::Mat scale(const cv::Mat& frame) const;
private: const Config& config_;
};
}
