#pragma once
#include "config.hpp"
#include <deque>
#include <opencv2/video/background_segm.hpp>

namespace projectile {
struct MotionMasks { cv::Mat tophat; cv::Mat frame_diff; cv::Mat background; cv::Mat combined; cv::Mat cleaned; bool warmed_up = false; };
class MotionDetector {
public: explicit MotionDetector(const Config& config);
	MotionMasks process(const cv::Mat& bgr, const cv::Mat& gray);
private: Config config_; cv::Ptr<cv::BackgroundSubtractorMOG2> mog_; std::deque<cv::Mat> frames_; int bg_frames_ = 0;
};
}
