#pragma once

#include <opencv2/core.hpp>
#include <cstdint>
#include <deque>
#include <optional>
#include <vector>

namespace projectile {

struct BoundingBox { int x = 0; int y = 0; int width = 0; int height = 0; };
struct Detection {
	BoundingBox box;
	cv::Point2f center;
	double area = 0.0;
	double circularity = 0.0;
	double solidity = 0.0;
	double aspect_ratio = 0.0;
};
struct Track {
	int id = -1;
	cv::Point2f center;
	cv::Point2f velocity;
	BoundingBox box;
	float speed = 0.0F;
	int missing = 0;
	int age = 0;
	bool predicted = false;
	double innovation = 0.0;
	std::optional<Detection> detection;
};
struct TrailPoint { cv::Point2f point; bool observed = false; };
using Trail = std::deque<TrailPoint>;
struct ArcResult { bool valid = false; std::vector<cv::Point> arc; std::vector<cv::Point> extrapolation; double residual = 0.0; };
struct FramePacket { cv::Mat frame; std::uint64_t sequence = 0; std::int64_t capture_time_us = 0; };

}
