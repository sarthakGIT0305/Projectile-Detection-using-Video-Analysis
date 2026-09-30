#pragma once

#include "frame_buffer.hpp"
#include <atomic>
#include <opencv2/videoio.hpp>
#include <thread>

namespace projectile {

class CameraCapture {
public:
	CameraCapture(int index, int width, int height, double fps, std::string video_path, LatestFrameBuffer& buffer);
	~CameraCapture();
	void start(); void stop(); bool is_open() const { return opened_; } std::uint64_t captured() const { return captured_; }
private:
	void run(); int index_; int width_; int height_; double fps_; std::string video_path_; LatestFrameBuffer& buffer_; cv::VideoCapture capture_; std::thread worker_; std::atomic_bool stopping_{false}; std::atomic_bool opened_{false}; std::atomic_uint64_t captured_{0};
};

}
