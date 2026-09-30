#pragma once
#include "blob_detector.hpp"
#include "camera_capture.hpp"
#include "config.hpp"
#include "frame_buffer.hpp"
#include "kalman_tracker.hpp"
#include "preprocessor.hpp"
#include "trajectory_analyzer.hpp"
#include "trail_store.hpp"
#include "visualizer.hpp"
#include <atomic>

namespace projectile {
class Pipeline {
public: explicit Pipeline(Config config); int run(); void request_stop() { stopping_ = true; buffer_.close(); }
private: Config config_; LatestFrameBuffer buffer_{1}; CameraCapture capture_; Preprocessor preprocessor_; MotionDetector motion_; BlobDetector detector_; KalmanTracker tracker_; TrailStore trails_; TrajectoryAnalyzer analyzer_; Visualizer visualizer_; std::atomic_bool stopping_{false};
};
}
