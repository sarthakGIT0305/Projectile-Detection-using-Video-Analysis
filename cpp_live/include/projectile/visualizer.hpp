#pragma once
#include "config.hpp"
#include "trajectory_analyzer.hpp"
#include "types.hpp"

namespace projectile {
class Visualizer {
public: explicit Visualizer(const Config& config) : config_(config) {}
	cv::Mat draw(const cv::Mat& frame, const std::vector<Track>& tracks, const TrailStore& trails, const std::unordered_map<int, ArcResult>& arcs, std::uint64_t frame_number, bool warmed_up) const;
private: Config config_;
};
}
