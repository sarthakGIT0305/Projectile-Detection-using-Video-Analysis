#pragma once
#include "config.hpp"
#include "types.hpp"
#include "trail_store.hpp"
#include <unordered_map>

namespace projectile {
class TrajectoryAnalyzer {
public: explicit TrajectoryAnalyzer(const Config& config) : config_(config) {}
	std::unordered_map<int, ArcResult> analyze(const TrailStore& trails, const std::vector<Track>& tracks, cv::Size frame_size) const;
private: Config config_;
};
}
