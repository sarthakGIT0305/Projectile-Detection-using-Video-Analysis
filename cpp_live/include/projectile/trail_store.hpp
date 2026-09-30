#pragma once
#include "types.hpp"
#include <unordered_map>

namespace projectile {
class TrailStore {
public: explicit TrailStore(std::size_t max_length) : max_length_(max_length) {}
	void update(const std::vector<Track>& tracks); const Trail* get(int id) const; std::vector<cv::Point2f> observed(int id) const; void prune(const std::vector<Track>& tracks);
private: std::size_t max_length_; std::unordered_map<int, Trail> trails_;
};
}
