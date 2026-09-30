#pragma once

#include "types.hpp"
#include <condition_variable>
#include <atomic>
#include <mutex>

namespace projectile {

class LatestFrameBuffer {
public:
	explicit LatestFrameBuffer(std::size_t capacity = 1) : capacity_(capacity < 1 ? 1 : capacity) {}
	bool push(FramePacket packet);
	bool wait_push(FramePacket packet, const std::atomic_bool& stop);
	bool wait_pop(FramePacket& packet, const std::atomic_bool& stop);
	void close();
	std::size_t size() const;
	std::uint64_t dropped() const { return dropped_; }
private:
	const std::size_t capacity_; mutable std::mutex mutex_; std::condition_variable cv_; std::deque<FramePacket> frames_; bool closed_ = false; std::uint64_t dropped_ = 0;
};

}
