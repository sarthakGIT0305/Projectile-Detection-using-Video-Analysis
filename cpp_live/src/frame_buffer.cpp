#include "projectile/frame_buffer.hpp"
namespace projectile {
bool LatestFrameBuffer::push(FramePacket p){std::lock_guard lock(mutex_);if(closed_)return false;if(frames_.size()>=capacity_){frames_.pop_front();++dropped_;}frames_.push_back(std::move(p));cv_.notify_one();return true;}
bool LatestFrameBuffer::wait_push(FramePacket p,const std::atomic_bool& stop){std::unique_lock lock(mutex_);cv_.wait(lock,[&]{return closed_||frames_.size()<capacity_||stop.load();});if(closed_||stop.load())return false;frames_.push_back(std::move(p));cv_.notify_one();return true;}
bool LatestFrameBuffer::wait_pop(FramePacket& p,const std::atomic_bool& stop){std::unique_lock lock(mutex_);cv_.wait(lock,[&]{return closed_||!frames_.empty()||stop.load();});if(frames_.empty())return false;p=std::move(frames_.back());frames_.clear();cv_.notify_all();return true;}
void LatestFrameBuffer::close(){std::lock_guard lock(mutex_);closed_=true;cv_.notify_all();}
std::size_t LatestFrameBuffer::size()const{std::lock_guard lock(mutex_);return frames_.size();}
}
