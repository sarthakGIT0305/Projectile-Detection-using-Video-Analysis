 #include "projectile/camera_capture.hpp"
#include <chrono>
#include <iostream>
namespace projectile {
CameraCapture::CameraCapture(int i,int w,int h,double f,std::string path,LatestFrameBuffer& b):index_(i),width_(w),height_(h),fps_(f),video_path_(std::move(path)),buffer_(b){}
CameraCapture::~CameraCapture(){stop();}
void CameraCapture::start(){if(worker_.joinable())return;bool ok=video_path_.empty()?capture_.open(index_):capture_.open(video_path_);if(!ok){std::cerr<<(video_path_.empty()?"Unable to open camera "+std::to_string(index_):"Unable to open video "+video_path_)<<"\n";return;}if(video_path_.empty()){if(width_>0)capture_.set(cv::CAP_PROP_FRAME_WIDTH,width_);if(height_>0)capture_.set(cv::CAP_PROP_FRAME_HEIGHT,height_);if(fps_>0)capture_.set(cv::CAP_PROP_FPS,fps_);}opened_=true;worker_=std::thread(&CameraCapture::run,this);}
void CameraCapture::stop(){stopping_=true;buffer_.close();if(worker_.joinable())worker_.join();if(capture_.isOpened())capture_.release();opened_=false;}
void CameraCapture::run(){std::uint64_t s=0;while(!stopping_){cv::Mat f;if(!capture_.read(f)||f.empty()){std::cerr<<(video_path_.empty()?"Camera read failed or disconnected":"Video playback finished or failed")<<"\n";break;}FramePacket packet{f,s++,std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()};if(video_path_.empty()){buffer_.push(std::move(packet));}else if(!buffer_.wait_push(std::move(packet),stopping_)){break;}++captured_;}buffer_.close();}
}
