#include "projectile/blob_detector.hpp"
#include "projectile/frame_buffer.hpp"
#include "projectile/hungarian.hpp"
#include <opencv2/imgproc.hpp>
#include <cassert>
#include <iostream>
using namespace projectile;
int main(){
	auto assignment=hungarian({{4,1},{2,3}});assert(assignment.size()==2);
	Config c;c.isolation_filter=false;c.min_area=2;c.max_area=80;c.min_circularity=0.1;c.min_solidity=0.1;
	cv::Mat mask=cv::Mat::zeros(32,32,CV_8U);cv::circle(mask,{10,10},3,255,-1);BlobDetector d(c);auto detections=d.detect(mask);assert(!detections.empty());
	LatestFrameBuffer buffer(1);buffer.push({cv::Mat::zeros(2,2,CV_8U),1,0});buffer.push({cv::Mat::ones(2,2,CV_8U),2,0});assert(buffer.dropped()==1);std::atomic_bool stop=false;FramePacket packet;assert(buffer.wait_pop(packet,stop));assert(packet.sequence==2);
	std::cout<<"algorithm tests passed\n";return 0;
}
