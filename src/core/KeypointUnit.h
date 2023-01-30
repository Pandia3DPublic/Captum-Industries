#pragma once
#include "match.h"
#include <vector>
#include "opencv2/core.hpp"
class KeypointUnit
{
public:
	KeypointUnit();
	~KeypointUnit();
	std::vector<c_keypoint> orbKeypoints; //contains all keypoints, after basic filtering for depth
	cv::Mat orbDescriptors; // redundant to c_keypoint descriptors for opencv
	std::vector<c_keypoint> efficientKeypoints; // contains only the non filtered keypoints that are used for sparse alignment. Filled by chunk
	//transformations
	Eigen::Matrix4d chunktoworldtrans = Eigen::Matrix4d::Identity(); //what comes out of chunk optimize
	int unique_id;


private:


};

