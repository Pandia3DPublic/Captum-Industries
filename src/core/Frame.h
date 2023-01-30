#pragma once

#include <Eigen/Dense>
#include <Open3D/Open3D.h>
#include <Open3D/Geometry/KDTreeFlann.h>
#include <Open3D/Geometry/RGBDImage.h>
#include <iostream>
#include <string>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/cudafeatures2d.hpp>
#include "match.h"
#include "Optimizable.h"
#include "KeypointUnit.h"
#include <Cuda/Container/HashTableCuda.h>
#include "k4a/k4a.h"


//#include "realUtility.h"

//#define printstuf
//#ifdef printstuf
//#define log(x) std::cout<<x<<std::endl;
//#else
//#define log(x)
//#endif


using namespace open3d;

struct indexPointPair {
	Eigen::Vector3d p;
	int index;
};

class Frame : public KeypointUnit{
public:
    Frame();
    ~Frame();

	//basic variables
	//depth image is float imagein rgbd
	std::shared_ptr<geometry::RGBDImage> rgbd = std::make_shared<geometry::RGBDImage>(); //rgbd image, here depth is cutoff
	//std::shared_ptr<geometry::Image> depth; //16 bit depth image. Without cutoff filter
	std::shared_ptr<geometry::Image> rgblow;
	std::shared_ptr<geometry::Image> depthlow;
	std::shared_ptr<geometry::Image> segmentationImage;
	std::shared_ptr<k4a_imu_sample_t> imuSample;
	std::string rgbPath;
	std::string depthPath;
	bool integrated = false; //this is used in multiple threads
	bool worldtransset = false;
	bool duplicate = false; // to avoid double integration
	bool pushedIntoIntegrationBuffer = false;
	Eigen::Vector6d integrateddofs;
	std::shared_ptr<geometry::PointCloud> lowpcd;
	std::vector<cuda::HashEntry<cuda::Vector3i>> touchedSubvolumes; //subvolumes into which the pcd has been integrated. used for deintegration. Super necesssary!!!!
	Eigen::Matrix4d chunktransform = Eigen::Matrix4d::Identity(); //within the chunk


	//getter and setter taht are actually important!
	Eigen::Vector6d getWorlddofs();
	Eigen::Matrix4d getFrametoWorldTrans();
	void setFrametoWorldTrans(Eigen::Matrix4d a);
	void setworlddofs(Eigen::Vector6d& a);

	
private:
	Eigen::Matrix4d frametoworldtrans= Eigen::Matrix4d::Identity(); //world coordiante transformation. Is garanteed to be equal to worlddofs. Must be handled threadsafe with integrationlock
	Eigen::Vector6d worlddofs = Eigen::Vector6d::Zero(); //contains the dofs of the most current world trans. Is garanteed to be equal to frametoworldtrans



};