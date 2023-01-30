#include "genKps.h"
#include "core/KeypointUnit.h"
#include "configvars.h"
//#include "cgf.h" in oldcode for now

bool goodpixel(float& x, float& y, shared_ptr<Frame> f) {
	if (*(f->rgbd->depth_.PointerAt<float>(x, y)) > 0 &&
		*(f->rgbd->depth_.PointerAt<float>(x + 1, y)) > 0 &&
		*(f->rgbd->depth_.PointerAt<float>(x, y + 1)) > 0 &&
		*(f->rgbd->depth_.PointerAt<float>(x - 1, y)) > 0 &&
		*(f->rgbd->depth_.PointerAt<float>(x, y - 1)) > 0 &&
		*(f->rgbd->depth_.PointerAt<float>(x - 1, y + 1)) > 0 &&
		*(f->rgbd->depth_.PointerAt<float>(x + 1, y + 1)) > 0 &&
		*(f->rgbd->depth_.PointerAt<float>(x - 1, y - 1)) > 0 &&
		*(f->rgbd->depth_.PointerAt<float>(x + 1, y - 1)) > 0) {
		return true;
	} else {
		return false;
	}
}
//todo put on gpu
void generateOrbKeypoints(std::shared_ptr<Frame> f, cv::Mat& cvColor) { //take opencv mat seperately
	if(g_nkeypoints == -1)
		utility::LogError("Number of keypoints is not set!\n");
	//check if adjacent pixels have depth values
	//auto goodpixel = [&](float x, float y)-> bool { // lambda function fuck yeah
	//	//x = (int)x;
	//	//y = (int)y;
	//	if (*(f->rgbd->depth_.PointerAt<float>(x, y)) > 0 &&
	//		*(f->rgbd->depth_.PointerAt<float>(x + 1, y)) > 0 &&
	//		*(f->rgbd->depth_.PointerAt<float>(x, y + 1)) > 0 &&
	//		*(f->rgbd->depth_.PointerAt<float>(x - 1, y)) > 0 &&
	//		*(f->rgbd->depth_.PointerAt<float>(x, y - 1)) > 0 &&
	//		*(f->rgbd->depth_.PointerAt<float>(x - 1, y + 1)) > 0 &&
	//		*(f->rgbd->depth_.PointerAt<float>(x + 1, y + 1)) > 0 &&
	//		*(f->rgbd->depth_.PointerAt<float>(x - 1, y - 1)) > 0 &&
	//		*(f->rgbd->depth_.PointerAt<float>(x + 1, y - 1)) > 0) {
	//		return true;
	//	}
	//	else {
	//		return false;
	//	}
	//};
	//check if adjacent pixels have depth values //todo doesnt work for some stupid reason
	//auto goodpixel = [&](float x, float y)-> bool { // lambda function fuck yeah
	//	//x = (int)x;
	//	//y = (int)y;
	//	if(*(f->depth->PointerAt<uint16_t>(x, y)) > 0 &&
	//		*(f->depth->PointerAt<uint16_t>(x + 1, y)) > 0 &&
	//		*(f->depth->PointerAt<uint16_t>(x, y + 1)) > 0 &&
	//		*(f->depth->PointerAt<uint16_t>(x - 1, y)) > 0 &&
	//		*(f->depth->PointerAt<uint16_t>(x, y - 1)) > 0 &&
	//		*(f->depth->PointerAt<uint16_t>(x - 1, y + 1)) > 0 &&
	//		*(f->depth->PointerAt<uint16_t>(x + 1, y + 1)) > 0 &&
	//		*(f->depth->PointerAt<uint16_t>(x - 1, y - 1)) > 0 &&
	//		*(f->depth->PointerAt<uint16_t>(x + 1, y - 1)) > 0) {
	//		return true;
	//	}
	//	else {
	//		return false;
	//	}
	//};
	
	//compact geometric features
	//vector<int> inds;
	//inds.reserve(f->lowpcd->points_.size() / 10);
	//for (int i = 0; i < f->lowpcd->points_.size(); i = i + 10) {
	//	inds.push_back(i);
	//}
	//vector<vector<double>> hists;
	//getSphericalHistogramms(f->lowpcd, hists, inds);



	cv::Mat orbDescriptors;
	std::vector<cv::KeyPoint> orbKeypoints;
	//orbKeypoints.reserve(500);
	auto kpd = cv::ORB::create(g_nkeypoints);
	kpd->detect(cvColor, orbKeypoints);
	kpd->compute(cvColor, orbKeypoints, orbDescriptors);
	//log(orbDescriptors.depth()) //use this to find out the datatype
	//auto intr = camera::PinholeCameraIntrinsic(camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault).intrinsic_matrix_;
	auto intr = g_intrinsic.intrinsic_matrix_;
	int nvalid = 0;
	std::vector<int> validentries(orbKeypoints.size());
	//check if depth is non zero
	for (int i = 0; i < orbKeypoints.size(); i++) {
		auto k = orbKeypoints[i];
		if (goodpixel(k.pt.x, k.pt.y,f)) {
			validentries[nvalid] = i;
			nvalid++;
		}
	}
	//keypoint_pcd->points_.reserve(nvalid);
	f->orbKeypoints.reserve(nvalid);
	f->orbDescriptors = cv::Mat::zeros(nvalid, 32, CV_8U);

	//auto fpfh_pcd = make_shared<geometry::PointCloud>();
	//fpfh_pcd->points_.reserve(nvalid);
	//fpfh_pcd->normals_.reserve(nvalid);



	for (int i = 0; i < nvalid; i++) {
		auto k = orbKeypoints[validentries[i]];
		//float x = round(k.pt.x);
		//float y = round(k.pt.y);
		float x = floor(k.pt.x);
		float y = floor(k.pt.y);
		auto d = f->rgbd->depth_.FloatValueAt(x, y).second;
		//double d = (double)*f->depth->PointerAt<uint16_t>(x,y)/1000.0;
		Eigen::Vector3d tmp;
		tmp[0] = (x - intr(0, 2)) * d / intr(0, 0);
		tmp[1] = (y - intr(1, 2)) * d / intr(1, 1);
		tmp[2] = d;
		//fpfh_pcd->points_.push_back(tmp);
		//fpfh_pcd->normals_.push_back(f->pcd->normals_[f->pcd->indeces(x, y)]);

		// fill keypoints
		c_keypoint ke;
		//Eigen::Vector3d tmp = f->pcd->points_[f->pcd->indeces(x,y)];
		ke.p = Eigen::Vector4d(tmp(0), tmp(1), tmp(2), 1);
		//todo use this when getting rid of rgbd images
		//ke.p = Eigen::Vector4d((x - intr(0, 2)) * d / intr(0, 0), (y - intr(1, 2)) * d / intr(1, 1), d, 1);
		ke.des.reserve(32);
		// build the filtered descriptor matrix
		for (int j = 0; j < 32; j++) {
			f->orbDescriptors.at<uchar>(i, j) = orbDescriptors.at<uchar>(validentries[i], j);
			ke.des.push_back(orbDescriptors.at<uchar>(validentries[i], j));
		}
		f->orbKeypoints.push_back(ke);
	}

	//compute fpfh features
	//auto fpfh_features = registration::ComputeFPFHFeature(*fpfh_pcd, open3d::geometry::KDTreeSearchParamHybrid(0.25, 100));

}
