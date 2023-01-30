#pragma once


#include "Frame.h"
#include "filters/kabsch.h"
#include <Open3D/Open3D.h>
#include <Open3D/Geometry/KDTreeFlann.h>
#include <Open3D/Registration/Feature.h>
#include <Eigen/Dense>
#include <iostream>
#include <limits>
#include <algorithm>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "ceres/ceres.h"
#include "glog/logging.h"
#include "match.h"
#include "Optimizable.h"
#include "frustum.h"
//#include "nllsFunctors.h"

//#define printstuf
//#ifdef printstuf
//#define log(x) std::cout<<x<<std::endl;
//#else
//#define log(x)
//#endif


using namespace open3d;




//bool matchsort(match &i, match &j);

class Chunk : public Optimizable, public KeypointUnit{
public:

	Chunk();
	~Chunk();
	//base variables
	std::vector<std::shared_ptr<Frame>> frames;

	//std::vector<c_keypoint> keypoints;
	int ncorframes; // number edges between frames
	bool chunktransapplied = false;
	int modelindex = -1; //indicates which place in the model this chunk takes
	shared_ptr<Frustum> frustum; // only gets set after chunk is complete

	//methods
	//void performSparseOptimization();
	void performSparseOptimization(vector<Eigen::Vector6d>& initx);
	void generateChunkKeypoints(int it = 1); //keypoints used for global alignment, fuse same keypoints
	void generateEfficientStructures(); //keypoints used for sparse optimization
	void deleteStuff(); //delete everything that is not needed when chunk is finished
	bool Chunk::doFrametoModelforNewestFrame();
	void generateFrustum();

	//for visualization
	//variables
	std::shared_ptr<geometry::LineSet> ls = std::make_shared<geometry::LineSet>();
	std::vector<std::shared_ptr<geometry::TriangleMesh>> spheres;
	//methods
	void addLine(Eigen::Vector3d startpoint, Eigen::Vector3d endpoint);
	void addValidLines();
	void markChunkKeypoints();

	//debugging
	bool output = true;



private:
	vector<int> chunkkpweights; // used for keypoint merger weighting
	void Chunk::getMajorityDescriptor(vector<int>& indeces,const vector<c_keypoint>& kps, vector<uchar>& des);
};

