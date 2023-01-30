#pragma once

#include "Chunk.h"
#include "semantics/LabeledClasses.h"
#include <Cuda/Open3DCuda.h>


class Model: public Optimizable
{
public:
	Model();
	~Model();

	std::vector<std::shared_ptr<Chunk>> chunks;
	std::shared_ptr<LabeledScalTSDFVolume> tsdf;
	cuda::ScalableTSDFVolumeCuda tsdf_cuda;
	cuda::RGBDImageCuda rgbd;
	vector<shared_ptr<Chunk>> invalidChunks;
	cuda::ScalableMeshVolumeCuda mesher;
	std::atomic<bool> meshchanged = false;
	Eigen::Matrix4d currentPos = Eigen::Matrix4d::Identity(); //current camera position. Only use in integrationlock
	list <shared_ptr<Frame>> recordbuffer; //never used if RECORDBUFFER is not defined


	void Model::generateEfficientStructures();
	void Model::performSparseOptimization(vector<Eigen::Vector6d>& initx);
	bool Model::doChunktoModelforNewestChunk();
	void Model::performSparseOptimization2();

	void Model::applyModelTransform();
	void Model::setWorldTransforms();

	//vector<shared_ptr<Frame>> frames; //contains all frames of the model

	void integrateCPU(); 
	void saveCPUMesh(string name);
	double getCost(); //returns the cost of chunk optimization
	double Model::getCost2(); //for debug of gpu solver


	void Model::drawKeypoints(int start= 0, int end = 0) ;
private:
	bool allChunkshavePos();
	bool getXor(int& a, int& b);
	void addToCoeffs(Eigen::VectorXd& v,const Eigen::Vector3d& summand,const int& pos);
	void drawKeypoints(const int& a,const int& b, Eigen::Matrix4d Ta, Eigen::Matrix4d Tb);
	void getResiduals(int& a, int& b, vector<double>& res);
	void getMaxResidual(double& max, int&ares, int&bres);
	std::pair<Eigen::Matrix4d, Eigen::Matrix4d> Model::getOptTransforms(int& a, int& b);
	std::pair<Eigen::Matrix4d, Eigen::Matrix4d> Model::getOptDofTransforms(int& a, int& b);
	void performOptIteration();






};

