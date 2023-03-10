#pragma once
#include <Eigen/Geometry>
#include <vector>
#include <Open3D/Camera/PinholeCameraIntrinsic.h>
#include <iostream>

class Plane {

public:
	Plane(const Eigen::Vector3d& n,const Eigen::Vector3d& point);

	Eigen::Vector3d normal;
	double d; //distance of normal vector to origin

	void transform(const Eigen::Matrix4d& trans);

};

class Frustum {
public:

	Frustum(const Eigen::Matrix4d& camerapos,const open3d::camera::PinholeCameraIntrinsic& intrinsic, const double& dmin, const double& dmax);
	~Frustum(){};


	bool intersect(const Frustum& other) const;
	bool Frustum::inLocalGroup(const Frustum& f2);


	std::vector<Plane> planes;
	std::vector<Eigen::Vector3d> corners; //starting topleft, going cc. first near than far
	Eigen::Matrix4d pos;


	Eigen::Vector3d cordtoPoint(const int& x,const int& y,const double& d, const Eigen::Matrix3d& intr);
	void reposition(const Eigen::Matrix4d& newpos);

};