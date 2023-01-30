#pragma once
#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <unordered_map> 
#include <string>


using namespace std;

//only makes sense for string keys 
template<typename key, typename val>
struct c_umap {
	unordered_map<key, val> x;
	val& operator()(int k1, int k2);
};

template<typename key, typename val>
inline val& c_umap<key, val>::operator()(int k1, int k2)
{
	if (k1 > k2) {
		int tmp = k1;
		k1 = k2;
		k2 = tmp;
	}
	return x[to_string(k1) + to_string(k2)];
}

//for rawmatches and matches (frame pairs). Indeces fit to keypoints variable, not validkeypoints
//p1 always refers to the frame with lower unique id.
struct match {
	double d;
	Eigen::Vector2i indeces;
	Eigen::Vector4d p1; //pcd A
	Eigen::Vector4d p2; //pcd B

	friend std::ostream& operator<<(std::ostream& os, const match& r);

};

//contains the kabschtrans for one constraint and the ideal transforms
struct pairTransform{
	bool set = false;
	Eigen::Matrix4d kabschtrans; //transform from pcd a to pcd b
	Eigen::Matrix4d invkabschtrans; //transform from pcd b to pcd a
	Eigen::Matrix4d Tk2A; //transform to the initial position of PA for ideal position
	Eigen::Matrix4d Tk2B; //transform to the initial position of PB for ideal position
	Eigen::Matrix4d midrot; //transform to the interpolated rotation position
	Eigen::Matrix4d invmidrot; //transform to the inverted interpolated rotation position
	//Eigen::Matrix<double, 6, 1> kabschdofs; //transform to the initial position of PA for ideal position
	//Eigen::Matrix<double, 6, 1> Tk2A; //transform to the initial position of PA for ideal position
	//Eigen::Matrix<double, 6, 1> Tk2B; //transform to the initial position of PB for ideal position
};

//for valid keypoints
struct rmatch {
	rmatch(int a, int b, int c, int d) : fi1(a), fi2(b), i1(c), i2(d) {};
	int fi1; //frameindex 1
	int fi2; //frameindex 2
	int i1; //keypoint index 1 
	int i2; //keypoint index 2

	friend std::ostream& operator<<(std::ostream& os, const rmatch& r);
};

//keypoint with descriptor
struct c_keypoint {
	Eigen::Vector4d p;
	vector<unsigned char> des;
	// overload equal operator
	bool operator==(const c_keypoint& k);
	void transform(Eigen::Matrix4d& m);
};


struct unique_id_counter {
	int id = 0;

	operator int() {
		id++;
		return id - 1;;
	}

};

