#pragma once
#include <Open3D/Open3D.h>
#include "core/Frame.h"
#include "core/Chunk.h"
#include "core/Model.h"
#include "frustum.h"
#include "GlobalDefines.h"


void prepareDatapath(std::string& s);
camera::PinholeCameraIntrinsic getLowIntr(camera::PinholeCameraIntrinsic intrinsic);
camera::PinholeCameraIntrinsic getScaledIntr(camera::PinholeCameraIntrinsic intrinsic, int width, int height);
//own data from file, not production relevant
std::shared_ptr<Frame> getSingleFrame(std::string path, int nstart);
bool checkValid(Model& c);
bool checkValid(const shared_ptr<Chunk> c, vector<int>& removeIndices);
bool getBit(const unsigned char& a, const int& n);
void setBitOne(unsigned char& a, const int& n);

//get the initial 6 dofs for optimization for a intrachunk
vector<Eigen::Vector6d> getDoffromKabschChunk(const shared_ptr<Chunk> c);
//get the initial 6 dofs for optimization for model opt
vector<Eigen::Vector6d> getInitialDofs(Model& m);
Eigen::Vector6d MattoDof(const Eigen::Matrix4d& R);
void generateFrame(shared_ptr<open3d::geometry::Image> color, shared_ptr<open3d::geometry::Image> depth, shared_ptr<Frame> out);
std::shared_ptr<Frame> getSingleFrame(list <shared_ptr<Frame>>& framebuffer, list <shared_ptr<Frame>>& recordbuffer);
shared_ptr < geometry::PointCloud> genpcd(shared_ptr<open3d::geometry::Image> color, shared_ptr<open3d::geometry::Image> depth);

void setDefaultIntrinsic();
int getdircount(string path);
std::string getPicNumberString(int a);
void setFromIntrinsicFile(string& filepath);
bool fileExists(string& filename);

#ifdef RECORDBUFFER
void saveImagestoDisc(string& path, list <shared_ptr<Frame>>& recordbuffer);
#endif

#ifdef SAVETRAJECTORY
void saveTrajectorytoDisk(const string& path, Model& m, string name = "trajectory.txt");
#endif
void readTrajectory(const string& path, vector<Eigen::Matrix4d>& poses);
