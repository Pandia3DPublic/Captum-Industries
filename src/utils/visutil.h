#pragma once

#include <Eigen/Dense>
#include <Open3D/Open3D.h>
#include "core/Frame.h"
#include "core/Chunk.h"
#include "core/Model.h"
#include "imageutil.h"
#include <string>
#include "Gui/geometry.h"


std::shared_ptr<geometry::TriangleMesh> getCameraPathMesh(Model& m);
shared_ptr<geometry::TriangleMesh> createPathMesh(Model &m);


void visThreadFunction(Model* m, std::atomic<bool>& stopVisualizing); //deprecated only for debug now

std::shared_ptr<geometry::TriangleMesh> getSpherePointer(Eigen::Vector3d v);
void visualizeChunk(Chunk& c, bool worldcoords = false);
void visualizetsdfModel(Model& m);
//other data formats
void drawEigen(std::vector<Eigen::Matrix3Xd> vvec); //draws an eigen matrix containing points
void drawPointVector(std::vector<Eigen::Vector3d> v); //draws a vector containing eigen 3d poitns
void visualizeHomogeniousVector(vector<Eigen::Vector4d>& v);
//rest
void visualizecustomskeypoints(std::vector<c_keypoint>& kps);
//draws all pcds with keypoints and lines between them
void visualizecurrentMatches(shared_ptr<Chunk> c);
//draws all pcds with keypoints and lines between them
void visualizecurrentMatches(Model& m, bool raw = false);
std::shared_ptr<geometry::LineSet> getCamera(Eigen::Matrix4d& t, Eigen::Vector3d color = Eigen::Vector3d(0.0,0.0,1.0));
std::shared_ptr<geometry::LineSet> getCameraPath(std::vector<Eigen::Matrix4d>& v, Eigen::Vector3d color = Eigen::Vector3d(0.0,0.0,1.0)); 

std::shared_ptr<geometry::LineSet> getOrigin();
std::shared_ptr<geometry::LineSet> getvisFrusti(Model& m, Frustum& corefrustum);
std::shared_ptr<geometry::LineSet> getFrustumLineSet(Frustum& fr,const Eigen::Vector3d& color);

std::shared_ptr<geometry::TriangleMesh> getCameraPathMesh(Model& m, int& divider);
shared_ptr<geometry::TriangleMesh> createCameraScaffolding(shared_ptr<GLGeometry>& cameraPath, int& divider);
