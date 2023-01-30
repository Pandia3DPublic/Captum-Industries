//#include "core/reconrun.h"
//
//#include "solverWrapper.h"
//#include <iostream>
//#include <Open3D/Open3D.h>
//#include "configvars.h"
//#include "utils/coreutil.h"
//#include "ptimer.h"
//#include <Cuda/Open3DCuda.h>
//#include "utils/visutil.h"
//#include <GL/glew.h>
//#include <GLFW/glfw3.h>
//#include "Gui/Shader.h"
//#include "imgui-docking/imgui.h"
//#include "imgui-docking/imgui_impl_glfw.h"
//#include "imgui-docking/imgui_impl_opengl3.h"
//#include "cmakedefines.h"
//#include "Gui/geometry.h"
//#include "Gui/PandiaView.h"
//#include "frustum.h"
//
//using namespace std;
//using namespace open3d;
////
//
//
//std::shared_ptr<geometry::LineSet> getFrustumLineSet(Frustum& fr,const Eigen::Vector3d& color) {
//	auto ls = std::make_shared<geometry::LineSet>();
//
//	for (auto& p : fr.corners) {
//		ls->points_.push_back(p);
//	}
//
//	//cc again starting with back half
//	ls->lines_.push_back(Eigen::Vector2i(0, 1));
//	ls->lines_.push_back(Eigen::Vector2i(1, 2));
//	ls->lines_.push_back(Eigen::Vector2i(2, 3));
//	ls->lines_.push_back(Eigen::Vector2i(3, 0));
//
//	//sides
//	ls->lines_.push_back(Eigen::Vector2i(0, 4));
//	ls->lines_.push_back(Eigen::Vector2i(1, 5));
//	ls->lines_.push_back(Eigen::Vector2i(2, 6));
//	ls->lines_.push_back(Eigen::Vector2i(3, 7));
//
//	//front 
//	ls->lines_.push_back(Eigen::Vector2i(4, 5));
//	ls->lines_.push_back(Eigen::Vector2i(5, 6));
//	ls->lines_.push_back(Eigen::Vector2i(6, 7));
//	ls->lines_.push_back(Eigen::Vector2i(7, 4));
//
//
//	for (int i = 0; i < 12; i++) {
//		ls->colors_.push_back(color);
//	}
//
//
//	return ls;
//}
//
//
//std::shared_ptr<geometry::LineSet> getvisFrusti(vector<Frustum>& frusti) {
//	auto ls = std::make_shared<geometry::LineSet>();
//	auto& corefrustum = frusti.front();
//	*ls+= *getFrustumLineSet(corefrustum,Eigen::Vector3d(0,0,1));//blue
//
//	for (int i = 1; i < frusti.size(); i++) {
//		//if (corefrustum.intersect(frusti[i])) {
//		if (corefrustum.inLocalGroup(frusti[i])) {
//			*ls+= *getFrustumLineSet(frusti[i],Eigen::Vector3d(0,1,0)); //green
//		} else {
//			*ls+= *getFrustumLineSet(frusti[i],Eigen::Vector3d(1,0,0)); //red
//		}
//
//	}
//
//	return ls;
//}
//
//
//
//int main (){
//
//	readconfig("config.txt");
//	cout << g_mincutoff << endl;
//	Model m;
//	threadCameraInit();
//	reconrun(std::ref(m), g_fromData,false ,false);//model, test, livevis, integration
//
//
//	vector<Frustum> frusti;
//	vector<Eigen::Matrix4d> cameraposes; 
//
//
//	for (int i = 0; i < m.chunks.size(); i++) {
//		//for (int j = 0; j < m.chunks[i]->frames.size(); j++) {
//		Frustum tmp(m.chunks[i]->frames[0]->getFrametoWorldTrans(),camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault, 0.25,5.0);
//		frusti.push_back(tmp);
//		cameraposes.push_back(m.chunks[i]->frames[0]->getFrametoWorldTrans());
//		//}
//	}
//
//	//Eigen::Matrix4d pos = Eigen::Matrix4d::Identity();
//	//Eigen::Matrix4d pos2 = Eigen::Matrix4d::Identity();
//	//pos2(0,3) = 5;
//	//Frustum fr1(pos,camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault, 0.1,5.0);
//	//Frustum fr2(pos2,camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault, 0.1,5.0);
//	//frusti.push_back(fr1);
//	//frusti.push_back(fr2);
//	//cout << fr1.intersect(fr2) << endl;
//
//	//visualization::DrawGeometries({getvisFrusti(frusti), getOrigin()});
//	m.integrateCPU();
//	auto tmp = m.tsdf->ExtractTriangleMesh();
//	//visualization::DrawGeometries({tmp,getCameraPath(cameraposes),getOrigin()});
//	visualization::DrawGeometries({tmp,getCameraPath(cameraposes), getvisFrusti(frusti),getOrigin()});
//
//
//	//shared_ptr<geometry::TriangleMesh> mesh = make_shared<geometry::TriangleMesh>();
//	//io::ReadTriangleMesh("Testmesh.ply", *mesh);
//
//	//visualization::DrawGeometries({mesh});
//
//
//	////mesh = mesh->SimplifyQuadricDecimation(5e4);
//	//mesh = mesh->SimplifyVertexClustering(0.03);
//
//	//visualization::DrawGeometries({mesh});
//
//}
