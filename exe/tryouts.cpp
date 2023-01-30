#include <iostream>
#include <stdio.h>
#include "core/reconrun.h"
#include <list>
#include <GL/glew.h>     
#include <GLFW/glfw3.h>
#include "Gui/imgui-docking/imgui.h"
#include "Gui/imgui-docking/imgui_impl_glfw.h"
#include "Gui/imgui-docking/imgui_impl_opengl3.h"
#include "Gui/imgui-docking/FileBrowser/dirent.h"
#include "Gui/imgui-docking/FileBrowser/ImGuiFileBrowser.h"
//#include "cmakedefines.h"
//#include "Gui/geometry.h"
//#include "Gui/PandiaView.h"
//#include "frustum.h"
//#include "readconfig.h"
//#include "configvars.h"


using namespace std;
//using namespace open3d;


int main() {
	
    float a = 0.282977;
    float b = 0.221915;
    float c = 0.151337;
    unsigned char* temp = NULL;
    
    temp = (unsigned char*)&(a);
    auto adress = &a;
    cout << "Adress: " << adress << endl;
    const float* ptr = reinterpret_cast<const float*>(temp);
    cout << "ptr: " << *ptr << endl;




}


//if (streamptr->is_open()) {
//
//	string line;
//	while (getline(*streamptr, line)) {
//
//		auto itr = line.begin();
//		advance(itr, line.find_first_of("\""));
//		line.erase(std::remove_if(line.begin(), itr, ::isspace), itr);
//
//		if (line[0] == '#' || line.empty())
//			continue;
//		auto delimiterPos = line.find("=");
//		auto name = line.substr(0, delimiterPos);
//		auto value = line.substr(delimiterPos + 1);
//		cout << name << "-" << value << "|" << endl;
//	}
//	cout << ".txt successfully read" << endl;
//}
//else {
//	cout << "no .txt found" << endl;
//}


//struct A {
//	A(){cout << "constructor called \n";};
//
//	~A(){cout << "destroyed A \n";};
//};
//
//int main() {
//
//	//get current folder with trajectory data
//	readconfig("config.txt");
//	//g_readimagePath = "C:/Users/Tim/Documents/Scenes/scene0000_00";
//	Model m;
//	g_take_dataCam = true;
//	g_camType = camtyp::typ_data;
//	initialiseCamera();
//	//auto f = getSingleFrame(g_readimagePath, 0);
//	//auto f2 = getSingleFrame(g_readimagePath, 100);
//	reconrun(m, false, true);
//	cuda::ImageCuda<float, 3> raycasted(640, 480);
//	cuda::ImageCuda<float, 3> raycasted2(640, 480);
//	m.tsdf_cuda.transform_volume_to_world_.ToEigen();
//	cuda::TransformCuda cudapos = cuda::TransformCuda::Identity();
//	cudapos.FromEigen(m.currentPos);
//
//
//	m.tsdf_cuda.RayCasting(raycasted, g_intrinsic_cuda, cudapos);
//	auto img = raycasted.DownloadImage();
//
//
//	for (int i = 0; i < 1; i++) {
//		Timer t;
//		m.tsdf_cuda.RayCasting2(raycasted2, g_intrinsic_cuda, cudapos);
//		//m.tsdf_cuda.RayCasting_custom(raycasted, g_intrinsic_cuda, cudapos);
//	}
//	auto img2=  raycasted2.DownloadImage();
//
//
//	visualization::DrawGeometries({img});
//	visualization::DrawGeometries({ img2 });
//
//	//m.integrateCPU();
//	//visualization::DrawGeometries({ m.tsdf->ExtractTriangleMesh() });
//}
//	//auto rgbd_source =f->rgbd;
//	//auto rgbd_target = f2->rgbd;
//
//	//std::shared_ptr<geometry::PointCloud> source_origin, target_origin;
//	//source_origin= f->pcd;
//	//source_origin->EstimateNormals();
//
//	//target_origin=  f2->pcd;
//	//target_origin->EstimateNormals();
//
//	////auto source_down = f->pcd->VoxelDownSample(0.02);
//	////auto target_down = f2->pcd->VoxelDownSample(0.02);
//	//int* a;
//	//cudaMalloc(&a, sizeof(int));
//
//	//auto source_down = f->lowpcd;
//	//auto target_down = f2->lowpcd;
//	///** Load data **/
//	//cuda::RegistrationCuda registration(registration::TransformationEstimationType::PointToPlane);
//	//Timer t2("init time");
//	//registration.Initialize(*source_down, *target_down, 0.05f);
//	//t2.~Timer();
//	///** Prepare visualizer **/
//	//visualization::VisualizerWithCudaModule visualizer;
//	//if (!visualizer.CreateVisualizerWindow("ColoredICP", 640, 480, 0, 0)) {
//	//	utility::LogWarning("Failed creating OpenGL window.\n");
//	//	return -1;
//	//}
//	//visualizer.BuildUtilities();
//	//visualizer.UpdateWindowTitle();
//	//visualizer.AddGeometry(source_down);
//	//visualizer.AddGeometry(target_down);
//
//	//bool finished = false;
//	//int iter = 0, max_iter = 50;
//	//visualizer.RegisterKeyCallback(GLFW_KEY_SPACE, [&](visualization::Visualizer *vis) {
//	//	if (finished) return false;
//
//	//	/* Registration (1 iteration) */
//	//	Timer t("single iteration");
//	//	auto delta = registration.DoSingleIteration(iter++);
//	//	t.~Timer();
//	//	/* Updated source */
//	//	source_down->Transform(delta.transformation_);
//	//	vis->UpdateGeometry();
//
//	//	/* Update flags */
//	//	if (iter >= max_iter)
//	//		finished = true;
//	//	return !finished;
//	//	});
//
//	//bool should_close = false;
//	//while (!should_close) {
//	//	should_close = !visualizer.PollEvents();
//	//}
//	//visualizer.DestroyVisualizerWindow();
//
