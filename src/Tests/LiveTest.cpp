#include "LiveTest.h"
#include <chrono>

double diff2(shared_ptr<Frame> a) {
	Eigen::Vector6d diff;
	Eigen::Vector6d tmp = MattoDof(a->getFrametoWorldTrans());
	diff.block<3, 1>(0, 0) = tmp.block<3, 1>(0, 0) - a->integrateddofs.block<3, 1>(0, 0);
	diff.block<3, 1>(3, 0) = 2 * (tmp.block<3, 1>(3, 0) - a->integrateddofs.block<3, 1>(3, 0));
	return diff.norm();
}

bool liveTestwithVis() {

	readconfig("config.txt");
	Model m;
	reconrun(m,false,true); //model, test, livevis, integration
	auto& livetsdf = m.tsdf_cuda;

	cout << "If frames ids or integrated and world dofs are unequal the corresponding warning will come here \n";

	for (int i = 0; i < m.chunks.size(); i++) {
		for (int j = 0; j < m.chunks[i]->frames.size(); j++) {
			if(!m.chunks[i]->frames[j]->duplicate){
				bool found = false;
				for (auto& f : pandia_integration::integratedframes) {
					if (f == m.chunks[i]->frames[j]) {
						found =true;
					}
				}
				if (!found) {
					cout << "Id " << m.chunks[i]->frames[j]->unique_id << " in offline not existing in online \n";
					cout << "frame is frame number " << j << " in chunk number " << i << endl;
			}
			}
		}
	}

	for (auto& f : pandia_integration::integratedframes) {
		bool found = false;
		for (int i = 0; i < m.chunks.size(); i++) {
			for (int j = 0; j < m.chunks[i]->frames.size(); j++) {
				if (f == m.chunks[i]->frames[j]) {
					found =true;
				}
			}
		}
		if (!found) {
			cout << "Id " << f->unique_id  << "in online not existing in offline \n";
		}
	}
	for (auto& f : pandia_integration::integratedframes) {
		if (diff2(f) > 0.02) {
			cout << "difference between integrated dofs and world dofs to large! \n";
		}
	}

	cout << "end warning \n";

	//integrate offline
	float voxel_length = 0.01f;
	cuda::TransformCuda extrinsic = cuda::TransformCuda::Identity();
	cuda::ScalableTSDFVolumeCuda offlinetsdf(8, voxel_length, 3 * voxel_length, extrinsic);

	cuda::RGBDImageCuda rgbd = cuda::RGBDImageCuda(g_resx, g_resy, g_cutoff, 1000.0f);
	int count =0;
	for (int i = 0; i < m.chunks.size(); i++) {
		for (int j = 0; j < m.chunks[i]->frames.size(); j++) {
			auto& f = m.chunks[i]->frames[j];
			if (!f->duplicate){
				count++;
				rgbd.UploadFloat(f->rgbd->depth_, f->rgbd->color_);
				extrinsic.FromEigen(f->getFrametoWorldTrans());
				offlinetsdf.Integrate(rgbd, g_intrinsic_cuda, extrinsic);
			}
		}
	}
	cout << "number of offline frames " << count << endl;
	cout << "number of online frames " << pandia_integration::integratedframes.size() << endl;

	//need two mesher for some reasons
	cuda::ScalableMeshVolumeCuda mesher(cuda::VertexWithNormalAndColor, 8, 120000);
	cuda::ScalableMeshVolumeCuda mesher2(cuda::VertexWithNormalAndColor, 8, 120000);
	std::shared_ptr<cuda::TriangleMeshCuda> livemesh = std::make_shared<cuda::TriangleMeshCuda>();
	std::shared_ptr<cuda::TriangleMeshCuda> offlinemesh = std::make_shared<cuda::TriangleMeshCuda>();
	std::shared_ptr<cuda::TriangleMeshCuda> rendermesh = std::make_shared<cuda::TriangleMeshCuda>();
	
	mesher2.MarchingCubes(offlinetsdf,true);
	*offlinemesh = mesher2.mesh();
	mesher.MarchingCubes(livetsdf,true);
	*livemesh = mesher.mesh();


	visualization::VisualizerWithCudaModule visualizer;
	if (!visualizer.CreateVisualizerWindow("Live TSDF", 640,480, 250, 250)) {
		utility::LogWarning("Failed creating OpenGL window.\n");
	}


	visualizer.BuildUtilities();
	visualizer.UpdateWindowTitle();
	*rendermesh = *livemesh;
	visualizer.AddGeometry(rendermesh);
	visualizer.AddGeometry(getOrigin());
	std::cout << "live mesh now \n";
	auto start = std::chrono::high_resolution_clock::now();
	auto end = std::chrono::high_resolution_clock::now();
	visualizer.setWindowName("Live Mesh");
	while (true){
		start = std::chrono::high_resolution_clock::now();
		while (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() < 1000) {
			visualizer.PollEvents(); //this takes long (5-20ms) probably blocks a lot!
			visualizer.UpdateGeometry();
			end = std::chrono::high_resolution_clock::now();
		}


		std::cout << "offline mesh now \n";
		*rendermesh = *offlinemesh;
		visualizer.UpdateRender();
		visualizer.setWindowName("Offline Mesh");

		std::cout << "offline mesh now  success\n";
		start = std::chrono::high_resolution_clock::now();
		while (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() < 1000) {
			visualizer.PollEvents(); //this takes long (5-20ms) probably blocks a lot!
			visualizer.UpdateGeometry();
			end = std::chrono::high_resolution_clock::now();

		}
		std::cout << "live mesh now \n";
		*rendermesh = *livemesh;
		visualizer.UpdateRender();
		visualizer.setWindowName("Live Mesh");

		std::cout << "live mesh now  success\n";

	}


	return false;	
}


//bool liveTest() {
//
//	readconfig("config.txt");
//	Model m;
//	reconrun(m,g_test,false,true); //model, test, livevis, integration
//	auto& livetsdf = m.tsdf_cuda;
//
//
//
//	//integrate offline
//	float voxel_length = 0.01f;
//	cuda::TransformCuda extrinsic = cuda::TransformCuda::Identity();
//	cuda::ScalableTSDFVolumeCuda offlinetsdf(8, voxel_length, 3 * voxel_length, extrinsic);
//
//	cuda::RGBDImageCuda rgbd(640, 480, g_cutoff, 1000.0f);
//	for (int i = 0; i < m.chunks.size(); i++) {
//		for (int j = 0; j < m.chunks[i]->frames.size(); j++) {
//			auto& f = m.chunks[i]->frames[j];
//			rgbd.Upload(*f->depth, *f->rgb);
//			extrinsic.FromEigen(f->getFrametoWorldTrans());
//			offlinetsdf.Integrate(rgbd, g_intrinsic_cuda, extrinsic);
//		}
//	}
//
//
//	cuda::ScalableMeshVolumeCuda mesher(cuda::VertexWithNormalAndColor, 8, 120000);
//	std::shared_ptr<cuda::TriangleMeshCuda> livemesh = std::make_shared<cuda::TriangleMeshCuda>();
//	std::shared_ptr<cuda::TriangleMeshCuda> offlinemesh = std::make_shared<cuda::TriangleMeshCuda>();
//	
//	mesher.MarchingCubes(livetsdf,true);
//	*livemesh = mesher.mesh();
//
//	mesher.MarchingCubes(offlinetsdf,true);
//	*offlinemesh = mesher.mesh();
//
//	if (livemesh == offlinemesh) {
//		return true;
//	}
//
//
//	return false;	
//}