#include "postprocessing.h"


namespace PandiaGui {

	std::atomic<bool> g_postProcessing(false); //sets gui block and pp indicator in gui
	bool stopPostProcessing = false; //cancel flag by button press

	//postprocessing variables
	double voxelSliderValue = 0;
	double meshSliderValue = 0;
	float overallPostProgress = 0.f;
	float currentProgress = 0.f;
	bool pp_denseAlign = false;
	bool pp_meshReduction = false;
	bool pp_voxelLength = false;
	bool postpro_disable = true;
	float voxelSlider = -1.f;
	float meshSlider = 100;
	float minSlider = 0.01f;
	float maxSlider = 0.05f;

	void setVoxelSliderValue() {
		voxelSlider = (float) g_voxel_length * 100;
	}
//todo move
void PostProcessingThread(Model& m) {




	//denseAlignment
	if (pp_denseAlign) {
		pp_denseAlign = false;
	}


	//voxel Length
	if (pp_voxelLength) {
		cuda::RGBDImageCuda c_rgbd = cuda::RGBDImageCuda(g_resx, g_resy, g_cutoff, 1000.0f);
		cuda::TransformCuda c_extrinsic = cuda::TransformCuda::Identity();
		//calculate the appropriate tsdf size here!
		//double factor = (double)m.tsdf_cuda.active_subvolume_entry_array_.size()/ 120000.0 *  (g_voxel_length / voxelSliderValue) * (g_voxel_length / voxelSliderValue)* (g_voxel_length / voxelSliderValue);
		//if (factor > 2) factor = 2;
		m.tsdf_cuda.Release();
		//m.mesher.~ScalableMeshVolumeCuda();
	//	new(&m.mesher) cuda::ScalableMeshVolumeCuda(cuda::VertexWithNormalAndColor, 8, factor * 120000);
		//cout << "Created new tsdf with " << factor * 120000 << " subvolumes \n";
		
		cuda::ScalableTSDFVolumeCuda tsdf_cuda(8, voxelSliderValue, 0.04, cuda::TransformCuda::Identity(), 7000,  120000);
		tsdf_cuda.device_->min_dist = g_mincutoff;
		tsdf_cuda.device_->max_dist = g_cutoff;

		//counts all frames as progress indicator
		for (int i = 0; i < m.chunks.size(); i++) {
			overallPostProgress += m.chunks[i]->frames.size();
		}

		for (int i = 0; i < m.chunks.size(); i++) {
			for (int j = 0; j < m.chunks[i]->frames.size(); j++) {
				auto& f = m.chunks[i]->frames[j];
				c_rgbd.UploadFloat(f->rgbd->depth_, f->rgbd->color_); //should not need look since images are never changed
				c_extrinsic.FromEigen(f->getFrametoWorldTrans());
				tsdf_cuda.Integrate(c_rgbd, g_intrinsic_cuda, c_extrinsic);
				tsdf_cuda.Integrate(c_rgbd, g_intrinsic_cuda, c_extrinsic);
				if (stopPostProcessing) {
					pp_meshReduction = false;
					g_postProcessing = false;
					return;
				}

				currentProgress++;
			}
		}
		m.mesher.MarchingCubes(tsdf_cuda, true);
		m.meshchanged = true;
		pp_voxelLength = false;
	}

	//meshReduction
	if (pp_meshReduction) {
		auto tmp = m.mesher.mesh().Download();

		int targetsize = tmp->triangles_.size() * meshSliderValue;
		tmp = tmp->SimplifyQuadricDecimation(targetsize);

		m.mesher.mesh().Upload(*tmp);
		m.meshchanged = true;
		pp_meshReduction = false;
	}

	//frees gui
	g_postProcessing = false;

}



shared_ptr<open3d::geometry::TriangleMesh> PostColorOptimization(Model& m) {
	
	std::vector<std::shared_ptr<geometry::RGBDImage>> rgbd_vector;
	int nchunks = m.chunks.size();
	open3d::camera::PinholeCameraTrajectory cameraTrajectory;
	//auto cameraTrajectory = std::make_shared<camera::PinholeCameraTrajectory>();
	auto tmp = m.mesher.mesh().Download();
	//m.integrateCPU();

	//gets all RGBD-Images from frames in chunks and pushes them in vector
	for (size_t i = 0; i < nchunks; i++) { //goes through every chunk

		auto& current_chunk = m.chunks[i];

		for (size_t j = 0; j < current_chunk->frames.size(); j++) { //goes through every frame in current chunk

			auto& current_frame = current_chunk->frames[j];
			rgbd_vector.push_back(current_frame->rgbd);

			//gets cameraParameters and sets them in CameraTrajectory
			camera::PinholeCameraParameters tmp_parameters;
			tmp_parameters.intrinsic_ = g_intrinsic;
			//set extrinsic to Matrix4d_u
			tmp_parameters.extrinsic_ = current_frame->getFrametoWorldTrans().inverse();
			
			cameraTrajectory.parameters_.push_back(tmp_parameters);

		}
	}
	

	//auto trimesh = m.tsdf->ExtractTriangleMesh();

	cout << tmp->triangles_.size() << endl;
	cout << tmp->vertices_.size() << endl;

	color_map::ColorMapOptimizationOption option;
	option.maximum_iteration_ = 100;
	option.non_rigid_camera_coordinate_ = true;
	color_map::ColorMapOptimization(*tmp, rgbd_vector, cameraTrajectory, option);


	return tmp;

}
}
