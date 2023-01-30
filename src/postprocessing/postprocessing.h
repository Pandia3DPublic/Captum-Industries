#pragma once
#include "core/Model.h"
#include "Open3D/Open3D.h"
#include "configvars.h"


namespace PandiaGui {
	extern std::atomic<bool> g_postProcessing; //sets gui block and pp indicator in gui
	extern bool stopPostProcessing; //cancel flag by button press

	//postprocessing variables
	extern double voxelSliderValue;
	extern double meshSliderValue;
	extern float overallPostProgress;
	extern float currentProgress;
	extern bool pp_denseAlign;
	extern bool pp_meshReduction;
	extern bool pp_voxelLength;
	extern bool postpro_disable;
	extern 	float voxelSlider;
	extern float meshSlider;
	extern 	float minSlider;
	extern 	float maxSlider;

	shared_ptr<open3d::geometry::TriangleMesh> PostColorOptimization(Model& m);

	//postprocessing thread
	void PostProcessingThread(Model& m);
	void setVoxelSliderValue();

}