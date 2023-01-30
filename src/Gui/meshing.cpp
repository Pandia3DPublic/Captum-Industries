#include "meshing.h"
#include "../core/threadvars.h"
#include "../integrate.h"
#include "../configvars.h"

//generates the cuda mesh for visualization
//todo get subvolumes in frustum for vis frustum
void meshThreadFunction(Model* m) {
	std::chrono::time_point<std::chrono::steady_clock> meshing_time = std::chrono::high_resolution_clock::now();
	auto currentTime = std::chrono::high_resolution_clock::now();
	bool minIntervalpassed = true;
	
	while (m->tsdf_cuda.active_subvolume_entry_array_.size() == 0 && !stopMeshing) {
		std::this_thread::sleep_for(20ms); //sleep since no task is necessary
	}
	while (!stopMeshing) {
		if (m->tsdf_cuda.unmeshed_data && minIntervalpassed){ //gets set with time delay
			minIntervalpassed = false;							
			cuda::TransformCuda extrinsic;
			currentposlock.lock();
			extrinsic.FromEigen(m->currentPos);
			currentposlock.unlock();


			meshlock.lock();
			pandia_integration::tsdfLock.lock();
			//reset
			m->tsdf_cuda.active_subvolume_entry_array_.Clear(); //sets array iterator to 0
			m->tsdf_cuda.ResetActiveSubvolumeIndices(); // write zeros in device array

			if (!g_wholeMesh) {
				m->tsdf_cuda.GetSubvolumesInFrustum(virtualIntrinsic_cuda, extrinsic); 
			}
			m->mesher.MarchingCubes(m->tsdf_cuda,g_wholeMesh);
			meshlock.unlock();
			pandia_integration::tsdfLock.unlock();
			m->meshchanged = true;


			if(g_wholeMesh)
				std::this_thread::sleep_for(200ms); //if in pause mode gpu needs to integrate not render
		} else {// todo dont know if this or thread sleep
			std::this_thread::sleep_for(20ms); //sleep since no task is necessary
		}
		currentTime = std::chrono::high_resolution_clock::now();
		//if less than 30ms have passed, do not remesh
		if (std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - meshing_time).count() > 30) {
			meshing_time = currentTime;
			minIntervalpassed = true;
		}
	}


		utility::LogDebug("Stopped Meshing Thread \n");

}