#include "DataCam.h"
#include <thread>
#include <mutex>
#include "configvars.h"
#include "utils/coreutil.h"
#include <chrono>
#include "ptimer.h"

void DataCamThreadFunction(list <shared_ptr<Frame>>& framebuffer, std::atomic<bool>& stop) {
	int i= g_nstart;
	while (!stop) {
		auto start = std::chrono::high_resolution_clock::now();

		while (g_pause) {
			this_thread::sleep_for(20ms);
		}
		framebuffer.push_back(getSingleFrame(g_readimagePath, i));
		i++;
		//no sleep necessary since hardrive access takes forever (50ms)
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		if (duration.count() < 33) {
			this_thread::sleep_for(10ms); //todo super stupid
		}

	}
}