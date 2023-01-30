#include "solverthread.h"
#include "core/threadvars.h"



//to perform timely global solve operations while live vis continious
void gpuSolverThread(Model* m, solverWrapper* solver, vector<Eigen::Vector6d>* dofs) {
	//// is set in startGlobalsolve so that context is okay.
	while (!g_current_slam_finished){
		if (g_solverRunning == true) {
			//copy init vars
			solverlock.lock();
			vector<Eigen::Vector6d> localDofs = *dofs;
			solverlock.unlock();
			//solve while main thread continous
			solver->solve(localDofs);
			g_solverRunning = false;
		} else {
			this_thread::sleep_for(20ms);
		}
	}

}