#pragma once
#include "core/Model.h"
#include <vector>
#include "solverWrapper.h"


void gpuSolverThread(Model* m, solverWrapper* solver, vector<Eigen::Vector6d>* dofs);
