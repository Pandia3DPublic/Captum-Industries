#pragma once
#include "core/Model.h"
#include "SolverState.h" //includes entryJ
#include "CUDASolver.h"
#include <atomic>

class solverWrapper {


public: 
	solverWrapper(Model* m_ex);
	~solverWrapper();

	//functions
	void solve(vector<Eigen::Vector6d>& dofs);
	EntryJ* constructGPUCors();

	//vars
	Model* m;
	EntryJ* d_cors;
	float3* d_rotdofs;
	float3* d_transdofs;
	shared_ptr<CUDASolver> solver;
	unsigned int maxNFrames = 5e2;
	unsigned int maxNRes = 2.5e4;

	int nmatches; //current number of matches in model
	int nChunks; //current number of chunks in model
	vector<float> x; //current solution



};