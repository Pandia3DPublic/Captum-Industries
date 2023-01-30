#include "core/threadvars.h"
#include "solverWrapper.h"
#include <Cuda/Common/UtilsCuda.h>// for open3d cuda error check
#include "utils/matrixutil.h"
#include "ptimer.h"



solverWrapper::solverWrapper(Model* m_ex) {
	//init here
	m = m_ex;
	solver = make_shared<CUDASolver>(maxNFrames, maxNRes);
	CheckCuda(cudaMalloc(&d_rotdofs, sizeof(float3)*maxNFrames));
	CheckCuda(cudaMalloc(&d_transdofs, sizeof(float3)*maxNFrames));
	CheckCuda(cudaMalloc(&d_cors, sizeof(EntryJ) *maxNRes));


}

solverWrapper::~solverWrapper() {
	//free memory
	CheckCuda(cudaFree(d_rotdofs));
	CheckCuda(cudaFree(d_transdofs));
	CheckCuda(cudaFree(d_cors));

}

// this must be callled from startGlobalsolve and thread for context to be right
void solverWrapper::solve(vector<Eigen::Vector6d>& dofs) {
	//call gpu solver
	if (nmatches  > maxNRes) {
		utility::LogError("more residuals than hardcoded gpu solver can handle! Abort \n");
		return;
	}
	unsigned int nNonLinearMax = 25;
	unsigned int nLinearMax = 15;
	std::vector<float> sparseWeights(nNonLinearMax,1.0); //seems to be sparse weight per nonlinear it, not per cor

	vector<float> init_rotdofs;
	vector<float> init_transdofs;

	if (dofs.size() != nChunks) {
		cout << "error dofs size msut be be chunks size in solverwrapper solve \n";
	}

	for (int i = 0; i < nChunks; i++) { //dofs size must be equal to nchunks
		init_rotdofs.push_back(dofs[i](0));
		init_rotdofs.push_back(dofs[i](1));
		init_rotdofs.push_back(dofs[i](2));

		init_transdofs.push_back(dofs[i](3));
		init_transdofs.push_back(dofs[i](4));
		init_transdofs.push_back(dofs[i](5));
	}

	CheckCuda(cudaMemcpy(d_rotdofs, init_rotdofs.data(), sizeof(float3)* nChunks, cudaMemcpyHostToDevice));
	CheckCuda(cudaMemcpy(d_transdofs, init_transdofs.data(), sizeof(float3)* nChunks, cudaMemcpyHostToDevice));

	//CheckCuda(cudaMemset(d_rotdofs, 0, sizeof(float3)*nChunks)); //nChunks is correct since solver gives identiy for first dofs
	//CheckCuda(cudaMemset(d_transdofs, 0, sizeof(float3)*nChunks));
	unsigned int revalIdx = 0;
	
	auto start =std::chrono::high_resolution_clock::now();
	solver->solve(d_cors, (unsigned int)nmatches,(unsigned int)nChunks, nNonLinearMax, nLinearMax,sparseWeights,d_rotdofs,d_transdofs,true,true, revalIdx); //rebuildJT, findmaxRes, no idea what revalIDx is, is used in print and guided remove
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
	auto end  =std::chrono::high_resolution_clock::now();
	if (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() > 2000) {
		utility::LogWarning("GPU Solver takes longer than 2 Seconds! might corrupt results \n");
	}
	cout << "############ Time in solver: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << endl;
	//solver->evaluateTimings();

	//transfer results
	//copy back to cpu
	float* rotdofs = new  float[3*nChunks]; 
	float* transdofs = new  float[3*nChunks]; 
	CheckCuda(cudaMemcpy(rotdofs, d_rotdofs, sizeof(float3)*nChunks, cudaMemcpyDeviceToHost));
	CheckCuda(cudaMemcpy(transdofs, d_transdofs, sizeof(float3)*nChunks, cudaMemcpyDeviceToHost));

	x.clear();
	x.resize(6*nChunks);
	//float* x = new float[6*nChunks]; //vector with free variables on heap
	//todo test vector variant
	for (int i = 0; i < nChunks; i++) {
		x.data()[6*i + 0] =  rotdofs[3*i+0];
		x.data()[6*i + 1] =  rotdofs[3*i+1];
		x.data()[6*i + 2] =  rotdofs[3*i+2];
		x.data()[6*i + 3] =  transdofs[3*i+0];
		x.data()[6*i + 4] =  transdofs[3*i+1];
		x.data()[6*i + 5] =  transdofs[3*i+2];
	}


	//check high residuals



	//delete [] x;
	delete [] rotdofs;
	delete [] transdofs;
}

Eigen::Vector3d makeEigen(float3 x) {
	return(Eigen::Vector3d(x.x,x.y,x.z));
}

//note. Make sure there are no memory leaks
//converts cpu efficient matches to gpu cors
//todo just use normal matches here and get rid of efficient cpu stuff
EntryJ* solverWrapper::constructGPUCors() {
	nmatches = m->efficientMatches.size();
	nChunks = m->chunks.size();

	//construct a cpu vector
	std::vector<EntryJ> cpucors;
	cpucors.reserve(nmatches);
	for (auto& v : m->efficientMatches) {
		EntryJ tmp;
		tmp.imgIdx_i = v.fi1;
		tmp.imgIdx_j = v.fi2;
		auto& p =  m->chunks[v.fi1]->efficientKeypoints[v.i1].p;
		tmp.pos_i  = make_float3((float)p(0),(float)p(1),(float)p(2)); //keypoint
		auto& p2 =  m->chunks[v.fi2]->efficientKeypoints[v.i2].p;
		tmp.pos_j  = make_float3((float)p2(0),(float)p2(1),(float)p2(2)); //keypoint
		cpucors.push_back(tmp);
		//cout << "points between " << v.fi1 <<" and " << v.fi2 << " are "  << p.transpose() << " and " << p2.transpose()  << endl;
	}	

	CheckCuda(cudaMemcpy(d_cors, cpucors.data(), sizeof(EntryJ) * nmatches, cudaMemcpyHostToDevice));

	//cout << "cpucors \n";
	//for (int i = 0; i < nmatches; i++) {
	//	cout << "cpucors indices i " << cpucors.data()[i].imgIdx_i << endl;
	//	cout << "cpucors indices j " << cpucors.data()[i].imgIdx_j << endl;
	//	cout << "cpucors indices posi " << makeEigen(cpucors.data()[i].pos_i) << endl;
	//	cout << "cpucors indices posj " << makeEigen(cpucors.data()[i].pos_j) << endl;
	//}
	//
	//cout << "jentry from cuda \n";
	//EntryJ * entries = new EntryJ[cpucors.size()];
	//CheckCuda(cudaMemcpy(entries, d_cors, sizeof(EntryJ) * nmatches, cudaMemcpyDeviceToHost));
	//for (int i = 0; i < nmatches; i++) {
	//	cout << "cuda index i " << entries[i].imgIdx_i <<endl;
	//	cout << "cuda index j " <<  entries[i].imgIdx_j <<endl;
	//	cout << "cuda index j " <<  makeEigen(entries[i].pos_i) <<endl;
	//	cout << "cuda index j " <<  makeEigen(entries[i].pos_j) <<endl;
	//}




	return d_cors;
}
