#pragma once

#ifndef _SOLVER_STATE_
#define _SOLVER_STATE_

#include <cuda_runtime.h> 
#include "cuda_SimpleMatrixUtil.h" //is otherwise included by the image manager
//#include "../SiftGPU/SIFTImageManager.h" //should be unnecesary due to having entryj local
//#include "../CUDACacheUtil.h" //only for dense needed, since its gpu images

//correspondence_idx -> image_Idx_i,j
struct EntryJ {
	unsigned int imgIdx_i;
	unsigned int imgIdx_j;
	float3 pos_i;
	float3 pos_j;

	__host__ __device__
		void setInvalid() {
		imgIdx_i = (unsigned int)-1;
		imgIdx_j = (unsigned int)-1;
	}
	__host__ __device__
		bool isValid() const {
		return imgIdx_i != (unsigned int)-1;
	}
};


struct SolverInput
{	
	EntryJ* d_correspondences;
	int* d_variablesToCorrespondences;
	int* d_numEntriesPerRow;

	unsigned int numberOfCorrespondences;
	unsigned int numberOfImages;

	unsigned int maxNumberOfImages;
	unsigned int maxCorrPerImage;

	const float* weightsSparse;
};

// State of the GN Solver
struct SolverState
{
	float3*	d_deltaRot;					// Current linear update to be computed
	float3*	d_deltaTrans;				// Current linear update to be computed
	
	float3* d_xRot;						// Current state
	float3* d_xTrans;					// Current state

	float3*	d_rRot;						// Residuum // jtf
	float3*	d_rTrans;					// Residuum // jtf
	
	float3*	d_zRot;						// Preconditioned residuum
	float3*	d_zTrans;					// Preconditioned residuum
	
	float3*	d_pRot;						// Decent direction
	float3*	d_pTrans;					// Decent direction
	
	float3*	d_Jp;						// Cache values after J

	float3*	d_Ap_XRot;					// Cache values for next kernel call after A = J^T x J x p
	float3*	d_Ap_XTrans;				// Cache values for next kernel call after A = J^T x J x p

	float*	d_scanAlpha;				// Tmp memory for alpha scan

	float*	d_rDotzOld;					// Old nominator (denominator) of alpha (beta)
	
	float3*	d_precondionerRot;			// Preconditioner for linear system
	float3*	d_precondionerTrans;		// Preconditioner for linear system

	float*	d_sumResidual;				// sum of the squared residuals //debug

	//float* d_residuals; // debugging
	//float* d_sumLinResidual; // debugging // helper to compute linear residual

	int* d_countHighResidual;

	__host__ float getSumResidual() const {
		float residual;
		cudaMemcpy(&residual, d_sumResidual, sizeof(float), cudaMemcpyDeviceToHost);
		return residual;
	}


	float4x4* d_xTransforms;
	float4x4* d_xTransformInverses;

	//!!!DEBUGGING
	int* d_corrCount;
	int* d_corrCountColor;
	float* d_sumResidualColor;
};

struct SolverStateAnalysis
{
	// residual pruning
	int*	d_maxResidualIndex;
	float*	d_maxResidual;

	int*	h_maxResidualIndex;
	float*	h_maxResidual;
};

#endif
