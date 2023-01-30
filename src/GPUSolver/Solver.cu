#include <iostream>
#include <Cuda/Common/UtilsCuda.h>// for open3d cuda error check

////for debug purposes
//#define PRINT_RESIDUALS_SPARSE
//#define CUDA_ERROR_CHECK
//#define PRINT_RESIDUALS_DENSE

#define ENABLE_EARLY_OUT

#include "../GlobalDefines.h"
#include "SolverParameters.h"
#include "SolverState.h"
#include "SolverUtil.h"
#include "SolverEquations.h"
#include "CUDATimer.h"
//#include "SolverBundlingEquationsLie.h"
//#include "SolverBundlingDenseUtil.h"

#include <conio.h>

#define THREADS_PER_BLOCK_DENSE_DEPTH 128
#define THREADS_PER_BLOCK_DENSE_DEPTH_FLIP 64

#define THREADS_PER_BLOCK_DENSE_OVERLAP 512



__global__ void FlipJtJ_Kernel(unsigned int total, unsigned int dim, float* d_JtJ)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total) {
		const unsigned int x = idx % dim;
		const unsigned int y = idx / dim;
		if (x > y) {
			d_JtJ[y * dim + x] = d_JtJ[x * dim + y];
		}
	}
}

//####################### all commented out residual stuf that for some reason uses lie algebra thingys ###############

////todo more efficient?? (there are multiple per image-image...)
////get high residuals
//todo why does this need lie algebra stuff?
//__global__ void collectHighResidualsDevice(SolverInput input, SolverState state, SolverStateAnalysis analysis, SolverParameters parameters, unsigned int maxNumHighResiduals)
//{
//	const unsigned int N = input.numberOfCorrespondences; // Number of block variables
//	const unsigned int corrIdx = blockIdx.x * blockDim.x + threadIdx.x;
//
//	if (corrIdx < N) {
//		float residual = evalAbsMaxResidualDevice(corrIdx, input, state, parameters);
//		if (residual > parameters.highResidualThresh) {
//			int idx = atomicAdd(state.d_countHighResidual, 1);
//			if (idx < maxNumHighResiduals) {
//				analysis.d_maxResidual[idx] = residual;
//				analysis.d_maxResidualIndex[idx] = corrIdx;
//			}
//		}
//	}
//}
//void collectHighResiduals(SolverInput& input, SolverState& state, SolverStateAnalysis& analysis, SolverParameters& parameters, CUDATimer* timer)
//{
//	if (timer) timer->startEvent(__FUNCTION__);
//	cutilSafeCall(cudaMemset(state.d_countHighResidual, 0, sizeof(int)));
//
//	const unsigned int N = input.numberOfCorrespondences; // Number of correspondences 
//	unsigned int maxNumHighResiduals = (input.maxCorrPerImage*input.maxNumberOfImages + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
//	collectHighResidualsDevice << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(input, state, analysis, parameters, maxNumHighResiduals);
//
//#ifdef _DEBUG
//	cutilSafeCall(cudaDeviceSynchronize());
//	cutilCheckMsg(__FUNCTION__);
//#endif
//	if (timer) timer->endEvent();
//}

/////////////////////////////////////////////////////////////////////////
// Eval Max Residual
/////////////////////////////////////////////////////////////////////////

__global__ void EvalMaxResidualDevice(SolverInput input, SolverState state, SolverStateAnalysis analysis, SolverParameters parameters)
{
	__shared__ int maxResIndex[THREADS_PER_BLOCK];
	__shared__ float maxRes[THREADS_PER_BLOCK];

	const unsigned int N = input.numberOfCorrespondences; // Number of block variables
	const unsigned int corrIdx = blockIdx.x * blockDim.x + threadIdx.x;

	maxResIndex[threadIdx.x] = 0;
	maxRes[threadIdx.x] = 0.0f;

	if (corrIdx < N) {
		float residual = evalAbsMaxResidualDevice(corrIdx, input, state, parameters);

		maxRes[threadIdx.x] = residual;
		maxResIndex[threadIdx.x] = corrIdx;

		__syncthreads();

		for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride /= 2) {

			if (threadIdx.x < stride) {
				int first = threadIdx.x;
				int second = threadIdx.x + stride;
				if (maxRes[first] < maxRes[second]) {
					maxRes[first] = maxRes[second];
					maxResIndex[first] = maxResIndex[second];
				}
			}

			__syncthreads();
		}

		if (threadIdx.x == 0) {
			//printf("d_maxResidual[%d] = %f (index %d)\n", blockIdx.x, maxRes[0], maxResIndex[0]);
			analysis.d_maxResidual[blockIdx.x] = maxRes[0];
			analysis.d_maxResidualIndex[blockIdx.x] = maxResIndex[0];
		}
	}
}

void evalMaxResidual(SolverInput& input, SolverState& state, SolverStateAnalysis& analysis, SolverParameters& parameters, CUDATimer* timer)
{
	if (timer) timer->startEvent(__FUNCTION__);

	const unsigned int N = input.numberOfCorrespondences; // Number of correspondences 
	EvalMaxResidualDevice << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(input, state, analysis, parameters);

#ifdef CUDA_ERROR_CHECK
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
#endif

	if (timer) timer->endEvent();
}

/////////////////////////////////////////////////////////////////////////
// Eval Cost
/////////////////////////////////////////////////////////////////////////

__global__ void ResetResidualDevice(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x == 0) state.d_sumResidual[0] = 0.0f;
}

__global__ void EvalResidualDevice(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.numberOfCorrespondences; // Number of cors
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	float residual = 0.0f;
	if (x < N) {
		residual = evalFDevice(x, input, state, parameters,false);
		//float out = warpReduce(residual);
		//unsigned int laneid;
		////This command gets the lane ID within the current warp
		//asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
		//if (laneid == 0) {
		//	atomicAdd(&state.d_sumResidual[0], out);
		//}
		atomicAdd(&state.d_sumResidual[0], residual);
	}
}

float EvalResidual(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer* timer)
{
	if (timer) timer->startEvent(__FUNCTION__);

	float residual = 0.0f;

	const unsigned int N = input.numberOfCorrespondences; // Number of cors
	ResetResidualDevice << < 1, 1, 1 >> >(input, state, parameters);
	EvalResidualDevice << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(input, state, parameters);

	residual = state.getSumResidual();

#ifdef CUDA_ERROR_CHECK
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
#endif


	if (timer) timer->endEvent();

	return residual;
}

/////////////////////////////////////////////////////////////////////////
// Count High Residuals
/////////////////////////////////////////////////////////////////////////
//
__global__ void CountHighResidualsDevice(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.numberOfCorrespondences; // Number of block variables
	const unsigned int corrIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (corrIdx < N) {
		float residual = evalAbsMaxResidualDevice(corrIdx, input, state, parameters);

		if (residual > parameters.verifyOptDistThresh)
			atomicAdd(state.d_countHighResidual, 1);
	}
}

int countHighResiduals(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer* timer)
{
	if (timer) timer->startEvent(__FUNCTION__);

	const unsigned int N = input.numberOfCorrespondences; // Number of correspondences
	cutilSafeCall(cudaMemset(state.d_countHighResidual, 0, sizeof(int)));
	CountHighResidualsDevice << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(input, state, parameters);

	int count;
	CheckCuda(cudaMemcpy(&count, state.d_countHighResidual, sizeof(int), cudaMemcpyDeviceToHost));
#ifdef CUDA_ERROR_CHECK
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
#endif


	if (timer) timer->endEvent();
	return count;
}

/////////////////////////////////////////////////////////////////////////
// Convergence Analysis
/////////////////////////////////////////////////////////////////////////

//uses same data store as max residual
__global__ void EvalGNConvergenceDevice(SolverInput input, SolverStateAnalysis analysis, SolverState state) //compute max of delta
{
	__shared__ float maxVal[THREADS_PER_BLOCK];

	const unsigned int N = input.numberOfImages;
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	maxVal[threadIdx.x] = 0.0f;

	if (x < N)
	{
		if (x == 0)
			maxVal[threadIdx.x] = 0.0f;
		else {
			float3 r3 = fmaxf(fabs(state.d_deltaRot[x]), fabs(state.d_deltaTrans[x]));
			float r = fmaxf(r3.x, fmaxf(r3.y, r3.z));
			maxVal[threadIdx.x] = r;
		}
		__syncthreads();

		for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride /= 2) {
			if (threadIdx.x < stride) {
				int first = threadIdx.x;
				int second = threadIdx.x + stride;
				maxVal[first] = fmaxf(maxVal[first], maxVal[second]);
			}
			__syncthreads();
		}
		if (threadIdx.x == 0) {
			analysis.d_maxResidual[blockIdx.x] = maxVal[0];
		}
	}
}

float EvalGNConvergence(SolverInput& input, SolverState& state, SolverStateAnalysis& analysis, CUDATimer* timer)
{
	if (timer) timer->startEvent(__FUNCTION__);

	const unsigned int N = input.numberOfImages;
	const unsigned int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	EvalGNConvergenceDevice << < blocksPerGrid, THREADS_PER_BLOCK >> >(input, analysis, state);

#ifdef CUDA_ERROR_CHECK
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
#endif

	//copy to host and compute max
	CheckCuda(cudaMemcpy(analysis.h_maxResidual, analysis.d_maxResidual, sizeof(float) * blocksPerGrid, cudaMemcpyDeviceToHost));
	CheckCuda(cudaMemcpy(analysis.h_maxResidualIndex, analysis.d_maxResidualIndex, sizeof(int) * blocksPerGrid, cudaMemcpyDeviceToHost));
	float maxVal = 0.0f;
	for (unsigned int i = 0; i < blocksPerGrid; i++) {
		if (maxVal < analysis.h_maxResidual[i]) maxVal = analysis.h_maxResidual[i];
	}
	if (timer) timer->endEvent();

	return maxVal;
}

// For the naming scheme of the variables see:
// http://en.wikipedia.org/wiki/Conjugate_gradient_method
// This code is an implementation of their PCG pseudo code

template<bool useDense>
__global__ void PCGInit_Kernel1(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.numberOfImages;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;

	float d = 0.0f;
	if (x > 0 && x < N)
	{
		float3 resRot, resTrans;
		evalMinusJTFDevice<useDense>(x, input, state, parameters, resRot, resTrans);  // residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0 

		state.d_rRot[x] = resRot;											// store for next iteration
		state.d_rTrans[x] = resTrans;										// store for next iteration

		const float3 pRot = state.d_precondionerRot[x] * resRot;			// apply preconditioner M^-1
		state.d_pRot[x] = pRot;

		const float3 pTrans = state.d_precondionerTrans[x] * resTrans;		// apply preconditioner M^-1
		state.d_pTrans[x] = pTrans;

		d = dot(resRot, pRot) + dot(resTrans, pTrans);						// x-th term of nomimator for computing alpha and denominator for computing beta

		state.d_Ap_XRot[x] = make_float3(0.0f, 0.0f, 0.0f);
		state.d_Ap_XTrans[x] = make_float3(0.0f, 0.0f, 0.0f);
	}

	d = warpReduce(d);
	if (threadIdx.x % WARP_SIZE == 0)
	{
		atomicAdd(state.d_scanAlpha, d);
	}
}

__global__ void PCGInit_Kernel2(unsigned int N, SolverState state)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x > 0 && x < N) state.d_rDotzOld[x] = state.d_scanAlpha[0];				// store result for next kernel call
}

void Initialization(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer* timer)
{
	const unsigned int N = input.numberOfImages;

	const int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	if (blocksPerGrid > THREADS_PER_BLOCK)
	{
		std::cout << "Too many variables for this block size. Maximum number of variables for two kernel scan: " << THREADS_PER_BLOCK*THREADS_PER_BLOCK << std::endl;
		while (1);
	}

	if (timer) timer->startEvent("Initialization");

	//!!!DEBUGGING //remember to uncomment the delete...
	//float3* rRot = new float3[input.numberOfImages]; // -jtf
	//float3* rTrans = new float3[input.numberOfImages];
	//!!!DEBUGGING

	CheckCuda(cudaMemset(state.d_scanAlpha, 0, sizeof(float)));
#ifdef CUDA_ERROR_CHECK
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
#endif

	
	PCGInit_Kernel1<false> << <blocksPerGrid, THREADS_PER_BLOCK >> >(input, state, parameters);
#ifdef CUDA_ERROR_CHECK
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
#endif


	PCGInit_Kernel2 << <blocksPerGrid, THREADS_PER_BLOCK >> >(N, state);
#ifdef CUDA_ERROR_CHECK
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
#endif

	if (timer) timer->endEvent();

	//float scanAlpha;
	//cutilSafeCall(cudaMemcpy(&scanAlpha, state.d_scanAlpha, sizeof(float), cudaMemcpyDeviceToHost));
	//if (rRot) delete[] rRot;
	//if (rTrans) delete[] rTrans;
}

/////////////////////////////////////////////////////////////////////////
// PCG Iteration Parts
/////////////////////////////////////////////////////////////////////////

__global__ void PCGStep_Kernel0(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.numberOfCorrespondences;					// Number of block variables
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < N)
	{
		const float3 tmp = applyJDevice(x, input, state, parameters);		// A x p_k  => J^T x J x p_k 
		state.d_Jp[x] = tmp;												// store for next kernel call
	}
}

__global__ void PCGStep_Kernel1a(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.numberOfImages;							// Number of block variables
	const unsigned int x = blockIdx.x;
	const unsigned int lane = threadIdx.x % WARP_SIZE;

	if (x > 0 && x < N)
	{
		float3 rot, trans;
		applyJTDevice(x, input, state, parameters, rot, trans, threadIdx.x, lane);			// A x p_k  => J^T x J x p_k 

		if (lane == 0)
		{
			atomicAdd(&state.d_Ap_XRot[x].x, rot.x);
			atomicAdd(&state.d_Ap_XRot[x].y, rot.y);
			atomicAdd(&state.d_Ap_XRot[x].z, rot.z);

			atomicAdd(&state.d_Ap_XTrans[x].x, trans.x);
			atomicAdd(&state.d_Ap_XTrans[x].y, trans.y);
			atomicAdd(&state.d_Ap_XTrans[x].z, trans.z);
		}
	}
}

__global__ void PCGStep_Kernel1b(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.numberOfImages;								// Number of block variables
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	float d = 0.0f;
	if (x > 0 && x < N)
	{
		d = dot(state.d_pRot[x], state.d_Ap_XRot[x]) + dot(state.d_pTrans[x], state.d_Ap_XTrans[x]);		// x-th term of denominator of alpha
	}

	d = warpReduce(d);
	if (threadIdx.x % WARP_SIZE == 0)
	{
		atomicAdd(state.d_scanAlpha, d);
	}
}

__global__ void PCGStep_Kernel2(SolverInput input, SolverState state)
{
	const unsigned int N = input.numberOfImages;
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	const float dotProduct = state.d_scanAlpha[0];

	float b = 0.0f;
	if (x > 0 && x < N)
	{
		float alpha = 0.0f;
		if (dotProduct > FLOAT_EPSILON) alpha = state.d_rDotzOld[x] / dotProduct;		// update step size alpha

		state.d_deltaRot[x] = state.d_deltaRot[x] + alpha*state.d_pRot[x];			// do a decent step
		state.d_deltaTrans[x] = state.d_deltaTrans[x] + alpha*state.d_pTrans[x];	// do a decent step

		float3 rRot = state.d_rRot[x] - alpha*state.d_Ap_XRot[x];					// update residuum
		state.d_rRot[x] = rRot;														// store for next kernel call

		float3 rTrans = state.d_rTrans[x] - alpha*state.d_Ap_XTrans[x];				// update residuum
		state.d_rTrans[x] = rTrans;													// store for next kernel call

		float3 zRot = state.d_precondionerRot[x] * rRot;							// apply preconditioner M^-1
		state.d_zRot[x] = zRot;														// save for next kernel call

		float3 zTrans = state.d_precondionerTrans[x] * rTrans;						// apply preconditioner M^-1
		state.d_zTrans[x] = zTrans;													// save for next kernel call

		b = dot(zRot, rRot) + dot(zTrans, rTrans);									// compute x-th term of the nominator of beta
	}
	b = warpReduce(b);
	if (threadIdx.x % WARP_SIZE == 0)
	{
		atomicAdd(&state.d_scanAlpha[1], b);
	}
}

template<bool lastIteration>
__global__ void PCGStep_Kernel3(SolverInput input, SolverState state)
{
	const unsigned int N = input.numberOfImages;
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x > 0 && x < N)
	{
		const float rDotzNew = state.d_scanAlpha[1];								// get new nominator
		const float rDotzOld = state.d_rDotzOld[x];								// get old denominator

		float beta = 0.0f;
		if (rDotzOld > FLOAT_EPSILON) beta = rDotzNew / rDotzOld;				// update step size beta

		state.d_rDotzOld[x] = rDotzNew;											// save new rDotz for next iteration
		state.d_pRot[x] = state.d_zRot[x] + beta*state.d_pRot[x];		// update decent direction
		state.d_pTrans[x] = state.d_zTrans[x] + beta*state.d_pTrans[x];		// update decent direction


		state.d_Ap_XRot[x] = make_float3(0.0f, 0.0f, 0.0f);
		state.d_Ap_XTrans[x] = make_float3(0.0f, 0.0f, 0.0f);

		if (lastIteration)
		{
			//if (input.d_validImages[x]) { //not really necessary
#ifdef USE_LIE_SPACE //TODO just keep that matrix transforms around
			float3 rot, trans;
			computeLieUpdate(state.d_deltaRot[x], state.d_deltaTrans[x], state.d_xRot[x], state.d_xTrans[x], rot, trans);
			state.d_xRot[x] = rot;
			state.d_xTrans[x] = trans;
#else
			state.d_xRot[x] = state.d_xRot[x] + state.d_deltaRot[x];
			state.d_xTrans[x] = state.d_xTrans[x] + state.d_deltaTrans[x];
#endif
			//}
		}
	}
}

template<bool useSparse, bool useDense>
bool PCGIteration(SolverInput& input, SolverState& state, SolverParameters& parameters, SolverStateAnalysis& analysis, bool lastIteration, CUDATimer *timer)
{
	const unsigned int N = input.numberOfImages;	// Number of block variables

	// Do PCG step
	const int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	if (blocksPerGrid > THREADS_PER_BLOCK)
	{
		std::cout << "Too many variables for this block size. Maximum number of variables for two kernel scan: " << THREADS_PER_BLOCK*THREADS_PER_BLOCK << std::endl;
		while (1);
	}
	if (timer) timer->startEvent("PCGIteration");

	CheckCuda(cudaMemset(state.d_scanAlpha, 0, sizeof(float) * 2));

	// sparse part
	if (useSparse) {
		const unsigned int Ncorr = input.numberOfCorrespondences;
		const int blocksPerGridCorr = (Ncorr + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		PCGStep_Kernel0 << <blocksPerGridCorr, THREADS_PER_BLOCK >> >(input, state, parameters);
#ifdef CUDA_ERROR_CHECK
		CheckCuda(cudaDeviceSynchronize());
		CheckCuda(cudaGetLastError());
#endif

		PCGStep_Kernel1a << < N, THREADS_PER_BLOCK_JT >> >(input, state, parameters);
#ifdef CUDA_ERROR_CHECK
		CheckCuda(cudaDeviceSynchronize());
		CheckCuda(cudaGetLastError());
#endif

	}

	PCGStep_Kernel1b << <blocksPerGrid, THREADS_PER_BLOCK >> >(input, state, parameters);
#ifdef CUDA_ERROR_CHECK
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
#endif

	PCGStep_Kernel2 << <blocksPerGrid, THREADS_PER_BLOCK >> >(input, state);
#ifdef CUDA_ERROR_CHECK
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
#endif

#ifdef ENABLE_EARLY_OUT //for convergence
	float scanAlpha; CheckCuda(cudaMemcpy(&scanAlpha, state.d_scanAlpha, sizeof(float), cudaMemcpyDeviceToHost));
	//if (fabs(scanAlpha) < 0.00005f) lastIteration = true;  //todo check this part
	//if (fabs(scanAlpha) < 1e-6) lastIteration = true;  //todo check this part
	if (fabs(scanAlpha) < 5e-7) { lastIteration = true; }  //todo check this part
#endif
	if (lastIteration) {
		PCGStep_Kernel3<true> << <blocksPerGrid, THREADS_PER_BLOCK >> >(input, state);
	}
	else {
		PCGStep_Kernel3<false> << <blocksPerGrid, THREADS_PER_BLOCK >> >(input, state);
	}

#ifdef CUDA_ERROR_CHECK
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
#endif
	if (timer) timer->endEvent();

	return lastIteration;
}

////////////////////////////////////////////////////////////////////
// Main GN Solver Loop
////////////////////////////////////////////////////////////////////

void solveBundlingStub(SolverInput& input, SolverState& state, SolverParameters& parameters, SolverStateAnalysis& analysis, float* convergenceAnalysis, CUDATimer *timer)
{
	if (convergenceAnalysis) { //this only occurs if m_bRecordConvergence is set to true in constructor
		float initialResidual = EvalResidual(input, state, parameters, timer);
		convergenceAnalysis[0] = initialResidual; // initial residual
	}

	//!!!DEBUGGING
#ifdef PRINT_RESIDUALS_SPARSE
	if (parameters.weightSparse > 0) {
		if (input.numberOfCorrespondences == 0) { printf("ERROR: %d correspondences\n", input.numberOfCorrespondences); getchar(); }
		float initialResidual = EvalResidual(input, state, parameters, timer);
		printf("initial sparse = %f*%f = %f\n", parameters.weightSparse, initialResidual / parameters.weightSparse, initialResidual);
		//note: for some stupid reason  parameters.weightSparse is just the first element of the sparse weight vector 
	}
#endif


	for (unsigned int nIter = 0; nIter < parameters.nNonLinearIterations; nIter++)
	{
		parameters.weightSparse = input.weightsSparse[nIter];

		//probably builds the local linear problem
		Initialization(input, state, parameters, timer);

		if (parameters.weightSparse > 0.0f) {
			for (unsigned int linIter = 0; linIter < parameters.nLinIterations; linIter++){
				if (PCGIteration<true, false>(input, state, parameters, analysis, linIter == parameters.nLinIterations - 1, timer)) {
					//totalLinIters += (linIter+1); numLin++; 
					break;
				}
			}
		}
		else {
			std::cout << "weight is zero, something is weird in solver.cu line 620 \n";
		}

#ifdef PRINT_RESIDUALS_SPARSE
		if (parameters.weightSparse > 0) {
			float residual = EvalResidual(input, state, parameters, timer);
			printf("[niter %d] weight * sparse = %f*%f = %f\t[#corr = %d]\n", nIter, parameters.weightSparse, residual / parameters.weightSparse, residual, input.numberOfCorrespondences);
		}
#endif
		if (convergenceAnalysis) {
			float residual = EvalResidual(input, state, parameters, timer);
			convergenceAnalysis[nIter + 1] = residual;
		}

		//if (timer) timer->evaluate(true);

#ifdef ENABLE_EARLY_OUT //convergence
		//if (nIter < parameters.nNonLinearIterations - 1 && EvalGNConvergence(input, state, analysis, timer) < 0.01f) { //!!! TODO CHECK HOW THESE GENERALIZE
		if (nIter < parameters.nNonLinearIterations - 1 && EvalGNConvergence(input, state, analysis, timer) < 0.005f) { //0.001?
		//if (nIter < parameters.nNonLinearIterations - 1 && EvalGNConvergence(input, state, analysis, timer) < 0.001f) { 
			//if (!parameters.useDense) { totalNonLinIters += (nIter+1); numNonLin++; }
#ifdef PRINT_RESIDUALS_SPARSE
			std::cout << "stopped due to convergence analysis \n";
#endif
			break;
		}
		//else if (!parameters.useDense && nIter == parameters.nNonLinearIterations - 1) { totalNonLinIters += (nIter+1); numNonLin++; }
#endif
		}

	}

////////////////////////////////////////////////////////////////////
// build variables to correspondences lookup
////////////////////////////////////////////////////////////////////

__global__ void BuildVariablesToCorrespondencesTableDevice(EntryJ* d_correspondences, unsigned int numberOfCorrespondences,
	unsigned int maxNumCorrespondencesPerImage, int* d_variablesToCorrespondences, int* d_numEntriesPerRow)
{
	const unsigned int N = numberOfCorrespondences; // Number of block variables
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < N) {
		EntryJ& corr = d_correspondences[x];
		if (corr.isValid()) {
			int offset0 = atomicAdd(&d_numEntriesPerRow[corr.imgIdx_i], 1); // may overflow - need to check when read
			int offset1 = atomicAdd(&d_numEntriesPerRow[corr.imgIdx_j], 1); // may overflow - need to check when read
			if (offset0 < maxNumCorrespondencesPerImage && offset1 < maxNumCorrespondencesPerImage)	{
				d_variablesToCorrespondences[corr.imgIdx_i * maxNumCorrespondencesPerImage + offset0] = x;
				d_variablesToCorrespondences[corr.imgIdx_j * maxNumCorrespondencesPerImage + offset1] = x;
			}
			else { //invalidate
				printf("EXCEEDED MAX NUM CORR PER IMAGE IN SOLVER, INVALIDATING %d(%d,%d) [%d,%d | %d]\n",
					x, corr.imgIdx_i, corr.imgIdx_j, offset0, offset1, maxNumCorrespondencesPerImage); //debugging
				corr.setInvalid(); //make sure j corresponds to jt
			}
		}
	}
}

void buildVariablesToCorrespondencesTableCUDA(EntryJ* d_correspondences, unsigned int numberOfCorrespondences, unsigned int maxNumCorrespondencesPerImage, int* d_variablesToCorrespondences, int* d_numEntriesPerRow, CUDATimer* timer)
{
	const unsigned int N = numberOfCorrespondences;

	if (timer) timer->startEvent(__FUNCTION__);

	BuildVariablesToCorrespondencesTableDevice << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(d_correspondences, numberOfCorrespondences, maxNumCorrespondencesPerImage, d_variablesToCorrespondences, d_numEntriesPerRow);
#ifdef CUDA_ERROR_CHECK
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
#endif

	if (timer) timer->endEvent();
}
