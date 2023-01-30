#pragma once

#include <cuda_runtime.h>
//removed the directx11 compatibility thing
#include "cudaUtil.h" // only math
#include "cuda_SimpleMatrixUtil.h" //matrix functions. only inlcudes cudaUtil

#include "SolverParameters.h" //paras, mostly thresholds and iteration numbers
#include "SolverState.h" //input and state
#include "CUDATimer.h"

#include <conio.h> //stupid text interface, maybe remove
#include <vector>
class CUDACache;
//#define NEW_GUIDED_REMOVE 

struct vec2ui {
	unsigned int x;
	unsigned int y;

};

class CUDASolver
{
public:

	CUDASolver(unsigned int maxNumberOfImages, unsigned int maxNumResiduals);
	~CUDASolver();

	//weightSparse*Esparse + (#iters*weightDenseLinFactor + weightDense)*Edense
	void solve(EntryJ* d_correspondences, unsigned int numberOfCorrespondences,unsigned int numberOfElements, unsigned int nNonLinearIterations, unsigned int nLinearIterations,	const std::vector<float>& weightsSparse,float3* d_rotationAnglesUnknowns, float3* d_translationUnknowns, bool rebuildJT, bool findMaxResidual, unsigned int revalidateIdx);

	const std::vector<float>& getConvergenceAnalysis() const { return m_convergence; }
	const std::vector<float>& getLinearConvergenceAnalysis() const { return m_linConvergence; }

	void getMaxResidual(float& max, int& index) const {
		max = m_solverExtra.h_maxResidual[0];
		index = m_solverExtra.h_maxResidualIndex[0];
	};
	bool getMaxResidual(unsigned int curFrame, EntryJ* d_correspondences, vec2ui& imageIndices, float& maxRes);
	bool useVerification(EntryJ* d_correspondences, unsigned int numberOfCorrespondences);

	const int* getVariablesToCorrespondences() const { return d_variablesToCorrespondences; }
	const int* getVarToCorrNumEntriesPerRow() const { return d_numEntriesPerRow; }

	void evaluateTimings() {
		if (m_timer) {
			//std::cout << "********* SOLVER TIMINGS *********" << std::endl;
			m_timer->evaluate(true);
			std::cout << std::endl << std::endl;
		}
	}

	void resetTimer() {
		if (m_timer) m_timer->reset();
	}

#ifdef NEW_GUIDED_REMOVE
	const std::vector<vec2ui>& getGuidedMaxResImagesToRemove() const { return m_maxResImPairs; }
#endif
private:

	////!helper
	//static bool isSimilarImagePair(const vec2ui& pair0, const vec2ui& pair1) {
	//	if ((std::abs((int)pair0.x - (int)pair1.x) < 10 && std::abs((int)pair0.y - (int)pair1.y) < 10) ||
	//		(std::abs((int)pair0.x - (int)pair1.y) < 10 && std::abs((int)pair0.y - (int)pair1.x) < 10))
	//		return true;
	//	return false;
	//}
	
	void buildVariablesToCorrespondencesTable(EntryJ* d_correspondences, unsigned int numberOfCorrespondences);
	void computeMaxResidual(SolverInput& solverInput, SolverParameters& parameters, unsigned int revalidateIdx);

	SolverState	m_solverState;
	SolverStateAnalysis m_solverExtra;
	const unsigned int THREADS_PER_BLOCK;

	unsigned int m_maxNumberOfImages;
	unsigned int m_maxCorrPerImage;


	int* d_variablesToCorrespondences;
	int* d_numEntriesPerRow;

	std::vector<float> m_convergence; // convergence analysis (energy per non-linear iteration)
	std::vector<float> m_linConvergence; // linear residual per linear iteration, concatenates for nonlinear its

	float m_verifyOptDistThresh;
	float m_verifyOptPercentThresh;

	bool		m_bRecordConvergence;
	CUDATimer *m_timer;
	SolverParameters m_defaultParams;
	float			 m_maxResidualThresh;

#ifdef NEW_GUIDED_REMOVE
	//for more than one im-pair removal
	std::vector<vec2ui> m_maxResImPairs;

	//!!!debugging
	float4x4*	d_transforms;
	//!!!debugging
#endif
};
