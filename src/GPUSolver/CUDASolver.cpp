
//removed the precompiled header
#include "../GlobalDefines.h"
#include "CUDASolver.h"
#include <Cuda/Common/UtilsCuda.h>// for open3d cuda error check
//#include "../GlobalBundlingState.h" we should not need it
//#include "../CUDACache.h" should also not need since we do not do dense
//#include "../SiftGPU/MatrixConversion.h" hopefully dont need it since we dont use mlib
#define SAFE_DELETE_ARRAY(p) { if (p) { delete[] (p);   (p)=NULL; } } //why not
//dont need stupid extern c stuff

//why declaration here instead of in header?
void evalMaxResidual(SolverInput& input, SolverState& state, SolverStateAnalysis& analysis, SolverParameters& parameters, CUDATimer* timer);
int countHighResiduals(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer* timer);

void buildVariablesToCorrespondencesTableCUDA(EntryJ* d_correspondences, unsigned int numberOfCorrespondences, unsigned int maxNumCorrespondencesPerImage, int* d_variablesToCorrespondences, int* d_numEntriesPerRow, CUDATimer* timer);
void solveBundlingStub(SolverInput& input, SolverState& state, SolverParameters& parameters, SolverStateAnalysis& analysis, float* convergenceAnalysis, CUDATimer* timer);


//#define DEBUG_PRINT_SPARSE_RESIDUALS
#ifdef DEBUG_PRINT_SPARSE_RESIDUALS
float EvalResidual(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer* timer);
#endif

CUDASolver::CUDASolver(unsigned int maxNumberOfImages, unsigned int maxNumResiduals)
	: m_maxNumberOfImages(maxNumberOfImages)
	, THREADS_PER_BLOCK(512) // keep consistent with the GPU
{
	//m_timer = new CUDATimer();
	m_timer = NULL;
	m_bRecordConvergence = false;

	//TODO PARAMS
	//const unsigned int submapSize = GlobalBundlingState::get().s_submapSize;
	m_verifyOptDistThresh = 0.02f;//GlobalAppState::get().s_verifyOptDistThresh;
	m_verifyOptPercentThresh = 0.05f;//GlobalAppState::get().s_verifyOptPercentThresh;

	const unsigned int numberOfVariables = maxNumberOfImages;
	m_maxCorrPerImage = clamp(maxNumResiduals / maxNumberOfImages, 1000u, 4000u); //math:: was deleted here

	// State
	CheckCuda(cudaMalloc(&m_solverState.d_deltaRot, sizeof(float3)*numberOfVariables));
	CheckCuda(cudaMalloc(&m_solverState.d_deltaTrans, sizeof(float3)*numberOfVariables));
	CheckCuda(cudaMalloc(&m_solverState.d_rRot, sizeof(float3)*numberOfVariables));
	CheckCuda(cudaMalloc(&m_solverState.d_rTrans, sizeof(float3)*numberOfVariables));
	CheckCuda(cudaMalloc(&m_solverState.d_zRot, sizeof(float3)*numberOfVariables));
	CheckCuda(cudaMalloc(&m_solverState.d_zTrans, sizeof(float3)*numberOfVariables));
	CheckCuda(cudaMalloc(&m_solverState.d_pRot, sizeof(float3)*numberOfVariables));
	CheckCuda(cudaMalloc(&m_solverState.d_pTrans, sizeof(float3)*numberOfVariables));
	CheckCuda(cudaMalloc(&m_solverState.d_Jp, sizeof(float3)*maxNumResiduals));
	CheckCuda(cudaMalloc(&m_solverState.d_Ap_XRot, sizeof(float3)*numberOfVariables));
	CheckCuda(cudaMalloc(&m_solverState.d_Ap_XTrans, sizeof(float3)*numberOfVariables));
	CheckCuda(cudaMalloc(&m_solverState.d_scanAlpha, sizeof(float) * 2));
	CheckCuda(cudaMalloc(&m_solverState.d_rDotzOld, sizeof(float) *numberOfVariables));
	CheckCuda(cudaMalloc(&m_solverState.d_precondionerRot, sizeof(float3)*numberOfVariables));
	CheckCuda(cudaMalloc(&m_solverState.d_precondionerTrans, sizeof(float3)*numberOfVariables));
	CheckCuda(cudaMalloc(&m_solverState.d_sumResidual, sizeof(float)));
	unsigned int n = (maxNumResiduals + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	CheckCuda(cudaMalloc(&m_solverExtra.d_maxResidual, sizeof(float) * n));
	CheckCuda(cudaMalloc(&m_solverExtra.d_maxResidualIndex, sizeof(int) * n));
	m_solverExtra.h_maxResidual = new float[n];
	m_solverExtra.h_maxResidualIndex = new int[n];

	CheckCuda(cudaMalloc(&d_variablesToCorrespondences, sizeof(int)*m_maxNumberOfImages*m_maxCorrPerImage));
	CheckCuda(cudaMalloc(&d_numEntriesPerRow, sizeof(int)*m_maxNumberOfImages));

	CheckCuda(cudaMalloc(&m_solverState.d_countHighResidual, sizeof(int)));

	CheckCuda(cudaMalloc(&m_solverState.d_corrCount, sizeof(int)));
	CheckCuda(cudaMalloc(&m_solverState.d_corrCountColor, sizeof(int)));
	CheckCuda(cudaMalloc(&m_solverState.d_sumResidualColor, sizeof(float)));

	m_solverState.d_xTransforms = NULL;
	m_solverState.d_xTransformInverses = NULL;

#ifdef NEW_GUIDED_REMOVE
	cudaMalloc(&d_transforms, sizeof(float4x4)*m_maxNumberOfImages);
#endif

	//solve params
	m_maxResidualThresh = 0.05; //changed

	//!!!DEBUGGING //is this really just debugging?!!!
	CheckCuda(cudaMemset(m_solverState.d_deltaRot, -1, sizeof(float3)*numberOfVariables));
	CheckCuda(cudaMemset(m_solverState.d_deltaTrans, -1, sizeof(float3)*numberOfVariables));
	CheckCuda(cudaMemset(m_solverState.d_rRot, -1, sizeof(float3)*numberOfVariables));
	CheckCuda(cudaMemset(m_solverState.d_rTrans, -1, sizeof(float3)*numberOfVariables));
	CheckCuda(cudaMemset(m_solverState.d_zRot, -1, sizeof(float3)*numberOfVariables));
	CheckCuda(cudaMemset(m_solverState.d_zTrans, -1, sizeof(float3)*numberOfVariables));
	CheckCuda(cudaMemset(m_solverState.d_pRot, -1, sizeof(float3)*numberOfVariables));
	CheckCuda(cudaMemset(m_solverState.d_pTrans, -1, sizeof(float3)*numberOfVariables));
	CheckCuda(cudaMemset(m_solverState.d_Jp, -1, sizeof(float3)*maxNumResiduals));
	CheckCuda(cudaMemset(m_solverState.d_Ap_XRot, -1, sizeof(float3)*numberOfVariables));
	CheckCuda(cudaMemset(m_solverState.d_Ap_XTrans, -1, sizeof(float3)*numberOfVariables));
	CheckCuda(cudaMemset(m_solverState.d_scanAlpha, -1, sizeof(float) * 2));
	CheckCuda(cudaMemset(m_solverState.d_rDotzOld, -1, sizeof(float) *numberOfVariables));
	CheckCuda(cudaMemset(m_solverState.d_precondionerRot, -1, sizeof(float3)*numberOfVariables));
	CheckCuda(cudaMemset(m_solverState.d_precondionerTrans, -1, sizeof(float3)*numberOfVariables));
	CheckCuda(cudaMemset(m_solverState.d_sumResidual, -1, sizeof(float)));
	CheckCuda(cudaMemset(m_solverExtra.d_maxResidual, -1, sizeof(float) * n));
	CheckCuda(cudaMemset(m_solverExtra.d_maxResidualIndex, -1, sizeof(int) * n));
	CheckCuda(cudaMemset(d_variablesToCorrespondences, -1, sizeof(int)*m_maxNumberOfImages*m_maxCorrPerImage));
	CheckCuda(cudaMemset(d_numEntriesPerRow, -1, sizeof(int)*m_maxNumberOfImages));
	CheckCuda(cudaMemset(m_solverState.d_countHighResidual, -1, sizeof(int)));

	CheckCuda(cudaMemset(m_solverState.d_corrCount, -1, sizeof(int)));
	CheckCuda(cudaMemset(m_solverState.d_corrCountColor, -1, sizeof(int)));
	CheckCuda(cudaMemset(m_solverState.d_sumResidualColor, -1, sizeof(float)));

	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
	//!!!DEBUGGING
}

CUDASolver::~CUDASolver()
{
	if (m_timer) delete m_timer;
	// State
	CheckCuda(cudaFree(m_solverState.d_deltaRot));
	CheckCuda(cudaFree(m_solverState.d_deltaTrans));
	CheckCuda(cudaFree(m_solverState.d_rRot));
	CheckCuda(cudaFree(m_solverState.d_rTrans));
	CheckCuda(cudaFree(m_solverState.d_zRot));
	CheckCuda(cudaFree(m_solverState.d_zTrans));
	CheckCuda(cudaFree(m_solverState.d_pRot));
	CheckCuda(cudaFree(m_solverState.d_pTrans));
	CheckCuda(cudaFree(m_solverState.d_Jp));
	CheckCuda(cudaFree(m_solverState.d_Ap_XRot));
	CheckCuda(cudaFree(m_solverState.d_Ap_XTrans));
	CheckCuda(cudaFree(m_solverState.d_scanAlpha));
	CheckCuda(cudaFree(m_solverState.d_rDotzOld));
	CheckCuda(cudaFree(m_solverState.d_precondionerRot));
	CheckCuda(cudaFree(m_solverState.d_precondionerTrans));
	CheckCuda(cudaFree(m_solverState.d_sumResidual));
	CheckCuda(cudaFree(m_solverExtra.d_maxResidual));
	CheckCuda(cudaFree(m_solverExtra.d_maxResidualIndex));
	SAFE_DELETE_ARRAY(m_solverExtra.h_maxResidual);
	SAFE_DELETE_ARRAY(m_solverExtra.h_maxResidualIndex);

	CheckCuda(cudaFree(d_variablesToCorrespondences));
	CheckCuda(cudaFree(d_numEntriesPerRow));

	CheckCuda(cudaFree(m_solverState.d_countHighResidual));


	CheckCuda(cudaFree(m_solverState.d_xTransforms));
	CheckCuda(cudaFree(m_solverState.d_xTransformInverses));

	CheckCuda(cudaFree(m_solverState.d_corrCount));
	CheckCuda(cudaFree(m_solverState.d_sumResidualColor));
	CheckCuda(cudaFree(m_solverState.d_corrCountColor));

#ifdef NEW_GUIDED_REMOVE
	CheckCuda(cudaFree(d_transforms);
#endif
}

void CUDASolver::solve(EntryJ* d_correspondences, unsigned int numberOfCorrespondences,unsigned int numberOfElements, unsigned int nNonLinearIterations, unsigned int nLinearIterations,	const std::vector<float>& weightsSparse,float3* d_rotationAnglesUnknowns, float3* d_translationUnknowns, bool rebuildJT, bool findMaxResidual, unsigned int revalidateIdx)
{
	nNonLinearIterations = std::min(nNonLinearIterations, (unsigned int)weightsSparse.size());
	//MLIB_ASSERT(numberOfImages > 1 && nNonLinearIterations > 0);
	if (numberOfElements <= 1 || nNonLinearIterations == 0){
		std::cerr << "Fatal in solve \n";
	}
	if (numberOfCorrespondences > m_maxCorrPerImage*m_maxNumberOfImages) {
		//warning: correspondences will be invalidated AT RANDOM!
		std::cerr << "WARNING: #corr (" << numberOfCorrespondences << ") exceeded limit (" << m_maxCorrPerImage << "*" << m_maxNumberOfImages << "), please increase max #corr per image in the GAS" << std::endl;
	}

	float* convergence = NULL;
	if (m_bRecordConvergence) {
		m_convergence.resize(nNonLinearIterations + 1, -1.0f);
		convergence = m_convergence.data();
	}

	m_solverState.d_xRot = d_rotationAnglesUnknowns;
	m_solverState.d_xTrans = d_translationUnknowns;

	SolverParameters parameters = m_defaultParams;
	parameters.nNonLinearIterations = nNonLinearIterations;
	parameters.nLinIterations = nLinearIterations;
	parameters.verifyOptDistThresh = m_verifyOptDistThresh;
	parameters.verifyOptPercentThresh = m_verifyOptPercentThresh;
	parameters.highResidualThresh = std::numeric_limits<float>::infinity();

	parameters.weightSparse = weightsSparse.front();

	SolverInput solverInput;
	solverInput.d_correspondences = d_correspondences;
	solverInput.d_variablesToCorrespondences = d_variablesToCorrespondences;
	solverInput.d_numEntriesPerRow = d_numEntriesPerRow;
	solverInput.numberOfImages = numberOfElements;
	solverInput.numberOfCorrespondences = numberOfCorrespondences;

	solverInput.maxNumberOfImages = m_maxNumberOfImages;
	solverInput.maxCorrPerImage = m_maxCorrPerImage;

	solverInput.weightsSparse = weightsSparse.data();


#ifdef NEW_GUIDED_REMOVE
	convertLiePosesToMatricesCU(m_solverState.d_xRot, m_solverState.d_xTrans, solverInput.numberOfImages, d_transforms, m_solverState.d_xTransformInverses); //debugging only (store transforms before opt)
#endif
#ifdef DEBUG_PRINT_SPARSE_RESIDUALS
	if (findMaxResidual) {
		float residualBefore = EvalResidual(solverInput, m_solverState, parameters, NULL);
		computeMaxResidual(solverInput, parameters, (unsigned int)-1);
		vec2ui beforeMaxImageIndices; 
		float beforeMaxRes; 
		unsigned int curFrame = (revalidateIdx == (unsigned int)-1) ? solverInput.numberOfImages - 1 : revalidateIdx;
		getMaxResidual(curFrame, d_correspondences, beforeMaxImageIndices, beforeMaxRes);
		std::cout << "\tbefore: (" << solverInput.numberOfImages << ") sumres = " << residualBefore << " / " << solverInput.numberOfCorrespondences << " = " << residualBefore / (float)solverInput.numberOfCorrespondences << " | maxres = " << beforeMaxRes << " images (" << beforeMaxImageIndices << ")" << std::endl;
	}
#endif


	if (rebuildJT) {
		buildVariablesToCorrespondencesTable(d_correspondences, numberOfCorrespondences);
	}
	solveBundlingStub(solverInput, m_solverState, parameters, m_solverExtra, convergence, m_timer);

	if (findMaxResidual) {
		computeMaxResidual(solverInput, parameters, revalidateIdx);
#ifdef DEBUG_PRINT_SPARSE_RESIDUALS
		float residualAfter = EvalResidual(solverInput, m_solverState, parameters, NULL);
		vec2ui afterMaxImageIndices; float afterMaxRes; unsigned int curFrame = (revalidateIdx == (unsigned int)-1) ? solverInput.numberOfImages - 1 : revalidateIdx;
		getMaxResidual(curFrame, d_correspondences, afterMaxImageIndices, afterMaxRes);
		std::cout << "\tafter: (" << solverInput.numberOfImages << ") sumres = " << residualAfter << " / " << solverInput.numberOfCorrespondences << " = " << residualAfter / (float)solverInput.numberOfCorrespondences << " | maxres = " << afterMaxRes << " images (" << afterMaxImageIndices << ")" << std::endl;
#endif
	}
}

void CUDASolver::buildVariablesToCorrespondencesTable(EntryJ* d_correspondences, unsigned int numberOfCorrespondences)
{
	cutilSafeCall(cudaMemset(d_numEntriesPerRow, 0, sizeof(int)*m_maxNumberOfImages));

	if (numberOfCorrespondences > 0)
		buildVariablesToCorrespondencesTableCUDA(d_correspondences, numberOfCorrespondences, m_maxCorrPerImage, d_variablesToCorrespondences, d_numEntriesPerRow, m_timer);
}

////not squared (per axis component)
////#define MAX_RESIDUAL_THRESH 0.16f //sun3d
//#define MAX_RESIDUAL_THRESH 0.08f //0.05f 

#ifdef NEW_GUIDED_REMOVE
#define GUIDED_SEARCH_MAX_RES_THRESH 0.2f // threshold to start searching for other image pairs to invalidate
template<>
struct std::hash<ml::vec2ui> : public std::unary_function < ml::vec2ui, size_t > {
	size_t operator()(const ml::vec2ui& v) const {
		//TODO larger prime number (64 bit) to match size_t
		const size_t p0 = 73856093;
		const size_t p1 = 19349669;
		//const size_t p2 = 83492791;
		const size_t res = ((size_t)v.x * p0) ^ ((size_t)v.y * p1);// ^ ((size_t)v.z * p2);
		return res;
	}
};
#endif

void CUDASolver::computeMaxResidual(SolverInput& solverInput, SolverParameters& parameters, unsigned int revalidateIdx)
{
	if (m_timer) m_timer->startEvent(__FUNCTION__);
	if (parameters.weightSparse > 0.0f) {
		evalMaxResidual(solverInput, m_solverState, m_solverExtra, parameters, NULL);//m_timer);
		// copy to cpu
		unsigned int n = (solverInput.numberOfCorrespondences + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		cutilSafeCall(cudaMemcpy(m_solverExtra.h_maxResidual, m_solverExtra.d_maxResidual, sizeof(float) * n, cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(m_solverExtra.h_maxResidualIndex, m_solverExtra.d_maxResidualIndex, sizeof(int) * n, cudaMemcpyDeviceToHost));
		// compute max
		float maxResidual = 0.0f; int maxResidualIndex = 0;
		for (unsigned int i = 0; i < n; i++) {
			if (maxResidual < m_solverExtra.h_maxResidual[i]) {
				maxResidual = m_solverExtra.h_maxResidual[i];
				maxResidualIndex = m_solverExtra.h_maxResidualIndex[i];
			}
		}
#ifdef NEW_GUIDED_REMOVE

		//if (solverInput.numberOfImages == 51) {
		//	SensorData sd; sd.loadFromFile("../data/iclnuim/aliv2.sens");
		//	std::vector<mat4f> trajectory(solverInput.numberOfImages);
		//	CheckCuda(cudaMemcpy(trajectory.data(), d_transforms, sizeof(mat4f)*trajectory.size(), cudaMemcpyDeviceToHost));
		//	sd.saveToPointCloud("debug/tmp.ply", trajectory, 0, solverInput.numberOfImages*10, 10, true);
		//	int a = 5;
		//}

		m_maxResImPairs.clear();
		if (maxResidual > GUIDED_SEARCH_MAX_RES_THRESH) {
			parameters.highResidualThresh = std::min(std::max(0.2f * maxResidual, 0.1f), 0.4f);
			collectHighResiduals(solverInput, m_solverState, m_solverExtra, parameters, m_timer);
			unsigned int highResCount;
			cutilSafeCall(cudaMemcpy(&highResCount, m_solverState.d_countHighResidual, sizeof(unsigned int), cudaMemcpyDeviceToHost));
			n = std::min(highResCount, (m_maxCorrPerImage*m_maxNumberOfImages + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
			cutilSafeCall(cudaMemcpy(m_solverExtra.h_maxResidual, m_solverExtra.d_maxResidual, sizeof(float) * n, cudaMemcpyDeviceToHost));
			cutilSafeCall(cudaMemcpy(m_solverExtra.h_maxResidualIndex, m_solverExtra.d_maxResidualIndex, sizeof(int) * n, cudaMemcpyDeviceToHost));

			if (n > 1) {
				// check high residuals with previous trajectory as reference //TODO MAKE EFFICIENT
				std::vector<float4x4> transforms(solverInput.numberOfImages);
				CheckCuda(cudaMemcpy(transforms.data(), d_transforms, sizeof(float4x4)*solverInput.numberOfImages, cudaMemcpyDeviceToHost));
				std::unordered_map<vec2ui, float> residualMap; //TODO should be something better than this... 
				std::unordered_map<vec2ui, float> allCollectedResidualMap; //debugging
				std::vector<EntryJ> corrs(n);
				for (unsigned int i = 0; i < n; i++) {
					CheckCuda(cudaMemcpy(corrs.data() + i, solverInput.d_correspondences + m_solverExtra.h_maxResidualIndex[i], sizeof(EntryJ), cudaMemcpyDeviceToHost));
					const EntryJ& h_corr = corrs[i];
					vec2ui imageIndices(h_corr.imgIdx_i, h_corr.imgIdx_j);
					//compute res at previous
					if (h_corr.imgIdx_j == solverInput.numberOfImages - 1 && std::abs((int)h_corr.imgIdx_i - (int)h_corr.imgIdx_j) > 10) { //introduced by latest image
						float3 prevRes = fabs(transforms[h_corr.imgIdx_i] * h_corr.pos_i - transforms[h_corr.imgIdx_j] * h_corr.pos_j); //eval new corrs with previous trajectory
						float prevMaxRes = fmaxf(prevRes.z, fmaxf(prevRes.x, prevRes.y));
						if (prevMaxRes > 1.5f*m_solverExtra.h_maxResidual[i]) {
							auto it = residualMap.find(imageIndices);
							if (it == residualMap.end()) residualMap[imageIndices] = m_solverExtra.h_maxResidual[i];
							else it->second = std::max(m_solverExtra.h_maxResidual[i], it->second);
						}
					}
					else if (h_corr.imgIdx_j == revalidateIdx && std::abs((int)h_corr.imgIdx_i - (int)h_corr.imgIdx_j) > 10) { //introduced by latest revalidate
						auto it = residualMap.find(imageIndices);
						if (it == residualMap.end()) residualMap[imageIndices] = m_solverExtra.h_maxResidual[i];
						else it->second = std::max(m_solverExtra.h_maxResidual[i], it->second);
					}
					auto it = allCollectedResidualMap.find(imageIndices);
					if (it == allCollectedResidualMap.end()) allCollectedResidualMap[imageIndices] = m_solverExtra.h_maxResidual[i];
					else it->second = std::max(m_solverExtra.h_maxResidual[i], it->second);
				}
				if (!residualMap.empty()) { //debug print
					unsigned int rep = residualMap.begin()->first.x;
					std::cout << "rep: (" << rep << ", " << solverInput.numberOfImages - 1 << ")" << std::endl;
					for (const auto& r : residualMap) m_maxResImPairs.push_back(r.first);

					////one extra solve
					//parameters.nNonLinearIterations = 1;
					//solveBundlingStub(solverInput, m_solverState, parameters, m_solverExtra, NULL, m_timer);

					////!!!debugging
					//{
					//	static SensorData sd;
					//	if (sd.m_frames.empty()) sd.loadFromFile("../data/iclnuim/aliv2.sens");
					//	std::vector<mat4f> trajectory(solverInput.numberOfImages);
					//	CheckCuda(cudaMemcpy(trajectory.data(), d_transforms, sizeof(mat4f)*trajectory.size(), cudaMemcpyDeviceToHost));
					//	sd.saveToPointCloud("debug/tmp/" + std::to_string(solverInput.numberOfImages) + "-init.ply", trajectory, 0, solverInput.numberOfImages*10, 10, true);
					//	convertLiePosesToMatricesCU(m_solverState.d_xRot, m_solverState.d_xTrans, solverInput.numberOfImages, d_transforms, m_solverState.d_xTransformInverses);
					//	CheckCuda(cudaMemcpy(trajectory.data(), d_transforms, sizeof(mat4f)*trajectory.size(), cudaMemcpyDeviceToHost));
					//	sd.saveToPointCloud("debug/tmp/" + std::to_string(solverInput.numberOfImages) + "-opt.ply", trajectory, 0, solverInput.numberOfImages*10, 10, true);
					//	int a = 5;
					//}
					////!!!debugging
				}

				//!!!debugging
				//std::vector<std::pair<vec2ui, float>> residuals(allCollectedResidualsMap.begin(), allCollectedResidualsMap.end());
				//std::sort(residuals.begin(), residuals.end(), [](const std::pair<vec2ui, float> &left, const std::pair<vec2ui, float> &right) { //debugging only
				//	return left.second > right.second;
				//});
				//if (m_maxResImPairs.size() > 1) {
				//	std::ofstream s("debug/_logs/" + std::to_string(solverInput.numberOfImages) + "_" + std::to_string(m_maxResImPairs.front().x) + "-" + std::to_string(m_maxResImPairs.front().y) + ".txt");
				//	s << "# im pairs to remove = " << m_maxResImPairs.size() << ", res thresh = " << parameters.highResidualThresh << std::endl;
				//	for (unsigned int i = 0; i < m_maxResImPairs.size(); i++) s << m_maxResImPairs[i] << std::endl;
				//	s.close();
				//}
				//!!!debugging
			}
		}
#endif
		m_solverExtra.h_maxResidual[0] = maxResidual;
		m_solverExtra.h_maxResidualIndex[0] = maxResidualIndex;
	}
	else {
		m_solverExtra.h_maxResidual[0] = 0.0f;
		m_solverExtra.h_maxResidualIndex[0] = 0;
	}
	if (m_timer) m_timer->endEvent();
}

bool CUDASolver::getMaxResidual(unsigned int curFrame, EntryJ* d_correspondences, vec2ui& imageIndices, float& maxRes)
{
	maxRes = m_solverExtra.h_maxResidual[0];

	// for debugging get image indices regardless
	EntryJ h_corr;
	unsigned int imIdx = m_solverExtra.h_maxResidualIndex[0];
	cutilSafeCall(cudaMemcpy(&h_corr, d_correspondences + imIdx, sizeof(EntryJ), cudaMemcpyDeviceToHost));
	imageIndices.x  = h_corr.imgIdx_i ;
	imageIndices.y  = h_corr.imgIdx_j ;

	//vec2ui(h_corr.imgIdx_i, h_corr.imgIdx_j);

	bool remove = false;
	//const float curThresh = (imageIndices.y == curFrame) ? m_maxResidualThresh : m_maxResidualThresh * 2.0f; //TODO try this out
	const float curThresh = m_maxResidualThresh;
	if (!(imageIndices.x == 0 && imageIndices.y < 10) && m_solverExtra.h_maxResidual[0] > curThresh) remove = true; //don't remove the first frame

	//!!!debugging //TODO REMOVE THIS
	if (m_solverExtra.h_maxResidual[0] > curThresh && imageIndices.x == 0 && imageIndices.y < 10) {
		std::cout << "warning! max residual would invalidate images. image indices missing todo look old code " <<  " (" << m_solverExtra.h_maxResidual[0] << ")" << std::endl;
		//getchar();
	}
	//!!!debugging

	return remove;
}

//never called
bool CUDASolver::useVerification(EntryJ* d_correspondences, unsigned int numberOfCorrespondences)
{
	SolverParameters parameters;
	parameters.nNonLinearIterations = 0;
	parameters.nLinIterations = 0;
	parameters.verifyOptDistThresh = m_verifyOptDistThresh;
	parameters.verifyOptPercentThresh = m_verifyOptPercentThresh;

	SolverInput solverInput;
	solverInput.d_correspondences = d_correspondences;
	solverInput.d_variablesToCorrespondences = NULL;
	solverInput.d_numEntriesPerRow = NULL;
	solverInput.numberOfImages = 0;
	solverInput.numberOfCorrespondences = numberOfCorrespondences;

	solverInput.maxNumberOfImages = m_maxNumberOfImages;
	solverInput.maxCorrPerImage = m_maxCorrPerImage;

	unsigned int numHighResiduals = countHighResiduals(solverInput, m_solverState, parameters, m_timer);
	//std::cout << "\t[ useVerification ] " << numHighResiduals << " / " << solverInput.numberOfCorrespondences << " = " << (float)numHighResiduals / solverInput.numberOfCorrespondences << " vs " << parameters.verifyOptPercentThresh << std::endl;
	if ((float)numHighResiduals / solverInput.numberOfCorrespondences >= parameters.verifyOptPercentThresh) return true;
	return false;
}
