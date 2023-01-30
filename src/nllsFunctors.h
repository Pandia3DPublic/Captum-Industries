#pragma once
#include "core/Chunk.h"
#include <Eigen/Dense>
#include <vector>
#include "core/Model.h"
#include "utils/matrixutil.h"
#include "ceres/ceres.h"


//todo why is this only 30% faster than transforming the whole keypoint pcd and using filteredmatchees? Should be way faster!!!!!
//news: this is way faster since validkeypoints was changed. no idea why
//note: If I get tired of the stupid redundant code, just rename chunks and frames into content and put it in optimizable
struct nccf2chunk {
	nccf2chunk(Chunk* chunkpointer) : cp(chunkpointer) {}

	bool operator()(double const* const* x, double* residuals) const { 
		std::vector<std::vector<Eigen::Vector4d>> kppcds; //vector containing copies of keypoint vectors
		kppcds.reserve(cp->frames.size()); //todo maybe use pragma for here
		//takes 2-3us
		for (int i = 0; i < cp->frames.size(); i++) {
			kppcds.push_back(std::vector<Eigen::Vector4d>());
			kppcds[i].reserve(cp->frames[i]->efficientKeypoints.size());
			for (int j = 0; j < cp->frames[i]->efficientKeypoints.size(); j++) {
				kppcds[i].push_back(cp->frames[i]->efficientKeypoints[j].p);
			}
		}
		//apply all transformations takes approx 2us
		for (int i = 0; i < cp->frames.size() - 1; i++) {
			auto tmp = getT22(*x + 6 * i);
			for (auto& k : kppcds[i + 1]) {
				k = tmp * k; //transform all points to their new world coords
			}
		}

		//takes 1 us
		int count = 0;
		for (auto& v : cp->efficientMatches) {
			residuals[count++] = kppcds[v.fi1][v.i1](0) - kppcds[v.fi2][v.i2](0);
			residuals[count++] = kppcds[v.fi1][v.i1](1) - kppcds[v.fi2][v.i2](1);
			residuals[count++] = kppcds[v.fi1][v.i1](2) - kppcds[v.fi2][v.i2](2);
		}

		return true;
	}
	//creator function
	static ceres::DynamicNumericDiffCostFunction<nccf2chunk>* create(Chunk* chunkpointer) {
		return (new ceres::DynamicNumericDiffCostFunction<nccf2chunk>(new nccf2chunk(chunkpointer)));
	}

	Chunk* cp;
};

//analytic cost funtor
//todo check if sparse makes sense (prob not)
//todo get working. Something is still wrong
class SparseChunkAna : public ceres::CostFunction {
public:
	SparseChunkAna(Chunk* chunkpointer) : cp(chunkpointer) {
		nfree = 6 * (cp->frames.size() - 1);
		//set parameters for ceres. We only have one parameter block with all dofs
		mutable_parameter_block_sizes()->push_back(nfree);
		set_num_residuals(3* cp->efficientMatches.size());
	};
	virtual ~SparseChunkAna() {};
	virtual bool Evaluate(double const* const* x, double* residuals, double** jacobians) const {

		//copy all keypoints of all frames into kppcds
		std::vector<std::vector<Eigen::Vector4d>> kppcds; //vector containing copies of keypoint vectors
		kppcds.reserve(cp->frames.size());
		for (int i = 0; i < cp->frames.size(); i++) {
			kppcds.push_back(std::vector<Eigen::Vector4d>());
			kppcds[i].reserve(cp->frames[i]->efficientKeypoints.size());
			//todo try to use memcpy here for max speed (minor), or maybe better insert  (https://stackoverflow.com/questions/259297/how-do-you-copy-the-contents-of-an-array-to-a-stdvector-in-c-without-looping)
			for (int j = 0; j < cp->frames[i]->efficientKeypoints.size(); j++) {
				kppcds[i].push_back(cp->frames[i]->efficientKeypoints[j].p);
			}
		}

		//if (cp->output) {
		//	cout << "ana dump ############################ \n";
		//	int nfree = 6 * (cp->frames.size() - 1); // number of free variables
		//	for (int i = 0; i < nfree; i++) {
		//		cout << x[0][i] << endl;
		//	}
		//	cp->output = false;
		//}

		//apply all transformations
		for (int i = 0; i < (int)cp->frames.size() - 1; i++) {
			auto tmp = getT(*x + 6 * i);
			for (auto& k : kppcds[i + 1]) {
				k = tmp * k; //transform all points to their new world coords
			}
		}


		//calculate residuals 
		int count = 0; //this gives the current residual index and also the jacobian row
		for (auto& v : cp->efficientMatches) {
			auto& pa = kppcds[v.fi1][v.i1];
			auto& pb = kppcds[v.fi2][v.i2];
			residuals[count++] = pa(0) - pb(0);
			residuals[count++] = pa(1) - pb(1);
			residuals[count++] = pa(2) - pb(2);
		}

		//checks are necessary because stupid ceres will input nullptr for jacobians in case they are not needed.
		if (!jacobians) return true;
		double* jacobian = jacobians[0]; //necessary for stupid block structure of ceres. We just have one block.
		if (!jacobian) return true;
		//calculate jacobian here
		//The derivates with respect to all dofs are written in cols. 
		//Each row correpsponds to one residual
		//For ceres we need to map this to a one-d array according to costfuntion documentation
		//Do Eigen matrix for warmup
		Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(3*cp->efficientMatches.size(), nfree); //allocated
		//precalculations
		//each iteration is three rows in jacobian

		count = 0; //this gives the current residual index and also the jacobian row
		for (auto& v : cp->efficientMatches) {
			auto& pa = kppcds[v.fi1][v.i1];
			auto& pb = kppcds[v.fi2][v.i2];
			//do jac here for residuals
			int a = 6 * (v.fi1 - 1); // jac col index of frame a alpha
			int b = 6 * (v.fi2 - 1);
			const double* xa = x[0] + a; //current xa cxa[0] is alpha, cxa[1] beta and so forth
			const double* xb = x[0] + b; //current xb cxb[0] is alpha, cxb[1] beta and so forth
			//todo do better precomputation outside the loop
			//Note: The jacobian here is almost certainly correct
			if (v.fi1 != 0) {
				//################################################Frame A residuals##############################################################
				//######################################################x-residual derivatives###################################################
				//derivates of x-component here
				jac(count, a) = pa(1) * (sin(xa[0]) * sin(xa[2]) + cos(xa[0]) * cos(xa[2]) * sin(xa[1])) + pa(2) * (cos(xa[0]) * sin(xa[2]) - cos(xa[2]) * sin(xa[0]) * sin(xa[1]));
				jac(count, a + 1) = -pa(0) * cos(xa[2]) * sin(xa[1]) + pa(2) * cos(xa[0]) * cos(xa[1]) * cos(xa[2]) + pa(1) * cos(xa[1]) * cos(xa[2]) * sin(xa[0]);
				jac(count, a + 2) = -pa(1)*(cos(xa[0])*cos(xa[2])+sin(xa[0])*sin(xa[1])*sin(xa[2]))+pa(2)*(cos(xa[2])*sin(xa[0])-cos(xa[0])*sin(xa[1])*sin(xa[2]))-pa(0)*cos(xa[1])*sin(xa[2]);
				jac(count, a + 3) = 1.0;
				count++;
				//derivatives with respect to dy and dz are zero
				//################################################# y-residual derivatives###################################################
				//derivates of y-component here
				jac(count, a) = -pa(1) * (cos(xa[2]) * sin(xa[0]) - cos(xa[0]) * sin(xa[1]) * sin(xa[2])) - pa(2) * (cos(xa[0]) * cos(xa[2]) + sin(xa[0]) * sin(xa[1]) * sin(xa[2]));
				jac(count, a + 1) = -pa(0) * sin(xa[1]) * sin(xa[2]) + pa(2) * cos(xa[0]) * cos(xa[1]) * sin(xa[2]) + pa(1) * cos(xa[1]) * sin(xa[0]) * sin(xa[2]);
				jac(count, a + 2) = -pa(1) * (cos(xa[0]) * sin(xa[2]) - cos(xa[2]) * sin(xa[0]) * sin(xa[1])) + pa(2) * (sin(xa[0]) * sin(xa[2]) + cos(xa[0]) * cos(xa[2]) * sin(xa[1])) + pa(0) * cos(xa[1]) * cos(xa[2]);
				jac(count, a + 4) = 1.0;
				count++;
				//derivatives with respect to dx and dz are zero
				//#################################################z-residual derivatives###################################################
				jac(count, a) = pa(1) * cos(xa[0]) * cos(xa[1]) - pa(2) * cos(xa[1]) * sin(xa[0]); 
				jac(count, a + 1) = -pa(0) * cos(xa[1]) - pa(2) * cos(xa[0]) * sin(xa[1]) - pa(1) * sin(xa[0]) * sin(xa[1]); 
				jac(count, a + 5) = 1.0;
				count++;
				//rest of derivatives is zero (weird for gamma component)
				if (v.fi2 != 0) {
					count = count - 3; //reset counter
				}
			}
			if (v.fi2 != 0) {
				//################################################Frame B residuals##############################################################
				//minus signs important here
				//######################################################x-residual derivatives###################################################
				jac(count, b) = -pb(1) * (sin(xb[0]) * sin(xb[2]) + cos(xb[0]) * cos(xb[2]) * sin(xb[1])) - pb(2) * (cos(xb[0]) * sin(xb[2]) - cos(xb[2]) * sin(xb[0]) * sin(xb[1]));
				jac(count, b + 1) = pb(0) * cos(xb[2]) * sin(xb[1]) - pb(2) * cos(xb[0]) * cos(xb[1]) * cos(xb[2]) - pb(1) * cos(xb[1]) * cos(xb[2]) * sin(xb[0]);
				jac(count, b + 2) = pb(1) * (cos(xb[0]) * cos(xb[2]) + sin(xb[0]) * sin(xb[1]) * sin(xb[2])) - pb(2) * (cos(xb[2]) * sin(xb[0]) - cos(xb[0]) * sin(xb[1]) * sin(xb[2])) + pb(0) * cos(xb[1]) * sin(xb[2]);
				jac(count, b + 3) = -1.0;
				count++;
				//################################################# y-residual derivatives###################################################
				jac(count, b) = pb(1) * (cos(xb[2]) * sin(xb[0]) - cos(xb[0]) * sin(xb[1]) * sin(xb[2])) + pb(2) * (cos(xb[0]) * cos(xb[2]) + sin(xb[0]) * sin(xb[1]) * sin(xb[2]));
				jac(count, b + 1) = pb(0) * sin(xb[1]) * sin(xb[2]) - pb(2) * cos(xb[0]) * cos(xb[1]) * sin(xb[2]) - pb(1) * cos(xb[1]) * sin(xb[0]) * sin(xb[2]);
				jac(count, b + 2) = pb(1) * (cos(xb[0]) * sin(xb[2]) - cos(xb[2]) * sin(xb[0]) * sin(xb[1])) - pb(2) * (sin(xb[0]) * sin(xb[2]) + cos(xb[0]) * cos(xb[2]) * sin(xb[1])) - pb(0) * cos(xb[1]) * cos(xb[2]); 
				jac(count, b + 4) = -1.0;
				count++;
				//#################################################z-residual derivatives###################################################
				jac(count, b) = -pb(1) * cos(xb[0]) * cos(xb[1]) + pb(2) * cos(xb[1]) * sin(xb[0]); 
				jac(count, b + 1) = pb(0) * cos(xb[1]) + pb(2) * cos(xb[0]) * sin(xb[1]) + pb(1) * sin(xb[0]) * sin(xb[1]); 
				jac(count, b + 5) = -1.0;
				count++;
			}

		}

		//now map jac matrix to one-d array
		//jacobians[i][r * parameter_block_sizes_[i] + c]
		for (int i = 0; i < count; i++) {
			for (int j = 0; j < nfree; j++) {
				jacobian[i * nfree + j] = jac(i, j);
			}
		}
		return true;
	}

	Chunk* cp;
	int nfree;
};

//news: this is way faster since validkeypoints was changed. no idea why
struct nccf2model {
	nccf2model(Model* modelpointer) : cp(modelpointer) {}

	bool operator()(double const* const* x, double* residuals) const { 
		std::vector<std::vector<Eigen::Vector4d>> kppcds; //vector containing copies of keypoint vectors
		kppcds.reserve(cp->chunks.size());

		for (int i = 0; i < cp->chunks.size(); i++) {
			kppcds.push_back(std::vector<Eigen::Vector4d>());
			kppcds[i].reserve(cp->chunks[i]->efficientKeypoints.size());
			for (int j = 0; j < cp->chunks[i]->efficientKeypoints.size(); j++) {
				kppcds[i].push_back(cp->chunks[i]->efficientKeypoints[j].p);
			}
		}

		//apply all transformations
		for (int i = 0; i < cp->chunks.size() - 1; i++) {
			auto tmp = getT22(*x + 6 * i);
			for (auto& k : kppcds[i + 1]) {
				k = tmp * k; //transform all points to their new world coords
			}
		}

		int count = 0;
		for (auto& v : cp->efficientMatches) {
			residuals[count++] = kppcds[v.fi1][v.i1](0) - kppcds[v.fi2][v.i2](0);
			residuals[count++] = kppcds[v.fi1][v.i1](1) - kppcds[v.fi2][v.i2](1);
			residuals[count++] = kppcds[v.fi1][v.i1](2) - kppcds[v.fi2][v.i2](2);
		}

		return true;
	}
	//creator function
	static ceres::DynamicNumericDiffCostFunction<nccf2model>* create(Model* modelpointer) {
		return (new ceres::DynamicNumericDiffCostFunction<nccf2model>(new nccf2model(modelpointer))); //todo check if this gets deleted
	}

	Model* cp;
};


//analytic cost funtor
//todo check if sparse makes sense (prob not)
class SparseModelAna : public ceres::CostFunction {
public:
	SparseModelAna(Model* modelpointer) : mp(modelpointer) {
		nfree = 6 * (mp->chunks.size() - 1);
		//set parameters for ceres. We only have one parameter block with all dofs
		mutable_parameter_block_sizes()->push_back(nfree);
		set_num_residuals(3 * mp->efficientMatches.size());
	};
	virtual ~SparseModelAna() {};
	virtual bool Evaluate(double const* const* x, double* residuals, double** jacobians) const {

		//copy all keypoints of all frames into kppcds
		std::vector<std::vector<Eigen::Vector4d>> kppcds; //vector containing copies of keypoint vectors
		kppcds.reserve(mp->chunks.size());
		for (int i = 0; i < mp->chunks.size(); i++) {
			kppcds.push_back(std::vector<Eigen::Vector4d>());
			kppcds[i].reserve(mp->chunks[i]->efficientKeypoints.size());
			//todo try to use memcpy here for max speed (minor), or maybe better insert  (https://stackoverflow.com/questions/259297/how-do-you-copy-the-contents-of-an-array-to-a-stdvector-in-c-without-looping)
			for (int j = 0; j < mp->chunks[i]->efficientKeypoints.size(); j++) {
				kppcds[i].push_back(mp->chunks[i]->efficientKeypoints[j].p);
			}
		}
		//apply all transformations
		for (int i = 0; i < (int)mp->chunks.size() - 1; i++) {
			auto tmp = getT22(*x + 6 * i);
			for (auto& k : kppcds[i + 1]) {
				k = tmp * k; //transform all points to their new world coords
			}
		}


		//calculate residuals 
		int count = 0; //this gives the current residual index and also the jacobian row
		for (auto& v : mp->efficientMatches) {
			auto& pa = kppcds[v.fi1][v.i1];
			auto& pb = kppcds[v.fi2][v.i2];
			residuals[count++] = pa(0) - pb(0);
			residuals[count++] = pa(1) - pb(1);
			residuals[count++] = pa(2) - pb(2);
		}

		//checks are necessary because stupid ceres will input nullptr for jacobians in case they are not needed.
		if (!jacobians) return true;
		double* jacobian = jacobians[0]; //necessary for stupid block structure of ceres. We just have one block.
		if (!jacobian) return true;
		//calculate jacobian here
		//The derivates with respect to all dofs are written in cols. 
		//Each row correpsponds to one residual
		//For ceres we need to map this to a one-d array according to costfuntion documentation
		//Do Eigen matrix for warmup
		Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(3 * mp->efficientMatches.size(), nfree); //allocated
		//precalculations
		//each iteration is three rows in jacobian

		count = 0; //this gives the current residual index and also the jacobian row
		for (auto& v : mp->efficientMatches) {
			auto& pa = kppcds[v.fi1][v.i1];
			auto& pb = kppcds[v.fi2][v.i2];
			//do jac here for residuals
			int a = 6 * (v.fi1 - 1); // jac col index of frame a alpha
			int b = 6 * (v.fi2 - 1);
			const double* xa = x[0] + a; //current xa cxa[0] is alpha, cxa[1] beta and so forth
			const double* xb = x[0] + b; //current xb cxb[0] is alpha, cxb[1] beta and so forth
			//todo do better precomputation outside the loop
			//Note: The jacobian here is almost certainly correct
			if (v.fi1 != 0) {
				//################################################Frame A residuals##############################################################
				//######################################################x-residual derivatives###################################################
				//derivates of x-component here
				jac(count, a) = pa(1) * (sin(xa[0]) * sin(xa[2]) + cos(xa[0]) * cos(xa[2]) * sin(xa[1])) + pa(2) * (cos(xa[0]) * sin(xa[2]) - cos(xa[2]) * sin(xa[0]) * sin(xa[1]));
				jac(count, a + 1) = -pa(0) * cos(xa[2]) * sin(xa[1]) + pa(2) * cos(xa[0]) * cos(xa[1]) * cos(xa[2]) + pa(1) * cos(xa[1]) * cos(xa[2]) * sin(xa[0]);
				jac(count, a + 2) = -pa(1) * (cos(xa[0]) * cos(xa[2]) + sin(xa[0]) * sin(xa[1]) * sin(xa[2])) + pa(2) * (cos(xa[2]) * sin(xa[0]) - cos(xa[0]) * sin(xa[1]) * sin(xa[2])) - pa(0) * cos(xa[1]) * sin(xa[2]);
				jac(count, a + 3) = 1.0;
				count++;
				//derivatives with respect to dy and dz are zero
				//################################################# y-residual derivatives###################################################
				//derivates of y-component here
				jac(count, a) = -pa(1) * (cos(xa[2]) * sin(xa[0]) - cos(xa[0]) * sin(xa[1]) * sin(xa[2])) - pa(2) * (cos(xa[0]) * cos(xa[2]) + sin(xa[0]) * sin(xa[1]) * sin(xa[2]));
				jac(count, a + 1) = -pa(0) * sin(xa[1]) * sin(xa[2]) + pa(2) * cos(xa[0]) * cos(xa[1]) * sin(xa[2]) + pa(1) * cos(xa[1]) * sin(xa[0]) * sin(xa[2]);
				jac(count, a + 2) = -pa(1) * (cos(xa[0]) * sin(xa[2]) - cos(xa[2]) * sin(xa[0]) * sin(xa[1])) + pa(2) * (sin(xa[0]) * sin(xa[2]) + cos(xa[0]) * cos(xa[2]) * sin(xa[1])) + pa(0) * cos(xa[1]) * cos(xa[2]);
				jac(count, a + 4) = 1.0;
				count++;
				//derivatives with respect to dx and dz are zero
				//#################################################z-residual derivatives###################################################
				jac(count, a) = pa(1) * cos(xa[0]) * cos(xa[1]) - pa(2) * cos(xa[1]) * sin(xa[0]);
				jac(count, a + 1) = -pa(0) * cos(xa[1]) - pa(2) * cos(xa[0]) * sin(xa[1]) - pa(1) * sin(xa[0]) * sin(xa[1]);
				jac(count, a + 5) = 1.0;
				count++;
				//rest of derivatives is zero (weird for gamma component)
				if (v.fi2 != 0) {
					count = count - 3; //reset counter
				}
			}
			if (v.fi2 != 0) {
				//################################################Frame B residuals##############################################################
				//minus signs important here
				//######################################################x-residual derivatives###################################################
				jac(count, b) = -pb(1) * (sin(xb[0]) * sin(xb[2]) + cos(xb[0]) * cos(xb[2]) * sin(xb[1])) - pb(2) * (cos(xb[0]) * sin(xb[2]) - cos(xb[2]) * sin(xb[0]) * sin(xb[1]));
				jac(count, b + 1) = pb(0) * cos(xb[2]) * sin(xb[1]) - pb(2) * cos(xb[0]) * cos(xb[1]) * cos(xb[2]) - pb(1) * cos(xb[1]) * cos(xb[2]) * sin(xb[0]);
				jac(count, b + 2) = pb(1) * (cos(xb[0]) * cos(xb[2]) + sin(xb[0]) * sin(xb[1]) * sin(xb[2])) - pb(2) * (cos(xb[2]) * sin(xb[0]) - cos(xb[0]) * sin(xb[1]) * sin(xb[2])) + pb(0) * cos(xb[1]) * sin(xb[2]);
				jac(count, b + 3) = -1.0;
				count++;
				//################################################# y-residual derivatives###################################################
				jac(count, b) = pb(1) * (cos(xb[2]) * sin(xb[0]) - cos(xb[0]) * sin(xb[1]) * sin(xb[2])) + pb(2) * (cos(xb[0]) * cos(xb[2]) + sin(xb[0]) * sin(xb[1]) * sin(xb[2]));
				jac(count, b + 1) = pb(0) * sin(xb[1]) * sin(xb[2]) - pb(2) * cos(xb[0]) * cos(xb[1]) * sin(xb[2]) - pb(1) * cos(xb[1]) * sin(xb[0]) * sin(xb[2]);
				jac(count, b + 2) = pb(1) * (cos(xb[0]) * sin(xb[2]) - cos(xb[2]) * sin(xb[0]) * sin(xb[1])) - pb(2) * (sin(xb[0]) * sin(xb[2]) + cos(xb[0]) * cos(xb[2]) * sin(xb[1])) - pb(0) * cos(xb[1]) * cos(xb[2]);
				jac(count, b + 4) = -1.0;
				count++;
				//#################################################z-residual derivatives###################################################
				jac(count, b) = -pb(1) * cos(xb[0]) * cos(xb[1]) + pb(2) * cos(xb[1]) * sin(xb[0]);
				jac(count, b + 1) = pb(0) * cos(xb[1]) + pb(2) * cos(xb[0]) * sin(xb[1]) + pb(1) * sin(xb[0]) * sin(xb[1]);
				jac(count, b + 5) = -1.0;
				count++;
			}

		}

		//now map jac matrix to one-d array
		//jacobians[i][r * parameter_block_sizes_[i] + c]
		for (int i = 0; i < count; i++) {
			for (int j = 0; j < nfree; j++) {
				jacobian[i * nfree + j] = jac(i, j);
			}
		}
		return true;
	}

	Model* mp;
	int nfree;
};



//accf2 automatic derivatives for chunk
struct accf2 {
	accf2(Chunk* chunkpointer) : cp(chunkpointer) {}

	template<typename T>
	bool operator()(T const* const* x, T* residuals) const { //residuals has size 3
		std::vector<std::vector<Eigen::Matrix<T, 4, 1>>> kppcds; //vector containing copies of keypoint vectors
		kppcds.reserve(cp->frames.size());
		//if (is_same<T, double>::value) {
		//	cout << "double \n";
		//} else {
		//	cout << "jet \n";
		//}

		//takes 6-8 ms (little to no difference for double and jet. jet has outliers at 20us)
		for (int i = 0; i < cp->frames.size(); i++) {
			kppcds.push_back(std::vector<Eigen::Matrix<T, 4, 1>>());
			kppcds[i].reserve(cp->frames[i]->efficientKeypoints.size());
			for (int j = 0; j < cp->frames[i]->efficientKeypoints.size(); j++) {
				Eigen::Matrix<T, 4, 1> tmpv; //todo find out if this is stable (allocated)
				tmpv(0, 0) = T(cp->frames[i]->efficientKeypoints[j].p(0));
				tmpv(1, 0) = T(cp->frames[i]->efficientKeypoints[j].p(1));
				tmpv(2, 0) = T(cp->frames[i]->efficientKeypoints[j].p(2));
				tmpv(3, 0) = T(1.0);
				kppcds[i].push_back(tmpv);
			}
		}

		//apply all transformations. Takes 2um for double and 60um for jet
		for (int i = 0; i < cp->frames.size() - 1; i++) {
			auto tmp = getT22(*x + 6 * i);
			for (auto& k : kppcds[i + 1]) {
				k = tmp * k; //transform all points to their new world coords
			}
		}
		//takes 1um for double and 6-7um for jet 
		int count = 0;
		for (auto& v : cp->efficientMatches) {
			residuals[count++] = kppcds[v.fi1][v.i1](0) - kppcds[v.fi2][v.i2](0);
			residuals[count++] = kppcds[v.fi1][v.i1](1) - kppcds[v.fi2][v.i2](1);
			residuals[count++] = kppcds[v.fi1][v.i1](2) - kppcds[v.fi2][v.i2](2);
		}

		return true;

	}
	//creator function
	static ceres::DynamicAutoDiffCostFunction<accf2>* create(Chunk * chunkpointer) {
		return (new ceres::DynamicAutoDiffCostFunction<accf2>(new accf2(chunkpointer)));
	}

	Chunk* cp;
};


//accf2 automatic derivatives for model
struct amf2 {
	amf2(Model* modelpointer) : mp(modelpointer) {}

	template<typename T>
	bool operator()(T const* const* x, T* residuals) const { //residuals has size 3
		std::vector<std::vector<Eigen::Matrix<T, 4, 1>>> kppcds; //vector containing copies of keypoint vectors
		kppcds.reserve(mp->chunks.size());
		//takes 6-8 ms (little to no difference for double and jet. jet has outliers at 20us)
		for (int i = 0; i < mp->chunks.size(); i++) {
			kppcds.push_back(std::vector<Eigen::Matrix<T, 4, 1>>());
			kppcds[i].reserve(mp->chunks[i]->efficientKeypoints.size());
			for (int j = 0; j < mp->chunks[i]->efficientKeypoints.size(); j++) {
				Eigen::Matrix<T, 4, 1> tmpv; //todo find out if this is stable (allocated)
				tmpv(0, 0) = T(mp->chunks[i]->efficientKeypoints[j].p(0));
				tmpv(1, 0) = T(mp->chunks[i]->efficientKeypoints[j].p(1));
				tmpv(2, 0) = T(mp->chunks[i]->efficientKeypoints[j].p(2));
				tmpv(3, 0) = T(1.0);
				kppcds[i].push_back(tmpv);
			}
		}

		//apply all transformations. Takes 2um for double and 60um for jet
		for (int i = 0; i < mp->chunks.size() - 1; i++) {
			auto tmp = getT22(*x + 6 * i);
			for (auto& k : kppcds[i + 1]) {
				k = tmp * k; //transform all points to their new world coords
			}
		}
		//takes 1um for double and 6-7um for jet 
		int count = 0;
		for (auto& v : mp->efficientMatches) {
			residuals[count++] = kppcds[v.fi1][v.i1](0) - kppcds[v.fi2][v.i2](0);
			residuals[count++] = kppcds[v.fi1][v.i1](1) - kppcds[v.fi2][v.i2](1);
			residuals[count++] = kppcds[v.fi1][v.i1](2) - kppcds[v.fi2][v.i2](2);
		}

		return true;

	}
	//creator function
	static ceres::DynamicAutoDiffCostFunction<amf2>* create(Model* modelpointer) {
		return (new ceres::DynamicAutoDiffCostFunction<amf2>(new amf2(modelpointer)));
	}

	Model* mp;
};



//
//
//if (v.fi1 != 0) {
//	//################################################Frame A residuals##############################################################
//	//######################################################x-residual derivatives###################################################
//	//derivates of x-component here
//	jac(count, a) = pa(1) * (sin(cxa[0]) * sin(cxa[2]) + cos(cxa[0]) * cos(cxa[2]) * sin(cxa[1])) + pa(2) * (cos(cxa[0]) * sin(cxa[2]) - cos(cxa[2]) * sin(cxa[0]) * sin(cxa[1]));
//	jac(count, a + 1) = pa(2) * cos(cxa[0]) * cos(cxa[1]) * cos(cxa[2]) - pa(0) * cos(cxa[2]) * sin(cxa[1]) + pa(1) * cos(cxa[1]) * cos(cxa[2]) * sin(cxa[0]);
//	jac(count, a + 2) = pa(2) * (cos(cxa[2]) * sin(cxa[0]) - cos(cxa[0]) * sin(cxa[1]) * sin(cxa[2])) - pa(1) * (cos(cxa[0]) * cos(cxa[2]) + sin(cxa[0]) * sin(cxa[1]) * sin(cxa[2])) - pa(0) * cos(cxa[1]) * sin(cxa[2]);
//	jac(count, a + 3) = 1;
//	count++;
//	//derivatives with respect to dy and dz are zero
//	//################################################# y-residual derivatives###################################################
//	//derivates of y-component here
//	jac(count, a) = -pa(1) * (cos(cxa[2]) * sin(cxa[0]) - cos(cxa[0]) * sin(cxa[1]) * sin(cxa[2])) - pa(2) * (cos(cxa[0]) * cos(cxa[2]) + sin(cxa[0]) * sin(cxa[1]) * sin(cxa[2]));
//	jac(count, a + 1) = pa(2) * cos(cxa[0]) * cos(cxa[1]) * sin(cxa[2]) - pa(0) * sin(cxa[1]) * sin(cxa[2]) + pa(1) * cos(cxa[1]) * sin(cxa[0]) * sin(cxa[2]);
//	jac(count, a + 2) = pa(2) * (sin(cxa[0]) * sin(cxa[2]) + cos(cxa[0]) * cos(cxa[2]) * sin(cxa[1])) - pa(1) * (cos(cxa[0]) * sin(cxa[2]) - cos(cxa[2]) * sin(cxa[0]) * sin(cxa[1])) + pa(0) * cos(cxa[1]) * cos(cxa[2]);
//	jac(count, a + 4) = 1;
//	count++;
//	//derivatives with respect to dx and dz are zero
//	//#################################################z-residual derivatives###################################################
//	jac(count, a) = pa(1) * cos(cxa[0]) * cos(cxa[1]) - pa(2) * cos(cxa[1]) * sin(cxa[0]);
//	jac(count, a + 1) = -pa(0) * cos(cxa[1]) - pa(2) * cos(cxa[0]) * sin(cxa[1]) - pa(1) * sin(cxa[0]) * sin(cxa[1]);
//	jac(count, a + 5) = 1;
//	count++;
//	//rest of derivatives is zero (weird for gamma component)
//	if (v.fi2 != 0) {
//		count = count - 3; //reset counter
//	}
//}
//if (v.fi2 != 0) {
//	//################################################Frame B residuals##############################################################
//	//minus signs important here
//	//######################################################x-residual derivatives###################################################
//	jac(count, b) = -pb(1) * (sin(cxb[0]) * sin(cxb[2]) + cos(cxb[0]) * cos(cxb[2]) * sin(cxb[1])) - pb(2) * (cos(cxb[0]) * sin(cxb[2]) - cos(cxb[2]) * sin(cxb[0]) * sin(cxb[1]));
//	jac(count, b + 1) = -pb(2) * cos(cxb[0]) * cos(cxb[1]) * cos(cxb[2]) + pb(0) * cos(cxb[2]) * sin(cxb[1]) - pb(1) * cos(cxb[1]) * cos(cxb[2]) * sin(cxb[0]);
//	jac(count, b + 2) = -pb(2) * (cos(cxb[2]) * sin(cxb[0]) - cos(cxb[0]) * sin(cxb[1]) * sin(cxb[2])) + pb(1) * (cos(cxb[0]) * cos(cxb[2]) + sin(cxb[0]) * sin(cxb[1]) * sin(cxb[2])) + pb(0) * cos(cxb[1]) * sin(cxb[2]);
//	jac(count, b + 3) = -1;
//	count++;
//	//################################################# y-residual derivatives###################################################
//	jac(count, b) = pb(1) * (cos(cxb[2]) * sin(cxb[0]) - cos(cxb[0]) * sin(cxb[1]) * sin(cxb[2])) + pb(2) * (cos(cxb[0]) * cos(cxb[2]) + sin(cxb[0]) * sin(cxb[1]) * sin(cxb[2]));
//	jac(count, b + 1) = -pb(2) * cos(cxb[0]) * cos(cxb[1]) * sin(cxb[2]) + pb(0) * sin(cxb[1]) * sin(cxb[2]) - pb(1) * cos(cxb[1]) * sin(cxb[0]) * sin(cxb[2]);
//	jac(count, b + 2) = -pb(2) * (sin(cxb[0]) * sin(cxb[2]) + cos(cxb[0]) * cos(cxb[2]) * sin(cxb[1])) + pb(1) * (cos(cxb[0]) * sin(cxb[2]) - cos(cxb[2]) * sin(cxb[0]) * sin(cxb[1])) - pb(0) * cos(cxb[1]) * cos(cxb[2]);
//	jac(count, b + 4) = -1;
//	count++;
//	//#################################################z-residual derivatives###################################################
//	jac(count, b) = -pb(1) * cos(cxb[0]) * cos(cxb[1]) + pb(2) * cos(cxb[1]) * sin(cxb[0]);
//	jac(count, b + 1) = pb(0) * cos(cxb[1]) + pb(2) * cos(cxb[0]) * sin(cxb[1]) + pb(1) * sin(cxb[0]) * sin(cxb[1]);
//	jac(count, b + 5) = -1;
//	count++;
//}