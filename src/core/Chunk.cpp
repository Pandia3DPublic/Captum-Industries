#include "Chunk.h"
#include "utils/matrixutil.h"
#include "nllsFunctors.h"
#include "configvars.h"
#include "utils/coreutil.h"

using namespace open3d;

//bool matchsort(match &i, match &j) { 
//	return i.d < j.d; 
//}


Chunk::Chunk()
{
	unique_id = chunk_id_counter;
}

Chunk::~Chunk()
{
}

void Chunk::generateFrustum() {
	double minz=2*g_cutoff;
	double maxz=0;
	for (auto& p : frames[0]->lowpcd->points_) {
		if (p(2) > maxz)
			maxz = p(2);
		if(p(2) < minz)
			minz  = p(2);
	}
	frustum = make_shared<Frustum>(Eigen::Matrix4d::Identity(), g_intrinsic, minz,maxz);
}


//add lines for all valid(used in sparse alignment) cors
void Chunk::addValidLines() {
	for (auto v: efficientMatches) {
		addLine(frames[v.fi1]->efficientKeypoints[v.i1].p.block<3,1>(0,0), frames[v.fi2]->efficientKeypoints[v.i2].p.block<3, 1>(0, 0));
	}
}

void Chunk::addLine(Eigen::Vector3d startpoint, Eigen::Vector3d endpoint){ //this does not work in some cases. Unclear why!
	ls->points_.push_back(startpoint);
	ls->points_.push_back(endpoint);
	ls->lines_.push_back(Eigen::Vector2i(ls->points_.size()-1,ls->points_.size()-2));

}

bool Chunk::doFrametoModelforNewestFrame() {
	Eigen::VectorXd xt = Eigen::Vector3d::Zero(); 
	Eigen::VectorXd tsum= Eigen::Vector3d::Zero();
	int count = 0;
	vector<Eigen::Quaterniond> quats; //quaternions for latter averagging


	auto& f= frames.back();
	int b = frames.size() - 1;
	if (b == 0 ) return true;
	for (int a = 0; a < frames.size() - 1; a++) {
		if (pairTransforms(a,b).set){
			if (!frames[a]->chunktransform.isIdentity() || a ==0) {
				count++;
				Eigen::Matrix4d pcb = frames[a]->chunktransform * pairTransforms(a,b).invkabschtrans;
				quats.push_back(Eigen::Quaterniond(pcb.block<3,3>(0,0)));
				tsum = tsum + pcb.block<3,1>(0,3);
			}

		}
	}

	if(count > 0){
		xt = tsum/count;
		Eigen::Matrix4d out;
		out.block<3,3>(0,0) = Eigen::Matrix3d(getQuaternionAverage(quats));
		out.block<3,1>(0,3) =xt;
		//does not need lock since frame cannot be in integrationbuffer here
		f->chunktransform = out;
		return true;
	} else {
		return false;
	}

}



//residual calculation
//todo use the same method for model and chunk
//todo chunk to non-dynamic cost functions with defines for chunk size
//note autodiff for chunk is slightly slowr than numeric. Fuck ceres
void Chunk::performSparseOptimization(vector<Eigen::Vector6d>& initx) {
	//Timer t("sparse chunk opt", millisecond);
	int nfree = 6 * (frames.size() - 1); // number of free variables
	double* x = new  double[nfree]; //vector with free variables on heap
	//for (int i = 0; i < nfree; i++) {
	//	x[i] = 0;
	//}
	for (int i = 0; i < frames.size() - 1; i++) {
		for (int j = 0; j < 6; j++) {
			x[6 * i + j] = initx[i](j);
		}
	}

	int nr = efficientMatches.size();
	nr = 3 * nr; //number of residuals
	std::vector<ceres::ResidualBlockId> residual_block_ids;
	ceres::Problem problem;

	//fast variant numeric
	ceres::DynamicNumericDiffCostFunction<nccf2chunk>* costfunction = nccf2chunk::create(this);
	costfunction->AddParameterBlock(nfree);
	costfunction->SetNumResiduals(nr);
	ceres::ResidualBlockId block_id = problem.AddResidualBlock(costfunction, NULL, x);

	//fast variant autodiff
	//ceres::DynamicAutoDiffCostFunction<accf2>* costfunction = accf2::create(this);
	//costfunction->AddParameterBlock(nfree);
	//costfunction->SetNumResiduals(nr);
	//problem.AddResidualBlock(costfunction, NULL, x);

	//fast variant analytic
	//ceres::CostFunction* costfunction = new SparseChunkAna(this);
	//ceres::ResidualBlockId block_id = problem.AddResidualBlock(costfunction, NULL, x);
	//residual_block_ids.push_back(block_id);




	ceres::Solver::Options options;
	options.max_num_iterations = 15;
	options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
	options.minimizer_progress_to_stdout = false;
	//options.check_gradients = true;
	//options.trust_region_problem_dump_directory = "C:\\Users\\Kosnoros\\Dropbox\\StartupTech\\Tests\\jacobians\\ana";
	//options.trust_region_minimizer_iterations_to_dump.push_back(1);
	//options.trust_region_minimizer_iterations_to_dump.push_back(2);
	options.num_threads = 8;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	//std::cout << summary.FullReport() << "\n";
	//cout << "jac eval time in model: " << summary.jacobian_evaluation_time_in_seconds << endl;

	//cout << "Successfull steps: " << summary.num_successful_steps << endl;
	//cout << "bad steps: " << summary.num_unsuccessful_steps << endl;
	utility::LogDebug("Optimized Chunk \n");
	// set transformations
	for (int i = 0; i < frames.size()-1; i++) {
		frames[i+1]->chunktransform = getT(x+6*i);
	}


	vector<double> residuals;
	double cost;
	problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residuals, NULL, NULL);
	int index = 0;
	double max = 0;
	for (int i = 0; i < residuals.size(); i++) {
		if (fabs(residuals[i]) > max) {
			max = fabs(residuals[i]);
			index = i;
		}
	}
	if (fabs(residuals[index]) > 0.05) {
		int matchindex = index/3;
		//remove matches
		filteredmatches(efficientMatches[matchindex].fi1, efficientMatches[matchindex].fi2) = vector<match>();
		generateEfficientStructures();
		utility::LogDebug("Matches removed by residual filter. Biggest residual was {} \n", residuals[index]);
		vector<Eigen::Vector6d> tmp;
		tmp.reserve(frames.size() - 1);
		for (int i = 0; i < frames.size() - 1; i++) {
			for (int j = 0; j < 6; j++) {
				tmp[i](j) = x[6 * i + j];
			}
		}
		performSparseOptimization(tmp); //use current dof for new staring point
		//performSparseOptimization();
	}


	delete[] x;

}



void Chunk::markChunkKeypoints()
{
	for (int i = 0; i < orbKeypoints.size(); i++) {
		auto mesh_ptr = geometry::TriangleMesh::CreateSphere(0.02, 10);
		mesh_ptr->Transform(gettrans(orbKeypoints[i].p.block<3, 1>(0, 0)));
		mesh_ptr->PaintUniformColor(Eigen::Vector3d(255, 0, 0));
		spheres.push_back(mesh_ptr);
	}
}


//this works as inteded.
void Chunk::getMajorityDescriptor(vector<int>& indeces,const vector<c_keypoint>& kps, vector<uint8_t>& merged)
{
	bool done = false;
	merged.reserve(g_nOrb);
	for (int i = 0; i < g_nOrb; i++) {
		merged.push_back(0);
	}
	while (!done) {
		done = true;
		vector<int> newindeces;
		for (int i = 0; i < g_nOrb; i++) {
			merged[i] =(0);
		}
		bool tmp;
		for (int i = 0; i < g_nOrb; i++) {
			for (int j = 0; j < 8; j++) {
				int count = 0;
				for (auto& ind : indeces) {
					tmp = getBit(kps[ind].des[i], j);
					count += tmp; //increment if true
				}
				if (count > indeces.size() / 2) { //this defaults to zero, if we have two points merging
					setBitOne(merged[i], j);
				}
			}
		}

		// test if each descriptor has at least 80% congruence with merged point. Otherwise remove descriptor and start over.
		for (int k = 0; k < indeces.size(); k++) { 
			int same = 0;
			for (int i = 0; i < g_nOrb; i++) {
				for (int j = 0; j < 8; j++) {
					if (getBit(kps[indeces[k]].des[i], j) == getBit(merged[i], j)) {
						same++;
					}
				}
			}
			if (same > 8*g_nOrb*0.8) { // good point over 80%
				newindeces.push_back(indeces[k]);
			} else {
				done = false;
			}
		}
		if (newindeces.size() == 0) { //if all point have been filtered ignore the result
			merged = kps[indeces[0]].des;
			break;
		}
		indeces = newindeces; // todo superflous without recursive
	}


	int same = 0;
	for (int i = 0; i < g_nOrb; i++) {
		for (int j = 0; j < 8; j++) {
			if (getBit(kps[indeces[0]].des[i], j) == getBit(merged[i], j)) {
				same++;
			}
		}
	}

}

//only call this after generating validkeypoints 
//todo evaluate iterative behaviour and radius size
//todo currently not working correctly for iterative behaviour. 
	// Add oldweights to newweights when melding points. 
	// only meld descriptors when successfull
	// take weights into account in majority vote
//for now its the non-recursive version, which has been visualy tested
void Chunk::generateChunkKeypoints(int it)
{
	if (chunktransapplied == false) {
		for (int i = 0; i < frames.size(); i++) {
			for (int j = 0; j < frames[i]->efficientKeypoints.size(); j++) {
				frames[i]->efficientKeypoints[j].transform(frames[i]->chunktransform);
			}
		}
		chunktransapplied = true;
	}

	// if its the first iteration fill keypoints with all frame points
	int n = 0;
	if (it == 1) {
		for (int i = 0; i < frames.size(); i++) {
			n += frames[i]->efficientKeypoints.size();
		}
		orbKeypoints.reserve(n); //this is a chunk variable, need to do this instead of just build pcd for recursive behaviour.
		for (int i = 0; i < frames.size(); i++) {
			for (auto& k : frames[i]->efficientKeypoints) {
				orbKeypoints.push_back(k);
			}
		}
		chunkkpweights.reserve(n);
		for (int i = 0; i <n ; i++) {
			chunkkpweights.push_back(1);
		}

	}
	//build pcd from keypoints for kdtree search
	auto pcd = make_shared<geometry::PointCloud>();
	pcd->points_.reserve(orbKeypoints.size());
	for (int i = 0; i < orbKeypoints.size(); i++) {
		pcd->points_.push_back(orbKeypoints[i].p.block<3, 1>(0, 0));
	}

	geometry::KDTreeFlann kdtree;
	kdtree.SetGeometry(*pcd);
	std::vector<int> ignore;
	int k;
	std::vector<c_keypoint> newkeypoints;
	std::vector<int> newweights;
	for (int i = 0; i < pcd->points_.size(); i++) {
		std::vector<int> indices_vec;
		std::vector<double> dists_vec;
		auto t = std::find(ignore.begin(), ignore.end(), i);
		if (t == ignore.end()) { //check if index is on the ignore list. Point is not on list
			auto& p = pcd->points_[i];
			k = kdtree.SearchRadius(p,g_mergeradius/(double)it, indices_vec, dists_vec);
			if (indices_vec.size() == 1) {
				newkeypoints.push_back(orbKeypoints[i]); // push back the point that was checked
				newweights.push_back(1);
			} else {
				Eigen::Vector3d middle(0, 0, 0);
				int weightsum = 0;
				for (int j = 0; j < indices_vec.size(); j++) {
					middle += chunkkpweights[indices_vec[j]] *  pcd->points_[indices_vec[j]];
					weightsum += chunkkpweights[indices_vec[j]];
					ignore.push_back(indices_vec[j]);
				}
				middle /= weightsum;
				c_keypoint tmpk;
				tmpk.p = Eigen::Vector4d(middle(0), middle(1), middle(2), 1.0);
				//tmpk.des = keypoints[i].des; //todo do  majority vote here
				getMajorityDescriptor(indices_vec, orbKeypoints, tmpk.des);
				newkeypoints.push_back(tmpk);
				newweights.push_back(indices_vec.size()); //todo should account previously merged points
			}
		}
	}


	orbKeypoints = newkeypoints;
	if (newkeypoints.size() != orbKeypoints.size() && it < 1) {
		//keypoints changed
		chunkkpweights = newweights;
		generateChunkKeypoints(++it);
	}
	else {
		//success finish method
		//build opencv descriptor from kps
		orbDescriptors = cv::Mat::zeros(orbKeypoints.size(), 32, CV_8U);
		for (int i = 0; i < orbKeypoints.size(); i++) {
			// build the filtered descriptor matrix
			for (int j = 0; j < 32; j++) {
				orbDescriptors.at<uchar>(i, j) = orbKeypoints[i].des[j];
			}
		}
	}


}


void Chunk::generateEfficientStructures() {
	//empty variables
	for (int i = 0; i < frames.size(); i++) {
		frames[i]->efficientKeypoints = vector<c_keypoint>();
	}
	efficientMatches = vector<rmatch>();

// build structures for efficient sparse alignment todo make it faster(not important, minor)
//efficientkeypoints contains all valid keypoints per frame and not more
//efficientmatches contains only the matches of these keypoints
	for (int i = 0; i < frames.size() - 1; i++) {
		auto f1 = frames[i];
		for (int j = i + 1; j < frames.size(); j++) {
			auto f2 = frames[j];
			for (int k = 0; k < filteredmatches(i , j).size(); k++) {
				int index1;
				int index2;
				auto& f = filteredmatches(i , j);	
				auto candidate = f1->orbKeypoints[f[k].indeces(f1->unique_id > f2->unique_id)]; //index is selected by boolean
				auto t = std::find(f1->efficientKeypoints.begin(), f1->efficientKeypoints.end(), candidate);
				if (t != f1->efficientKeypoints.end()) {
					//do nothing, element already contained
					index1 = std::distance(f1->efficientKeypoints.begin(), t);
				}
				else {
					f1->efficientKeypoints.push_back(candidate);
					index1 = f1->efficientKeypoints.size() - 1;
				}
				candidate = f2->orbKeypoints[f[k].indeces(f2->unique_id > f1->unique_id)];
				t = std::find(f2->efficientKeypoints.begin(), f2->efficientKeypoints.end(), candidate);
				if (t != f2->efficientKeypoints.end()) {
					//do nothing, element already contained
					index2 = std::distance(f2->efficientKeypoints.begin(), t);
				}
				else {
					f2->efficientKeypoints.push_back(candidate);
					index2 = f2->efficientKeypoints.size() - 1;
				}
				//add the correspondence
				efficientMatches.emplace_back(rmatch(i, j, index1, index2));
			}
		}
	}
}

void Chunk::deleteStuff()
{
	//todo
}
