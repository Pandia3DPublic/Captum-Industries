#include "Model.h"
#include "nllsFunctors.h"
#include "utils/matrixutil.h"
#include "utils/coreutil.h"
#include "configvars.h"
#include "utils/visutil.h"
#include "Gui/guiutil.h"


Model::Model()
{
	tsdf = std::make_shared<LabeledScalTSDFVolume>(g_voxel_length, 0.04, integration::TSDFVolumeColorType::RGB8);
	cuda::TransformCuda extrinsic = cuda::TransformCuda::Identity();
	rgbd = cuda::RGBDImageCuda(g_resx, g_resy, g_cutoff, 1000.0f);
	tsdf_cuda = cuda::ScalableTSDFVolumeCuda(8, g_voxel_length, 8 * g_voxel_length, extrinsic,7000,120000);
	tsdf_cuda.device_->min_dist = g_mincutoff;
	tsdf_cuda.device_->max_dist = g_cutoff;
	mesher = cuda::ScalableMeshVolumeCuda(cuda::VertexWithNormalAndColor, 8, 120000);

}

Model::~Model()
{
}




void Model::generateEfficientStructures() {
	for (int i = 0; i < chunks.size(); i++) {
		chunks[i]->efficientKeypoints = vector<c_keypoint>();
	}
	efficientMatches = vector<rmatch>();

	// build structures for efficient sparse alignment todo make it faster(not important, minor)
	//efficientkeypoints contains all valid keypoints per frame and not more
	//efficientmatches contains only the matches of these keypoints
	for (int i = 0; i < chunks.size() - 1; i++) {
		auto c1 = chunks[i];
		for (int j = i + 1; j < chunks.size(); j++) {
			auto c2 = chunks[j];
			for (int k = 0; k < filteredmatches(i, j).size(); k++) {
				int index1;
				int index2;
				auto& f = filteredmatches(i, j);
				//auto candidate = matches[n][k].p1;
				//auto candidate = c1->keypoints[f[k].indeces(0)];
				auto candidate = c1->orbKeypoints[f[k].indeces(c1->unique_id > c2->unique_id)]; //Index is selected by boolean
				auto t = std::find(c1->efficientKeypoints.begin(), c1->efficientKeypoints.end(), candidate);
				if (t != c1->efficientKeypoints.end()) {
					//do nothing, element already contained
					index1 = std::distance(c1->efficientKeypoints.begin(), t);
				}
				else {
					c1->efficientKeypoints.push_back(candidate);
					index1 = c1->efficientKeypoints.size() - 1;
				}
				//candidate = c2->keypoints[f[k].indeces(1)];
				candidate = c2->orbKeypoints[f[k].indeces(c2->unique_id > c1->unique_id)]; //here
				t = std::find(c2->efficientKeypoints.begin(), c2->efficientKeypoints.end(), candidate);
				if (t != c2->efficientKeypoints.end()) {
					//do nothing, element already contained
					index2 = std::distance(c2->efficientKeypoints.begin(), t);
				}
				else {
					c2->efficientKeypoints.push_back(candidate);
					index2 = c2->efficientKeypoints.size() - 1;
				}
				//if (j == chunks.size() - 1) {
				//	cout << rmatch(i, j, index1, index2)<< endl;
				//}
				//add the correspondence
				efficientMatches.emplace_back(rmatch(i, j, index1, index2));
			}
		}
	}
}

bool Model::allChunkshavePos() {
	bool allHavePos =true;
	for (int i=1; i< chunks.size(); i++) {
		if (chunks[i]->chunktoworldtrans == Eigen::Matrix4d::Identity()) { //todo dangerous in case two pcds are identical
			allHavePos =false;
		}
	}
	return allHavePos;
}

//takes first chunk stuff into account. Breaks if second image is identical to first (just does opt again which does nothing)
bool Model::getXor(int& a, int& b) {
	bool xor;
	if (a == 0){
		if (chunks[b]->chunktoworldtrans == Eigen::Matrix4d::Identity()) 
			xor = true;
	} else {
		if(!(chunks[a]->chunktoworldtrans == Eigen::Matrix4d::Identity()) != !(chunks[b]->chunktoworldtrans == Eigen::Matrix4d::Identity()))
			xor = true;
	}
	
	return xor;
}

void Model::addToCoeffs(Eigen::VectorXd& v, const Eigen::Vector3d& summand,const int& pos) {
	v[pos] += summand[0];
	v[pos+1] += summand[1];
	v[pos+2] += summand[2];
}

void Model::drawKeypoints(int start, int end) {
	if (end == 0) end = chunks.size()-1;
	std::vector<std::shared_ptr<const geometry::Geometry>> tmp;
	auto pcd = std::make_shared<geometry::PointCloud>();
	auto& colormap = visualization::GetGlobalColorMap();
	for (int a = start; a < end+1; a++) {
		//auto& kps = chunks[a]->orbKeypoints;
		auto& kps = chunks[a]->efficientKeypoints;
		for (int i = 0; i < kps.size(); i++) {
			pcd->points_.push_back((chunks[a]->chunktoworldtrans * kps[i].p).block<3, 1>(0, 0));
			pcd->colors_.push_back(colormap->GetColor((a-start) * (1.0/(double)(end+1-start))));
		}
	}


	std::shared_ptr<geometry::LineSet> ls = std::make_shared<geometry::LineSet>();
	int n = end + 1;
	for (int i = start; i < n - 1; i++) {
		for (int j = i + 1; j < n; j++) {
			auto& matches = filteredmatches(i, j);
			for (int k = 0; k < matches.size(); k++) {
				ls->points_.push_back((chunks[i]->chunktoworldtrans * matches[k].p1).block<3, 1>(0, 0));
				ls->points_.push_back((chunks[j]->chunktoworldtrans * matches[k].p2).block<3, 1>(0, 0));
				ls->lines_.push_back(Eigen::Vector2i(ls->points_.size() - 1, ls->points_.size() - 2));
			}
		}
	}
	tmp.push_back(ls);
	tmp.push_back(pcd);
	tmp.push_back(getOrigin());
	visualization::DrawGeometries(tmp);
}
void Model::drawKeypoints(const int& a,const int& b, Eigen::Matrix4d Ta, Eigen::Matrix4d Tb) {
	vector<match> local = filteredmatches(a,b);
	std::vector<std::shared_ptr<const geometry::Geometry>> tmp;
	auto pcd = std::make_shared<geometry::PointCloud>();
	for (int i = 0; i < local.size(); i++) {
		local[i].p1 = Ta * local[i].p1;
		local[i].p2 = Tb * local[i].p2;
		pcd->points_.push_back(local[i].p1.block<3,1>(0,0));
		pcd->points_.push_back(local[i].p2.block<3,1>(0,0));
		pcd->colors_.push_back(Eigen::Vector3d(1.0, 0.0, 0.0)); //red
		pcd->colors_.push_back(Eigen::Vector3d(0.0, 0.0, 1.0)); //blue
	}
	tmp.push_back(pcd);
	tmp.push_back(getOrigin());
	visualization::DrawGeometries(tmp);
}

void Model::getResiduals(int& a, int& b, vector<double>& res){
	res.clear();
	vector<match>& matches = filteredmatches(a,b);
	Eigen::Matrix4d& Ta = chunks[a]->chunktoworldtrans;
	Eigen::Matrix4d& Tb = chunks[b]->chunktoworldtrans;

	Eigen::Vector4d diff;
	for (auto& m : matches) {
		diff = Ta * m.p1 - Tb*m.p2;
		res.push_back(diff(0));
		res.push_back(diff(1));
		res.push_back(diff(2));
	}

}

double Model::getCost2(){
	//calculate max residual
	generateEfficientStructures();
	double cost =0;
	
	int nmatches = efficientMatches.size();
	//construct a cpu vector
	for (auto& v : efficientMatches) {
		auto& p1 = chunks[v.fi1]->efficientKeypoints[v.i1].p;
		auto& p2 = chunks[v.fi2]->efficientKeypoints[v.i2].p;
		Eigen::Vector4d tmp = chunks[v.fi1]->chunktoworldtrans * p1 - chunks[v.fi2]->chunktoworldtrans * p2;
		cost += tmp(0) * tmp(0);
		cost += tmp(1) * tmp(1);
		cost += tmp(2) * tmp(2);
		//cost += tmp.squaredNorm();
	}	

	return cost;
}
double Model::getCost(){
	//calculate max residual
	vector<double> res;
	double cost =0;
	for (int a = 0; a < chunks.size()-1; a++) {
		for (int b = a+1; b < chunks.size(); b++) {
			if (pairTransforms(a,b).set){
				getResiduals(a,b,res);
				for (int i = 0; i < res.size(); i++) {
					cost+=res[i] * res[i];
				}
			}
		}
	}

	return cost;
}

struct perfectDof {
	perfectDof(Model* modelpointer, int aex, int bex) : mp(modelpointer), a(aex), b(bex) {}

	bool operator()(double const* const* x, double* residuals) const { 

		Eigen::Matrix4d Tx = getT(x[0]);
		auto& trans = mp->pairTransforms(a,b);
		Eigen::Quaterniond qa(mp->chunks[a]->chunktoworldtrans.block<3,3>(0,0));
		Eigen::Quaterniond qap(Tx.block<3,3>(0,0));
		Eigen::Quaterniond qb(mp->chunks[b]->chunktoworldtrans.block<3,3>(0,0));
		Eigen::Quaterniond qbp((Tx * trans.invkabschtrans).block<3,3>(0,0));

		Eigen::Quaterniond erra = qa * qap.conjugate();
		residuals[0] = erra.coeffs()(0);
		residuals[1] = erra.coeffs()(1);
		residuals[2] = erra.coeffs()(2);
		residuals[3] = erra.coeffs()(3);
		Eigen::Vector3d diffa = mp->chunks[a]->chunktoworldtrans.block<3,1>(0,3) - Tx.block<3,1>(0,3);
		residuals[4] = diffa(0);
		residuals[5] = diffa(1);
		residuals[6] = diffa(2);

		Eigen::Quaterniond errb = qb * qbp.conjugate();
		residuals[7] = errb.coeffs()(0);
		residuals[8] = errb.coeffs()(1);
		residuals[9] = errb.coeffs()(2);
		residuals[10] = errb.coeffs()(3);
		Eigen::Vector3d diffb = mp->chunks[b]->chunktoworldtrans.block<3,1>(0,3) - (Tx * trans.invkabschtrans).block<3,1>(0,3);
		residuals[11] = diffb(0);
		residuals[12] = diffb(1);
		residuals[13] = diffb(2);

		
		return true;
	}
	//creator function
	static ceres::DynamicNumericDiffCostFunction<perfectDof>* create(Model* modelpointer, int a, int b) {
		return (new ceres::DynamicNumericDiffCostFunction<perfectDof>(new perfectDof(modelpointer, a, b))); //todo check if this gets deleted
	}

	Model* mp;
	int a;
	int b;
};

std::pair<Eigen::Matrix4d, Eigen::Matrix4d> Model::getOptDofTransforms(int& a, int& b) {
	std::pair<Eigen::Matrix4d, Eigen::Matrix4d> out;
	int nfree =6;
	double* x = new  double[nfree]; //vector with free variables on heap
	for (int i = 0; i < nfree; i++) {
		x[i] = 0;
	}
	int nr = 14; 
	ceres::Problem problem;
	//fast variant numeric
	ceres::DynamicNumericDiffCostFunction<perfectDof>* costfunction = perfectDof::create(this, a,b);
	costfunction->AddParameterBlock(nfree);
	costfunction->SetNumResiduals(nr);
	problem.AddResidualBlock(costfunction, NULL, x);

	ceres::Solver::Options options;
	options.max_num_iterations = 15;
	//options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
	options.linear_solver_type = ceres::DENSE_QR; //good
	options.minimizer_progress_to_stdout = false;
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT; // is default
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	out.first = getT(x);
	out.second = getT(x) * pairTransforms(a,b).invkabschtrans;

	delete[] x;
	//delete costfunction;

	return out;
}


struct perfectAlign {
	perfectAlign(Model* modelpointer, int aex, int bex) : mp(modelpointer), a(aex), b(bex) {}

	bool operator()(double const* const* x, double* residuals) const { 

		auto& matches = mp->filteredmatches(a,b);

		//each match creates 6 residuals
		for (int i = 0; i < matches.size(); i++) {
			Eigen::Vector4d diffa = getT(x[0]) * matches[i].p1 -  mp->chunks[a]->chunktoworldtrans * matches[i].p1;
			residuals[6*i] = diffa(0);
			residuals[6*i+1] = diffa(1);
			residuals[6*i+2] = diffa(2);
			//put in kabsch here to garantue perfect pcd align
			Eigen::Vector4d diffb = getT(x[0]) * mp->pairTransforms(a,b).invkabschtrans * matches[i].p2 - mp->chunks[b]->chunktoworldtrans * matches[i].p2;
			residuals[6*i+3] = diffb(0);
			residuals[6*i+4] = diffb(1);
			residuals[6*i+5] = diffb(2);
		}
		return true;
	}
	//creator function
	static ceres::DynamicNumericDiffCostFunction<perfectAlign>* create(Model* modelpointer, int a, int b) {
		return (new ceres::DynamicNumericDiffCostFunction<perfectAlign>(new perfectAlign(modelpointer, a, b))); //todo check if this gets deleted
	}

	Model* mp;
	int a;
	int b;
};

std::pair<Eigen::Matrix4d, Eigen::Matrix4d> Model::getOptTransforms(int& a, int& b) {
	std::pair<Eigen::Matrix4d, Eigen::Matrix4d> out;
	int nfree =6;
	double* x = new  double[nfree]; //vector with free variables on heap
	for (int i = 0; i < nfree; i++) {
		x[i] = 0;
	}
	int nr = filteredmatches(a,b).size() *6; //6 here is correct
	ceres::Problem problem;
	//fast variant numeric
	ceres::DynamicNumericDiffCostFunction<perfectAlign>* costfunction = perfectAlign::create(this, a,b);
	costfunction->AddParameterBlock(nfree);
	costfunction->SetNumResiduals(nr);
	problem.AddResidualBlock(costfunction, NULL, x);

	ceres::Solver::Options options;
	options.max_num_iterations = 15;
	//options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
	options.linear_solver_type = ceres::DENSE_QR; //good
	options.minimizer_progress_to_stdout = false;
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT; // is default
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	out.first = getT(x);
	out.second = getT(x) * pairTransforms(a,b).invkabschtrans;

	delete[] x;
	//delete costfunction;

	return out;
}

void Model::performOptIteration() {
	//init varibles
	Eigen::Vector3d ones = Eigen::Vector3d::Ones();
	int nfree = 6*(chunks.size()-1);
	Eigen::VectorXd xt = Eigen::VectorXd::Zero(nfree/2); //first 6 entries for second chunks and so on
	Eigen::VectorXd mu= Eigen::VectorXd::Zero(nfree/2);
	Eigen::VectorXd tsum= Eigen::VectorXd::Zero(nfree/2);
	vector<vector<Eigen::Quaterniond>> quats_all; //quaternions for latter averagging
	quats_all.resize(chunks.size() -1);
	//iterate over all cors
	for (int a = 0; a < chunks.size()-1; a++) {
		for (int b = a+1; b < chunks.size(); b++) {
			if (pairTransforms(a,b).set){
				int apos = 3*(a-1); //position of translation dofs for pcd a in vector
				int bpos = 3*(b-1);
				if (a !=0){
					addToCoeffs(mu,ones,apos);
					addToCoeffs(mu,ones,bpos);
					Eigen::Matrix4d T_ai = chunks[b]->chunktoworldtrans * pairTransforms(a,b).kabschtrans; 
					Eigen::Matrix4d T_bi = chunks[a]->chunktoworldtrans * pairTransforms(a,b).invkabschtrans;
					Eigen::Matrix4d pca = getHalf(chunks[a]->chunktoworldtrans, T_ai);
					Eigen::Matrix4d pcb = getHalf(chunks[b]->chunktoworldtrans, T_bi);

					//doesnt work well but should work in principle. Since cost results are extremely similar
					//it seems probable that the failure to converge for small values lies in the averaging technique
					//std::pair<Eigen::Matrix4d,Eigen::Matrix4d> transforms = getOptTransforms(a,b);
					//pca = transforms.first;
					//pcb = transforms.second;

					//rotation
					quats_all[a-1].push_back(Eigen::Quaterniond(pca.block<3,3>(0,0)));
					quats_all[b-1].push_back(Eigen::Quaterniond(pcb.block<3,3>(0,0)));
					//translation
					//pcd a 
					addToCoeffs(tsum,pca.block<3,1>(0,3),apos);
					//pcd b
					addToCoeffs(tsum,pcb.block<3,1>(0,3),bpos);
				} else { // a == 0
						 //note:  chunks[a]->chunktoworldtrans is always identity. 
					quats_all[b-1].push_back(Eigen::Quaterniond(pairTransforms(a,b).invkabschtrans.block<3,3>(0,0)));
					addToCoeffs(mu,ones,bpos);
					addToCoeffs(tsum,pairTransforms(a,b).invkabschtrans.block<3,1>(0,3),bpos);
				} 
			}
		}
	}

	xt = tsum.cwiseQuotient(mu);
	mu= Eigen::VectorXd::Zero(nfree);
	tsum= Eigen::VectorXd::Zero(nfree);

	//set transforms
	for (int i = 0; i < chunks.size() - 1; i++) {
		chunks[i+1]->chunktoworldtrans.block<3,3>(0,0) = Eigen::Matrix3d(getQuaternionAverage(quats_all[i]));
		chunks[i+1]->chunktoworldtrans.block<3,1>(0,3) = xt.block<3,1>(3*i,0);
	}
	//cout << "cost is " << getCost() << endl;
}

void Model::getMaxResidual(double& max, int& ares, int& bres) {
	//calculate max residual
	max = 0;
	vector<double> res;
	for (int a = 0; a < chunks.size()-1; a++) {
		for (int b = a+1; b < chunks.size(); b++) {
			if (pairTransforms(a,b).set){
				getResiduals(a,b,res);
				for (int i = 0; i < res.size(); i++) {
					if (abs(res[i]) > max) {
						max = abs(res[i]);
						ares = a;
						bres =b;
					}
				}
			}
		}
	}
}

bool Model::doChunktoModelforNewestChunk() {
	Eigen::VectorXd xt = Eigen::Vector3d::Zero(); 
	Eigen::VectorXd tsum= Eigen::Vector3d::Zero();
	int count = 0;
	vector<Eigen::Quaterniond> quats; //quaternions for latter averagging


	auto& c= chunks.back();
	int b = chunks.size() - 1;
	if (b == 0 ) return true;
	for (int a = 0; a < chunks.size() - 1; a++) {
		if (pairTransforms(a,b).set){
			if (!chunks[a]->chunktoworldtrans.isIdentity() || a ==0) {
				count++;
				Eigen::Matrix4d pcb = chunks[a]->chunktoworldtrans * pairTransforms(a,b).invkabschtrans;
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
		c->chunktoworldtrans = out;
		return true;
	} else {
		return false;
	}

}


//this is frame to model plus
void Model::performSparseOptimization2() {
	//first part is only frame to model tracking
	Eigen::Vector3d ones = Eigen::Vector3d::Ones();
	int nfree = 6*(chunks.size()-1);
	Eigen::VectorXd xt = Eigen::VectorXd::Zero(nfree/2); //first 6 entries for second chunks and so on
	Eigen::VectorXd mu= Eigen::VectorXd::Zero(nfree/2);
	Eigen::VectorXd tsum= Eigen::VectorXd::Zero(nfree/2);
	vector<vector<Eigen::Quaterniond>> quats_all; //quaternions for latter averagging
	quats_all.resize(chunks.size() -1);
	bool optimized = false;
	//calculate positions for all chunks. For a model this will only ever be necessary for the last chunk atm.
	while (!allChunkshavePos()) { //gets only ever called if all chunks have pos
		unordered_set<int> part; //collects indices of pcd that have been optimized.
		//iterate over all cors
		for (int a = 0; a < chunks.size()-1; a++) {
			for (int b = a+1; b < chunks.size(); b++) {
				if (pairTransforms(a,b).set){
					int apos = 3*(a-1); //position of translation dofs for pcd a in vector
					int bpos = 3*(b-1);
					if (getXor(a,b)) { //xor todo test, only one has a pos already
						if (chunks[a]->chunktoworldtrans == Eigen::Matrix4d::Identity()) {
							if (a != 0) {
								cout << "hi never in optimize model \n";
								//add to mu
								addToCoeffs(mu,ones,apos);
								//for intuition: chunktoworldtrans gives new origin
								Eigen::Matrix4d pca = chunks[b]->chunktoworldtrans * pairTransforms(a,b).kabschtrans;
								auto pc = pca.block<3,1>(0,3); //translation
								quats_all[a-1].push_back(Eigen::Quaterniond(pca.block<3,3>(0,0)));
								addToCoeffs(tsum,pc,apos);//A here since its consistent with looping in reconrun
								part.insert(a);
							}
						}
						if (chunks[b]->chunktoworldtrans == Eigen::Matrix4d::Identity()) {
							//if (b > 5) {
								//drawKeypoints(a,b,chunks[a]->chunktoworldtrans, chunks[a]->chunktoworldtrans * pairTransforms(a,b).invkabschtrans); //perfect
							//}
							addToCoeffs(mu,ones,bpos);
							//take inverse here and origin of a, since we transform b to a. as intial pos give new origin
							Eigen::Matrix4d pcb = chunks[a]->chunktoworldtrans * pairTransforms(a,b).invkabschtrans;
							quats_all[b-1].push_back(Eigen::Quaterniond(pcb.block<3,3>(0,0)));
							addToCoeffs(tsum,pcb.block<3,1>(0,3),bpos);//A here since its consistent with looping in reconrun
							part.insert(b);
							optimized = true;
						}
					}
				}
			}
		}
		xt = tsum.cwiseQuotient(mu); //solve, gives nan for zero mu entries, but should not matter
		mu= Eigen::VectorXd::Zero(nfree);
		tsum= Eigen::VectorXd::Zero(nfree);

		//set transforms only for the affected chunks
		for (const auto& elem: part) {
			chunks[elem]->chunktoworldtrans.block<3,3>(0,0) = Eigen::Matrix3d(getQuaternionAverage(quats_all[elem-1]));
			chunks[elem]->chunktoworldtrans.block<3,1>(0,3) = xt.block<3,1>(3*(elem-1),0);
		}

		for (auto& quats : quats_all) {
			quats.clear();
		}

	}

	if (!optimized) {
		utility::LogError("Did not optimize in frame to model \n");
	}
	// uncomment from there to get convergence and high res filter which don't work properly ################
	//int ntest = 1000;
	//if (chunks.size() == ntest){
	//	for (int i = 1; i < chunks.size(); i++) {
	//		Eigen::Matrix4d tmp2 = getRy(0.5 * i) * gettrans(Eigen::Vector3d(0, 0, 1.5));
	//		//chunks[i]->chunktoworldtrans = tmp2;
	//	}
	//	//drawKeypoints(3,5);
	//	drawKeypoints();
	//}


	////optimize with n iterations
	//for (int i = 0; i < g_nopt; i++) { //todo dynamic g_nopt
	//	performOptIteration();
	//}

	////calculate max residual
	//double max = 0;
	//int ares= -1;
	//int bres =-1;
	//getMaxResidual(max,ares,bres);

	//double oldcost;
	//double newcost;
	//double diff;
	//double thres = 0.075;
	//int maxit= 250;
	//int it = 0;
	//bool increasing = true;
	//if (max > thres) {
	//	oldcost = 1e6;
	//	newcost = getCost();
	//	diff = (newcost-oldcost)/(newcost + oldcost);
	//	while (abs(diff) > 0.0001 && it < maxit && increasing) {
	//		performOptIteration();
	//		it++;
	//		oldcost= newcost;
	//		newcost= getCost();
	//		diff = (newcost-oldcost)/(newcost + oldcost);
	//		if (it !=1 && newcost > oldcost) //the first iteration often increases cost todo bug.
	//			increasing = false;
	//	}
	//	getMaxResidual(max,ares,bres);
	//}


	////if (chunks.size() == 41)
	////drawKeypoints();
	//if (max > thres) {
	//	utility::LogInfo("Matches removed by residual filter in Model opt. Biggest residual was {} \n",max);
	//	pairTransforms(ares,bres).set= false;
	//	filteredmatches(ares,bres) = vector<match>();
	//	performSparseOptimization2();
	//}

}
//todo optimize. Try to find a more efficient linear sovler and find out why jacobi evalution takes so long despite superior
//residual calculation
//todo parallel with num_threads does nothing here
void Model::performSparseOptimization(vector<Eigen::Vector6d>& initx) {
	generateEfficientStructures();
	int nfree = 6 * (chunks.size() - 1); // number of free variables
	double* x = new  double[nfree]; //vector with free variables on heap
	//for (int i = 0; i < nfree; i++) {
	//	x[i] = 0;
	//}
	for (int i = 0; i < chunks.size() - 1; i++) {
		for (int j = 0; j < 6; j++) {
			x[6 * i + j] = initx[i](j);
		}
	}

	int nr = efficientMatches.size();
	nr = 3 * nr; //number of residuals
	std::vector<ceres::ResidualBlockId> residual_block_ids;

	if (nr > 0) {
		ceres::Problem problem;
		//fast variant numeric
		ceres::DynamicNumericDiffCostFunction<nccf2model>* costfunction = nccf2model::create(this);
		costfunction->AddParameterBlock(nfree);
		costfunction->SetNumResiduals(nr);
		problem.AddResidualBlock(costfunction, NULL, x);
		//delete costfunction; //todo test
		//analytic
		//ceres::CostFunction* costfunction = new SparseModelAna(this);
		//ceres::ResidualBlockId block_id = problem.AddResidualBlock(costfunction, NULL, x);
		//residual_block_ids.push_back(block_id);


		//fast variant autodiff
		//ceres::DynamicAutoDiffCostFunction<amf2>* costfunction = amf2::create(this);
		//costfunction->AddParameterBlock(nfree);
		//costfunction->SetNumResiduals(nr);
		//problem.AddResidualBlock(costfunction, NULL, x);



		ceres::Solver::Options options;
		options.max_num_iterations = 15;
		//options.linear_solver_type = ceres::DENSE_QR;
		options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY; //good
		options.minimizer_progress_to_stdout = false;
		options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT; // is default
		//options.num_threads = 8; // does apparently nothing, makes it slightly slower
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		//cout << "linear solver time in model: " << summary.linear_solver_time_in_seconds << endl;
		//cout << "jac eval time in model: " << summary.jacobian_evaluation_time_in_seconds << endl;

		//std::cout << summary.FullReport() << "\n";
		//cout << "Model Successfull steps: " << summary.num_successful_steps << endl;
		//cout << "Model bad steps: " << summary.num_unsuccessful_steps << endl;
		utility::LogDebug("Optimized Model \n");


		// set transformations
		for (int i = 0; i < chunks.size() - 1; i++) {
			chunks[i + 1]->chunktoworldtrans = getT(x + 6 * i);
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
			int matchindex = index / 3;
			//remove matches
			filteredmatches(efficientMatches[matchindex].fi1, efficientMatches[matchindex].fi2) = vector<match>();
			//generateEfficientStructures();
			utility::LogInfo("Matches removed by residual filter in Model opt. Biggest residual was {} \n",residuals[index]);
			//take current result as initial values
			vector<Eigen::Vector6d> tmp;
			tmp.reserve(chunks.size() - 1);
			for (int i = 0; i < chunks.size() - 1; i++) {
				for (int j = 0; j < 6; j++) {
					tmp[i](j) = x[6 * i + j]; 
				}
			}
			cout << "high res ! \n";
			performSparseOptimization(tmp); //use current dof for new staring point
		}
	}
	else {
		// set transformations
		for (int i = 0; i < chunks.size() - 1; i++) {
			chunks[i + 1]->chunktoworldtrans = getT(x + 6 * i);
		}
		utility::LogWarning("No matches between chunks, meaningless optimization \n");
	}



	delete[] x;

}

//only call this in integrationlock!
void Model::setWorldTransforms() {
	////test code
	//if (chunks.size() == 2) {
	//	double* x = new double[6];
	//	x[0] = 0; x[1] = 0; x[2] = 1; x[3] = 0; x[4] = 0; x[5] = 0;
	//	//chunks[1]->chunktoworldtrans = getT(x);
	//	//chunks[2]->chunktoworldtrans = getT(x);
	//	//chunks[3]->chunktoworldtrans = getT(x);
	//	chunks[1]->chunktoworldtrans = getT(x);
	//	delete[] x;
	//	cout << "#################################################### \n";
	//}
	////end test code
	for (int j = 0; j < chunks.size(); j++) {
		for (int k = 0; k < chunks[j]->frames.size(); k++) {
			auto& f = chunks[j]->frames[k];
			f->setFrametoWorldTrans(chunks[j]->chunktoworldtrans * f->chunktransform);
			//f->frametoworldtrans = chunks[j]->chunktoworldtrans * f->chunktransform;
			//f->worlddofs = MattoDof(f->frametoworldtrans);
			f->worldtransset = true;
		}
	}
}

void Model::integrateCPU() {
	for (int i = 0; i < chunks.size(); i++) {
		for (int j = 0; j < chunks[i]->frames.size(); j++) {
			auto& f = chunks[i]->frames[j];
			if (!f->duplicate)
				tsdf->Integrate(f->rgbd->color_, f->rgbd->depth_, g_intrinsic, f->getFrametoWorldTrans().inverse()); //inverse needed here,
		}
	}
}


void Model::saveCPUMesh(string name) {
	
	auto tmpmesh = mesher.mesh().Download();
	tmpmesh->ComputeTriangleNormals(); //also normalizes
	tmpmesh->ComputeAdjacencyList();

	
	//prevent things from crashing due to out of bounds colors
	if (PandiaGui::file_dialogSaveMesh.ext == ".ply") {
		for (auto& c : tmpmesh->vertex_colors_) {
			if (c(0) > 1 || c(0) < 0) {
				c(0) = 0;
			}
			if (c(1) > 1 || c(1) < 0) {
				c(1) = 0;
			}
			if (c(2) > 1 || c(2) < 0) {
				c(2) = 0;
			}
		}
	}
	tmpmesh->Transform(getflip());
	io::WriteTriangleMesh(name,*tmpmesh,false, false, true, true, true);

}

