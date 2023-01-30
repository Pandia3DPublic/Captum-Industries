#include "core/threadvars.h"
#include "trajectory.h"
#include "utils/coreutil.h"
#include "configvars.h"
#include "readconfig.h"
#include "cameras/CameraThreadandInit.h"
#include "core/reconrun.h"
#include "cmakedefines.h"
#include "utils/visutil.h"

//test variables. Use for recording and testing to minimize mistakes
static string imagepath = "C:\\Users\\Kosnoros\\Documents\\Scenes2\\";
//static string imagepath = "C:\\Users\\Tim\\Documents\\Scenes\\";

using namespace open3d;
bool compareTrajectories(Model& m, vector<Eigen::Matrix4d>& poses) {
	int count = 0;
	bool passed = true;
	vector<Eigen::Matrix4d> frameposes;
	for (int i = 0; i < m.chunks.size(); i++) {
		for (int j = 0; j < m.chunks[i]->frames.size(); j++) {
			auto& f = m.chunks[i]->frames[j];
			frameposes.push_back(f->getFrametoWorldTrans());
			if ((poses[count] - f->getFrametoWorldTrans()).maxCoeff() > 0.01) {
				cout << "Difference detected in Chunks number " << i+1 << " Frame number " << j << endl;
				cout << "Diff matrix \n";
				cout <<  f->getFrametoWorldTrans() -poses[count] << endl;

				passed = false;
			}
			count++;
		}
	}
	//visualization::DrawGeometries({getOrigin(), getCameraPath(poses, Eigen::Vector3d(0.0,0.0,1.0)), getCameraPath(frameposes,Eigen::Vector3d(0.0,1.0,0.0))});
	return passed;
}


string getSceneName(int n) {
	string out = "scene000"  + to_string(n) + "_00";
	return out;
}

string getTrajName(int n) {
	string out = "s"  + to_string(n) + "_nr" + to_string(g_nread) +"_traj.txt";
	return out;
}

bool testTrajectories(int nstart, int end) {
	utility::LogInfo("Comparing trajectories with precomputed data \n");
	//get current folder with trajectory data
	string referencePath = TOSTRING(REFERENCEPATH);
	string configpath = referencePath + "testconfig.txt";
	readconfig(configpath);
	Model m;
	g_take_dataCam = true;
	g_camType = camtyp::typ_data;
	initialiseCamera();
	vector<Eigen::Matrix4d> diskposes;

	//utility::LogInfo("Starting with Scene 0, nread 250 \n");
	//reconrun(std::ref(m),false ,false);//model, test, livevis, integration
	//readTrajectory(referencePath + getTrajName(0),diskposes);
	//if (compareTrajectories(m, diskposes)) {
	//	cout << "TEST PASSED!!!! \n";
	//}

	for (int i = nstart; i <= end; i++) {
		utility::LogInfo("Scene {}, nread {} \n",i,g_nread);
		m.~Model();
		new (&m) Model;
		g_readimagePath = imagepath + getSceneName(i);
		cout << g_readimagePath << endl;
		reconrun(std::ref(m),false ,false);//model, test, livevis, integration
		diskposes.clear();
		readTrajectory(referencePath + getTrajName(i) ,diskposes);
		if (compareTrajectories(m, diskposes)) {
			cout << "TEST PASSED!!!! \n";
		}
	}
	
	return true;

}

//only works for data with the same intrinsic atm!
void recordTrajectories(int nstart, int end) {
	//get current folder with trajectory data
	string referencePath = TOSTRING(REFERENCEPATH);
	string configpath = referencePath + "testconfig.txt";

	readconfig(configpath);
	Model m;
	g_take_dataCam = true;
	g_camType = camtyp::typ_data;
	initialiseCamera();

	for (int i = nstart; i <= end; i++) {
		utility::LogInfo("Scene {}, nread {} \n",i,g_nread);
		m.~Model();
		new (&m) Model;
		g_readimagePath = imagepath + getSceneName(i);
		reconrun(std::ref(m),false ,false);//model, test, livevis, integration
		saveTrajectorytoDisk(referencePath ,m, getTrajName(i));
	}

}