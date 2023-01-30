#include "Tests/LiveTest.h"
#include "Tests/trajectory.h"

int main() {
	open3d::utility::SetVerbosityLevel(open3d::utility::VerbosityLevel::Info);
	//
	//std::cout << "Performing Test: Is result of live vis equal to offline vis \n";
	//std::cout << "Result is: " << liveTestwithVis() <<std::endl;
	//std::cout << "Graphical Frustum intersection test: todo code is already there\n";


	cout << "Performing Trajectory Test againt trajectory file.";

	//performing this one after the other should always result in a perfect test. For real testing jsut use testTrajectories!.
	//recordTrajectories(0,5);
	testTrajectories(0,5);

	return 0;
}