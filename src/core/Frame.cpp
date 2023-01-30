#include "Frame.h"
#include "utils/matrixutil.h"
#include "utils/coreutil.h"
#include "configvars.h"



using namespace open3d;

Frame::Frame() {
	//utility::SetVerbosityLevel(utility::VerbosityLevel::VerboseAlways);
	 //utility::LogDebug("blubberdibu");
	unique_id = frame_id_counter;
	frametoworldtrans = Eigen::Matrix4d::Identity();
}

Frame::~Frame() {}//cout << "frame number " << unique_id << " destroyed. \n";}



Eigen::Vector6d Frame::getWorlddofs() {
	return worlddofs;
}

//only ever call this in the reconthread wihtout lock
Eigen::Matrix4d Frame::getFrametoWorldTrans() {
	return frametoworldtrans;
}
//must be called in integrationlock (todo)
void Frame::setFrametoWorldTrans(Eigen::Matrix4d a) {
	frametoworldtrans = a;
	worlddofs = MattoDof(a);
}

void Frame::setworlddofs(Eigen::Vector6d& a) {
	worlddofs = a;
	frametoworldtrans = getT(a.data());
}





