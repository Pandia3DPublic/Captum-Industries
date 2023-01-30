#include "integrate.h"
#include "utils/matrixutil.h"
#include "configvars.h"


//variable declaration for static variables
std::atomic<bool> pandia_integration::stopintegrating(false); //signaling bool to stop the integration thread
list <shared_ptr<Frame>> pandia_integration::integrationBuffer; //buffer that contains new frames that must be integrate 
list <shared_ptr<Frame>> pandia_integration::reintegrationBuffer; //buffer that contains new frames that must be integrate 
list <shared_ptr<Frame>> pandia_integration::deintegrationBuffer; //buffer that contains frames that are not valid and must be deintegrated
list <shared_ptr<Frame>> pandia_integration::integratedframes; //a vector that gets periodically sorted which contains the frames in order corresponding to the dif in integrateddofs and worlddofs
std::mutex pandia_integration::integrationlock; //integrationlock for all integration buffers
std::mutex pandia_integration::tsdfLock; //lock for real time meshing of tsdf
pandia_integration::comps pandia_integration::comparator;


//this is called in a seperate thread
//Note: Thread safety is super important for all position variables!
//todo lables


void pandia_integration::removeFrameFromIntegration(shared_ptr<Frame> f) {
	bool found = false;
	pandia_integration::integrationlock.lock();
	for (auto it = pandia_integration::integrationBuffer.begin(); it!= pandia_integration::integrationBuffer.end(); it++) {
		if (f == (*it)) {
			pandia_integration::integrationBuffer.erase(it);
			found = true;
			break;
		}
	}
	//frame was already integrated, need to deintegrate
	if (!found) {
		for (auto it = pandia_integration::integratedframes.begin(); it!= pandia_integration::integratedframes.end(); it++) {
			if (f == (*it)) {
				pandia_integration::deintegrationBuffer.push_back(f);
				break;
			}
		}
	}
	pandia_integration::integrationlock.unlock();
	f->pushedIntoIntegrationBuffer = false;
}

void pandia_integration::integrateThreadFunction(Model* m) {
	bool lastit = false;
	bool go = true;
	std::cout << "Start integration \n";
	while (!stopintegrating) { //while not stopped
		// if nothing happens this thread goes to sleep for 20 ms
		bool nonew = true;
		bool noreint = true;

		//################ thread sensity preperation ######################
		shared_ptr<Frame> fint; //integration frame
		shared_ptr<Frame> fre; //reintegration frame
		Eigen::Matrix4d tint; //integration transformation
		Eigen::Matrix4d tre; //reintegration transformation
		Eigen::Vector6d worlddofsint; //worlddofs for thread safety
		Eigen::Vector6d worlddofsre; //worlddofs for thread safety
		integrationlock.lock();
		bool emptyint = integrationBuffer.empty();
		bool emptyre = reintegrationBuffer.empty();
		if (!emptyint) {
			fint = integrationBuffer.front();
			integrationBuffer.pop_front();
			tint = fint->getFrametoWorldTrans().inverse();
			worlddofsint = fint->getWorlddofs();;
		}
		if (!emptyre) {
			fre = reintegrationBuffer.front();
			reintegrationBuffer.pop_front();
			tre = fre->getFrametoWorldTrans().inverse();
			worlddofsre = fre->getWorlddofs();

		}
		integrationlock.unlock();
		//##################################
		if (!emptyint) {
			if (!fint->segmentationImage) { //we are most times here
				m->tsdf->Integrate(fint->rgbd->color_, fint->rgbd->depth_, g_intrinsic, tint); //inverse needed here, since we need the camera transformation!
			} else {
				utility::LogError("threaded label integration currently not supported");
				m->tsdf->IntegratewithLabels(fint->rgbd->color_, fint->rgbd->depth_, *fint->segmentationImage, g_intrinsic, tint); //inverse needed here, since we need the camera transformation!
			}
			integrationlock.lock();
			fint->integrateddofs = worlddofsint;
			integratedframes.push_back(fint);
			integrationlock.unlock();
			nonew = false;
		}

		if (!emptyre) {
			//reintegrate here
			m->tsdf->DeIntegrate(fre->rgbd->color_, fre->rgbd->depth_, g_intrinsic, getT(fre->integrateddofs.data()).inverse()); //inverse needed here, since we need the camera transformation!
			m->tsdf->Integrate(fre->rgbd->color_, fre->rgbd->depth_, g_intrinsic, tre); //inverse needed here, since we need the camera transformation!
			integrationlock.lock();
			fre->integrateddofs = worlddofsre;
			integrationlock.unlock();
			noreint = false;
		}
		if (nonew && noreint) {
			std::this_thread::sleep_for(20ms); //sleep since no task is necessary
		}
	}

	//####################################################### stop called ##########################################
	//do rest after stop signal 

	integrationlock.lock();
	while (!integrationBuffer.empty()) {
		auto f = integrationBuffer.front();
		integrationBuffer.pop_front();
		if (!f->segmentationImage) { //we are most times here
			cout << "integration after stop called \n";
			//m.tsdf->Integrate(*(f->rgbd), g_intrinsic, f->frametoworldtrans.inverse()); //inverse needed here, since we need the camera transformation!
			m->tsdf->Integrate(f->rgbd->color_, f->rgbd->depth_, g_intrinsic, f->getFrametoWorldTrans().inverse()); //inverse needed here, since we need the camera transformation!
			//m.tsdf->Integrate(*f->rgb, *f->depth, g_intrinsic, f->frametoworldtrans.inverse()); //inverse needed here, since we need the camera transformation!

		} else {
			m->tsdf->IntegratewithLabels(f->rgbd->color_, f->rgbd->depth_, *f->segmentationImage, g_intrinsic, f->getFrametoWorldTrans().inverse()); //inverse needed here, since we need the camera transformation!
			//m.tsdf->Integrate(f->rgbd->color_, f->rgbd->depth_, g_intrinsic, f->frametoworldtrans.inverse()); //inverse needed here, since we need the camera transformation!
			//m.tsdf->IntegrateonlyLabels(*f->depth,*f->segmentationImage, g_intrinsic, f->frametoworldtrans.inverse()); //inverse needed here, since we need the camera transformation!
		}
		f->integrateddofs = f->getWorlddofs();
		integratedframes.push_back(f);
	}
	//updateReintegrationBuffer();
	while (!reintegrationBuffer.empty()) {
		//reintegrate here
		cout << "reintegrate final " << reintegrationBuffer.size() << "\n";
		shared_ptr<Frame> f = reintegrationBuffer.front();
		reintegrationBuffer.pop_front();
		m->tsdf->DeIntegrate(f->rgbd->color_, f->rgbd->depth_, g_intrinsic, getT(f->integrateddofs.data()).inverse()); //inverse needed here, since we need the camera transformation!
		m->tsdf->Integrate(f->rgbd->color_, f->rgbd->depth_, g_intrinsic, f->getFrametoWorldTrans().inverse()); //inverse needed here, since we need the camera transformation!
		f->integrateddofs = f->getWorlddofs();

	}
	integrationlock.unlock();

	std::cout << "Finished integrating \n";

}

//note: no inverse needed for the cuda variant of integrated. happens internally
void pandia_integration::integrateThreadFunctionCuda(Model* m) {
	bool lastit = false;
	bool go = true;
	cuda::RGBDImageCuda rgbd = cuda::RGBDImageCuda(g_resx, g_resy, g_cutoff, 1000.0f);
	std::cout << "Start integration \n";
	while (!stopintegrating) { //while not stopped do integration and reintegration
		// if nothing happens this thread goes to sleep for 20 ms
		bool nonew = true;
		bool noreint = true;
		bool nodeint = true;
		//################ thread sensity preperation ######################
		shared_ptr<Frame> fint; //integration frame
		shared_ptr<Frame> fre; //reintegration frame
		shared_ptr<Frame> fde; //deintegration frame
		Eigen::Matrix4d tint; //integration transformation
		Eigen::Matrix4d tre; //reintegration transformation
		Eigen::Matrix4d tde; //deintegration transformation
		Eigen::Vector6d worlddofsint; //worlddofs for thread safety
		Eigen::Vector6d worlddofsre; //worlddofs for thread safety
		integrationlock.lock();
		bool emptyint = integrationBuffer.empty();
		bool emptyre = reintegrationBuffer.empty();
		bool emptyde = deintegrationBuffer.empty();
		if (!emptyint) {
			fint = integrationBuffer.front();
			integrationBuffer.pop_front();
			tint = fint->getFrametoWorldTrans();
			worlddofsint = fint->getWorlddofs();
			integratedframes.push_back(fint);//do push_back here to avoid race condition with frame removal
		}
		if (!emptyre) {
			fre = reintegrationBuffer.front();
			reintegrationBuffer.pop_front();
			//if (fre->unique_id > 110) {
			//	cout << fre->unique_id << "in reint buffer \n";
			//	cout << diff(fre) << "pose dif \n";
			//}
			tre = fre->getFrametoWorldTrans();
			worlddofsre = fre->getWorlddofs();

		}
		if (!emptyde) {
			fde = deintegrationBuffer.front();
			deintegrationBuffer.pop_front();
			tde = fde->getFrametoWorldTrans();
			integratedframes.remove(fde);
			
		}
		integrationlock.unlock();
		//################################## end thread sensity preperation #################### 
		//################################## main loop part #################### 
		//################################## integration #################### 
		if (!emptyint) {
			rgbd.UploadFloat(fint->rgbd->depth_, fint->rgbd->color_); //should not need look since images are never changed
			//rgbd.Upload(*fint->depth, fint->rgbd->color_); //should not need look since images are never changed
			cuda::TransformCuda extrinsic = cuda::TransformCuda::Identity();
			extrinsic.FromEigen(tint);

			tsdfLock.lock();
			auto start = std::chrono::high_resolution_clock::now();

			m->tsdf_cuda.Integrate(rgbd, g_intrinsic_cuda, extrinsic,&fint->touchedSubvolumes);
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

			//cout << "num of active subvolumes " << m->tsdf_cuda.active_subvolume_entry_array_.size() << endl;
			//cout << "Time " << duration.count() << " us\n";
			//cout << "division " << (double)duration.count() / (double)m->tsdf_cuda.active_subvolume_entry_array_.size() << endl;
			tsdfLock.unlock();

			integrationlock.lock();
			fint->integrateddofs = worlddofsint; //lock should be unnecessary
			fint->integrated =true;
			integrationlock.unlock();
			nonew = false;
		}
//######################################## reintegration #####################################
		if (!emptyre) {
			//reintegrate here
			//m->rgbd.Upload(*fre->depth, fre->rgbd->color_); //should not need lock since images are never changed
			m->rgbd.UploadFloat(fre->rgbd->depth_, fre->rgbd->color_); //should not need lock since images are never changed
			cuda::TransformCuda extrinsic = cuda::TransformCuda::Identity();
			cuda::TransformCuda extrinsic2 = cuda::TransformCuda::Identity();
			extrinsic.FromEigen(getT(fre->integrateddofs.data()));
			extrinsic2.FromEigen(tre);

			tsdfLock.lock();
			m->tsdf_cuda.DeIntegrate(m->rgbd, g_intrinsic_cuda, extrinsic,&fre->touchedSubvolumes); //inverse needed here, since we need the camera transformation!
			m->tsdf_cuda.Integrate(m->rgbd, g_intrinsic_cuda, extrinsic2,&fre->touchedSubvolumes); //inverse needed here, since we need the camera transformation!
			tsdfLock.unlock();

			integrationlock.lock();
			fre->integrateddofs = worlddofsre;
			integrationlock.unlock();
			noreint = false;
		}
//######################################## deintegration #####################################

		if (!emptyde) {
			//deintegrate here
			//m->rgbd.Upload(*fde->depth, fde->rgbd->color_); //should not need lock since images are never changed
			m->rgbd.UploadFloat(fde->rgbd->depth_, fde->rgbd->color_); //should not need lock since images are never changed
			cuda::TransformCuda extrinsic = cuda::TransformCuda::Identity();
			extrinsic.FromEigen(getT(fde->integrateddofs.data()));

			tsdfLock.lock();
			m->tsdf_cuda.DeIntegrate(m->rgbd, g_intrinsic_cuda, extrinsic,&fde->touchedSubvolumes); //inverse needed here, since we need the camera transformation!
			tsdfLock.unlock();
			integrationlock.lock();
			fde->integrateddofs = Eigen::Vector6d::Zero();
			integrationlock.unlock();

			nodeint = false;
		}
		if (nonew && noreint && nodeint) {
			std::this_thread::sleep_for(20ms); //sleep since no task is necessary
		}
	}
	//####################################################### stop called ##########################################
	//do rest after stop signal 
	integrationlock.lock();
	while (!integrationBuffer.empty()) {
		auto f = integrationBuffer.front();
		integrationBuffer.pop_front();
		if (!f->segmentationImage) { //we are most times here
			cout << "integration after stop called \n";
			//m->rgbd.Upload(*f->depth, f->rgbd->color_); //should not need look since images are never changed
			m->rgbd.UploadFloat(f->rgbd->depth_, f->rgbd->color_); //should not need look since images are never changed
			cuda::TransformCuda extrinsic = cuda::TransformCuda::Identity();
			extrinsic.FromEigen(f->getFrametoWorldTrans());
			tsdfLock.lock();
			m->tsdf_cuda.Integrate(m->rgbd, g_intrinsic_cuda, extrinsic,&f->touchedSubvolumes); //inverse needed here, since we need the camera transformation!
			tsdfLock.unlock();
		} 

		f->integrateddofs = f->getWorlddofs();
		f->integrated =true;
		integratedframes.push_back(f);
	}

	while (!reintegrationBuffer.empty()) {
		//reintegrate here
		cout << "reintegrate final " << reintegrationBuffer.size() << "\n";
		shared_ptr<Frame> f = reintegrationBuffer.front();
		reintegrationBuffer.pop_front();


		m->rgbd.UploadFloat(f->rgbd->depth_, f->rgbd->color_); //should not need look since images are never changed
		cuda::TransformCuda extrinsic = cuda::TransformCuda::Identity();
		cuda::TransformCuda extrinsic2 = cuda::TransformCuda::Identity();
		extrinsic.FromEigen(getT(f->integrateddofs.data()));
		extrinsic2.FromEigen(f->getFrametoWorldTrans());
			
		tsdfLock.lock();
		m->tsdf_cuda.DeIntegrate(m->rgbd, g_intrinsic_cuda, extrinsic,&f->touchedSubvolumes);
		m->tsdf_cuda.Integrate(m->rgbd, g_intrinsic_cuda, extrinsic2,&f->touchedSubvolumes); 
		tsdfLock.unlock();

		f->integrateddofs = f->getWorlddofs();

	}

	while (!deintegrationBuffer.empty()) {
		//reintegrate here
		utility::LogInfo("deintegrate final {} \n", deintegrationBuffer.size());
		shared_ptr<Frame> f = deintegrationBuffer.front();
		deintegrationBuffer.pop_front();
		m->rgbd.UploadFloat(f->rgbd->depth_, f->rgbd->color_);
		cuda::TransformCuda extrinsic = cuda::TransformCuda::Identity();
		extrinsic.FromEigen(getT(f->integrateddofs.data()));
		tsdfLock.lock();
		m->tsdf_cuda.DeIntegrate(m->rgbd, g_intrinsic_cuda, extrinsic,&f->touchedSubvolumes);
		tsdfLock.unlock();
		f->integrateddofs = f->getWorlddofs();
		integratedframes.remove(f);

	}




	integrationlock.unlock();

	std::cout << "Finished integrating \n";

}

double pandia_integration::diff(shared_ptr<Frame> a) {
	Eigen::Vector6d diff;
	diff.block<3, 1>(0, 0) = a->getWorlddofs().block<3, 1>(0, 0) - a->integrateddofs.block<3, 1>(0, 0);
	diff.block<3, 1>(3, 0) = 2 * (a->getWorlddofs().block<3, 1>(3, 0) - a->integrateddofs.block<3, 1>(3, 0));
	return diff.norm();
}

bool pandia_integration::compFrames(shared_ptr<Frame> a, shared_ptr<Frame> b){
	return diff(a) > diff(b);
}

// must always be called in mutex with integrationlock
void pandia_integration::updateReintegrationBuffer() {
	if (!integratedframes.empty()) {
		integratedframes.sort(compFrames);
		reintegrationBuffer.clear();
		auto it = integratedframes.begin();
		for (int i = 0; diff(*it) > g_treint && i < integratedframes.size(); i++) {
			//if ((*it)->unique_id < 112){
				reintegrationBuffer.push_back(*it);
				it++;
			//}
		}
	}
}



