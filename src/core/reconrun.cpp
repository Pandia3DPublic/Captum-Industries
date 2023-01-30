#include "reconrun.h"
#include <random>
#include "GPUSolver/solverthread.h"
#include "utils/matrixutil.h"
using namespace std;

#define USEGPUSOLVER //for debugging


//to start global solve and enable a memcpy to gpu memory while main thread is blocked
void startGlobalsolve(Model& m,solverWrapper& solver, vector<Eigen::Vector6d>& globalSolverInit) {
	solverlock.lock();
	globalSolverInit.clear();
	for (int i = 0; i < m.chunks.size(); i++) {
		globalSolverInit.push_back(MattoDof(m.chunks[i]->chunktoworldtrans));
	}
	solverlock.unlock();

	m.generateEfficientStructures();
	solver.constructGPUCors(); //now stuff is in gpu memory
	g_solverRunning = true; //this start the solve

}

void transferGlobalResults(Model& m, solverWrapper& solver) {
	for (int i = 0; i < solver.nChunks; i++) {
		//setting chunktoworldtrans is not thread sensitive
		m.chunks[i]->chunktoworldtrans = getT(solver.x.data() + 6 * i);
	}
}

//gets all chunks in the local group, e.g. frustums overlap and camera points in the same direction
//contains itself
void getLocalGroup(Model& m, shared_ptr<Chunk> chunk, vector<shared_ptr<Chunk>>& localGroup) {
	localGroup.reserve(m.chunks.size());
	for (int i=0 ;i < m.chunks.size(); i++){
		m.chunks[i]->frustum->reposition(m.chunks[i]->chunktoworldtrans);
		if (chunk->frustum->inLocalGroup(*m.chunks[i]->frustum)) {
			localGroup.push_back(m.chunks[i]);
		} 
	}

	//limit to size of 30 random chunks
	int n = localGroup.size();
	if (n > g_nLocalGroup){
		vector<shared_ptr<Chunk>> tmpGroup;
		tmpGroup.reserve(g_nLocalGroup);
		vector<bool> taken(n,0);
		int ntaken = 0;
		while(ntaken < g_nLocalGroup){
			int r = rand()%n;
			if (!taken[r]){
				taken[r] = true;
				tmpGroup.push_back(localGroup[r]);
				ntaken++;
			}
		}
	
		localGroup = tmpGroup;
	}

}

//resets all global thread vars
void resetThreadVars() {
	pandia_integration::stopintegrating = false;


}


//if data we read data from a dataset. Otherwise a camera is enabled
int reconrun(Model& m, bool livevis, bool integrate) {
	using namespace open3d;
	//launchcudatest(); causes crashed prob due to unified cuda memory
	//string s;
	//s = TOSTRING(datapath);
	//prepareDatapath(s);
	srand(1); //seed for reasonable testing (local groups)

	//############################## variable declarations ##############################
	solverWrapper solver(&m);
	//thread vars 
	g_reconThreadFinished = false;
	g_current_slam_finished = false;
	thread integrationThread;
	thread visThread;
	list <shared_ptr<Frame>> framebuffer;
	list <shared_ptr<Frame>> segframebuffer;
	std::atomic<bool> stopRecording(false);
	std::atomic<bool> stopVisualizing(false); //signaling bool to stop the visualization  thread (deprecated)

	shared_ptr<std::thread> cameraThread;
	shared_ptr<std::thread> segThread;
	std::atomic<bool> modelLoaded(false);
	std::atomic<bool> stopSegment(false);
	vector<Eigen::Vector6d> globalSolverInit;



	//debug variables
	int npass = 0; //number of cors passed through kabsch (chunk)
	int npassm = 0;
	//utitlity variables
	std::shared_ptr<Frame> f;
	vector<int> nvalidFrameIndices;
	int nnv = 0; //number of invalid frames in consequtive order
	int cfi = 0; //current frame index
	int cci = 0; //current chunk index
	bool completeChunk = false; //  indicates that the last chunk is full and valid
	bool firstframe = true;
	bool firstchunk = true;
	bool lastChunkValid = true;
	Frame lastframe;
	shared_ptr<Chunk> currentChunk;
	auto timeofLastGlobalOpt = std::chrono::high_resolution_clock::now();
	bool copiedSolverResults =true; //indicates wheter global solve results have been incorporated into model.


	//segmentation stuff
	if (g_segment) {
		segThread = make_shared<std::thread>(segmentationThread, &segframebuffer, &stopSegment, &modelLoaded);
		while (!modelLoaded) {
			std::this_thread::sleep_for(0.1s); // time to warm up for camera
		}
		std::cout << "Neural net loaded. Please press any button to continue \n";
		cin.get();
	}

	//############################## camera or connection thread start ##############################
	if (!g_cameraParameterSet){
		utility::LogError("Cam paras must be set before starting camera thread in reconrun. \n");
	}

	// Camera Kinect
	if (g_camType == camtyp::typ_kinect) {
		cameraThread = make_shared<std::thread>(KinectThread, std::ref(framebuffer), std::ref(stopRecording), std::ref(g_cameraParameterSet));//startet neuen thread und liest bidler ein. Filtert bilder und schreibt diese in picturebuffer

	}
	if (g_camType == camtyp::typ_client) {
		cameraThread = make_shared<std::thread>(StartClientThreads, std::ref(framebuffer), std::ref(stopRecording)); //start server in thread, receive compressed images from client, decompress and push to framebuffer

	}

	//Realsense Camera without thread option
	if (g_camType == camtyp::typ_realsense) {
		cameraThread = make_shared<std::thread>(RealsenseThread, std::ref(framebuffer), std::ref(stopRecording), std::ref(g_cameraParameterSet)); //startet neuen thread und liest bidler ein. Filtert bilder und schreibt diese in picturebuffer
	}

	if (g_camType == camtyp::typ_data){
		cout << "starting data cam threads \n";
		cameraThread = make_shared<std::thread>(DataCamThreadFunction, std::ref(framebuffer), std::ref(stopRecording));//read images from hardrive
	}

	
	//#################################### start various threads
	//start threads
	if (integrate)
		integrationThread = thread(pandia_integration::integrateThreadFunctionCuda,&m);
	
	if (livevis)
		visThread = thread(visThreadFunction, &m, std::ref(stopVisualizing));


#ifdef USEGPUSOLVER
	thread globalSolverThread;
	globalSolverThread = thread(gpuSolverThread, &m, &solver, &globalSolverInit);
#endif

	int readUntil = g_nread;

	//######################################################### main loop ##################################################
	int i=0;
#ifdef WIHTOUT_GUI
	while(i < readUntil){
#else
	while(!g_clear_button && !PandiaGui::g_postProcessing){
#endif 
	//Timer t("Time in single iteration",TimeUnit::millisecond);
		//if number of read frames exceeds nread put in pause state
		if (i >= readUntil) {
			g_pause = true;
			//break; //remove, just for debug!
			g_programState = gui_PAUSE;
			g_wholeMesh  = true;
			m.tsdf_cuda.unmeshed_data = true;
			readUntil = readUntil + g_nread;
		}

	//#######################################generate new chunk if necessary ##############################################
		//generate new chunk if necessary
		if (firstframe) {
			currentChunk = make_shared<Chunk>();
			firstframe = false;
		}
		if (completeChunk) {
			currentChunk = make_shared<Chunk>();
			if (lastChunkValid){
				currentChunk->frames.push_back(make_shared<Frame>(lastframe));// put a copy of the last frame into the new chunk.
				g_trackingLost = false;
			} else {
				g_trackingLost = true;
			}
			nvalidFrameIndices.clear(); //this is new
			nnv = 0;
			completeChunk = false;
		}
		if (nnv > 5) {
			utility::LogWarning("Restarted Chunk since tracking was lost for 6 frames within the chunk\n");
			g_trackingLost = true;
			for (auto& f: currentChunk->frames){
				if (f->pushedIntoIntegrationBuffer) {
					pandia_integration::removeFrameFromIntegration(f);
				}
			}
			currentChunk = make_shared<Chunk>();
			nvalidFrameIndices.clear(); 
			nnv = 0;
			completeChunk = false;
			lastChunkValid = false;
		}

//########################################### acquire new chunk data #########################################
		//get image data

		f = getSingleFrame(framebuffer, m.recordbuffer); //this methods locks if no frame is available
		if (f == nullptr) goto pauseStateLabel; //happens if pause gets us out of getSingleFrame
		//check if image data gives enough orb keypoints
		if (f->orbKeypoints.size() < 50) {
			utility::LogDebug("warning: low keypoint size : {} \n", f->orbKeypoints.size());
			while (f->orbKeypoints.size() < 10 && i < g_nread-1 && !g_clear_button) {
				f = getSingleFrame(framebuffer,m.recordbuffer); //this methods locks if no frame is available
				if (f == nullptr) goto pauseStateLabel; //happens if pause gets us out of getSingleFrame
					
				
				i++;
				utility::LogWarning("Discarded Frame since it has less than 10 keypoints.\n");
				while (g_pause) { std::this_thread::sleep_for(20ms); };
			}
		}

		// trigger each 20 frames
		if (g_segment && i % 20 == 0) {
			seglock.lock();
			segframebuffer.push_back(f);
			seglock.unlock();

		}
//############################################### put frame in place #################################
		//put the new frame in the correct place 
		if (nvalidFrameIndices.size() != 0) { //insert new frame if invalid frame was detected. Note temporal realtions are not preserved this way
			utility::LogWarning("Refilling invalid frame in chunk at position number {}. \n", nvalidFrameIndices.front());
			utility::LogDebug("nnv: {} \n", nnv);
			//make sure the frame is deintegrated or not integrated to start with
			if (currentChunk->frames[nvalidFrameIndices.front()]->pushedIntoIntegrationBuffer)
				pandia_integration::removeFrameFromIntegration( currentChunk->frames[nvalidFrameIndices.front()]);
			currentChunk->frames[nvalidFrameIndices.front()] = f;
			cfi = nvalidFrameIndices.front();
			nvalidFrameIndices.erase(nvalidFrameIndices.begin());//todo just pop_front

		} else {
			currentChunk->frames.push_back(f); //frame is inserted but not yet valid. usefull for high res filter
			cfi = currentChunk->frames.size() - 1;
		}

//################################ core filter operations #####################################
		//int npass = 0;
		//constant effort, scales well
		for (int j = 0; j < currentChunk->frames.size(); j++) {
			if (j != cfi) {
				getCors(currentChunk->frames[j], f, currentChunk->filteredmatches(j, cfi));
				kabschfilter(currentChunk->filteredmatches(j, cfi), currentChunk->pairTransforms(j, cfi));
				if (currentChunk->filteredmatches(j, cfi).size() != 0) {
					reprojectionfilter(currentChunk->frames[j], currentChunk->frames[cfi], currentChunk->pairTransforms(j, cfi), currentChunk->filteredmatches(j, cfi));
				}
				//if (currentChunk->filteredmatches(j, cfi).size() != 0) {
				//	npass += 1;
				//}
			}
		}
		//cout << "npass " << npass << endl;
//######################## integrate frame #######################################
		Eigen::Matrix4d Tc = Eigen::Matrix4d::Zero() ; //chunk matrix
		Eigen::Matrix4d Tk = Eigen::Matrix4d::Zero(); //kabsch matrix

		if (firstchunk) {
			Tc = Eigen::Matrix4d::Identity();
		} else {
			if (lastChunkValid)
				Tc = m.chunks[m.chunks.size()-1]->frames.back()->getFrametoWorldTrans(); //-1 here since the new chunk isnt added yet
		}


		bool gotT = currentChunk->doFrametoModelforNewestFrame();
		if (lastChunkValid && gotT){
			Eigen::Matrix4d T = f->chunktransform * Tc;
			f->setFrametoWorldTrans(T);
			pandia_integration::integrationlock.lock();
			pandia_integration::integrationBuffer.push_back(f);
			pandia_integration::integrationlock.unlock();
			f->pushedIntoIntegrationBuffer = true;
			//if (m.chunks.size() == 11) {
				//cout << f->unique_id << "getting integrated before opt \n";
			//}
			//	cout << "unique id " << f->unique_id << endl;
				//visualization::DrawGeometries({geometry::PointCloud::CreateFromRGBDImage(*f->rgbd,g_intrinsic)});
		//	}
			currentposlock.lock();
			m.currentPos = T;
			currentposlock.unlock();
		}


//#####################if chunk is full check if all frames have cors, otherwise remove frame ############################
		if (currentChunk->frames.size() == nchunk && nvalidFrameIndices.size() == 0) {
			completeChunk = checkValid(currentChunk, nvalidFrameIndices);
			if (completeChunk) { //else jumps very far
				utility::LogDebug("Chunk number {} is full \n", m.chunks.size() + 1);
				currentChunk->generateEfficientStructures();
				int oldsize = currentChunk->efficientMatches.size();
				currentChunk->performSparseOptimization( getDoffromKabschChunk(currentChunk)); //this applies the high residual filter
				if (oldsize != currentChunk->efficientMatches.size()) { //high res was active, so check again for valid chunk
					completeChunk = checkValid(currentChunk, nvalidFrameIndices);
				}
				if (!completeChunk) {
					utility::LogError("Frame is invalid after high res filter \n");
					nnv += nvalidFrameIndices.size();
				}
//######################################## Model  ################################################################################
				if (completeChunk) {
					nnv = 0; //reset bad frame counter
					lastframe = *currentChunk->frames.back();
					//lastframe.unique_id = frame_id_counter; //not sure if this has side effects. atm duplicate has same id
					lastframe.worldtransset = false;
					lastframe.duplicate = true;
					lastframe.chunktransform = getIdentity();
					lastframe.setFrametoWorldTrans(getIdentity());
					lastframe.pushedIntoIntegrationBuffer = false;
					currentChunk->generateChunkKeypoints(); // must happen after lastframe was build
					currentChunk->generateFrustum();
					currentChunk->deleteStuff();//todo implement
					m.chunks.push_back(currentChunk); //add chunk to model
					cci = m.chunks.size() - 1;
					currentChunk->modelindex = cci;
					utility::LogDebug("Chunk number {}  has {} efficient kps\n", m.chunks.size(), currentChunk->orbKeypoints.size());
					utility::LogDebug("model stuff with chunk number {} \n", cci + 1);



//################################# build local group ###################################
					//note: new chunk is already in model
					vector<shared_ptr<Chunk>> localGroup;
					if (lastChunkValid && m.chunks.size() > 5) { //last chunk valid 
						currentChunk->frustum->reposition(m.chunks[m.chunks.size() -2]->frames.back()->getFrametoWorldTrans());
						 getLocalGroup(m,currentChunk,localGroup); //new chunk might have identity trans here
						 //if (m.chunks.size() == 60)
							// visualization::DrawGeometries({getOrigin(), getvisFrusti(m,*currentChunk->frustum)});
					} else {
						localGroup = m.chunks; //todo dont copy here 
					}
					//utility::LogInfo("chunks size: {} \n", m.chunks.size());
					//localGroup = m.chunks; 
		
//############################################# do all the filters for chunks ####################################
					npassm = 0;
					//Timer t2("match and filtering");
					for (int j = 0; j < localGroup.size(); j++) {
						if (localGroup[j] != m.chunks[cci]) {
							int Mind = localGroup[j]->modelindex;
							getCors(static_pointer_cast<KeypointUnit>(localGroup[j]), static_pointer_cast<KeypointUnit>(m.chunks[cci]), m.filteredmatches(Mind, cci));
							//m.rawmatches(j, cci) = m.filteredmatches(j, cci); //todo just for debug
							kabschfilter(m.filteredmatches(Mind, cci), m.pairTransforms(Mind, cci));
							if (m.filteredmatches(Mind, cci).size() != 0)
								reprojectionfilter(localGroup[j]->frames[0], m.chunks[cci]->frames[0], m.pairTransforms(Mind, cci), m.filteredmatches(Mind, cci)); //this can cause false negatives. todo
							if (m.filteredmatches(Mind, cci).size() != 0){
								npassm += 1;
								//cout << " cci " << cci << " Mind " << Mind << endl;
							}
						} 

					}
					//t2.~Timer();
					//utility::LogInfo("local group size: {} \n", localGroup.size());

				//	utility::LogInfo("{} matches have been found\n", npassm);

//############################################# do optimization ################################################

					bool ChunkhasCors = false;
					if (!firstchunk) {
						ChunkhasCors = checkValid(m);
						if (ChunkhasCors) {
							//do sparse align
							utility::LogInfo("Chunk number in opt {}\n", m.chunks.size());

#ifdef USEGPUSOLVER
							m.doChunktoModelforNewestChunk();
							//if more than 2 secs have passed start new global
							auto tmpTime =std::chrono::high_resolution_clock::now();
							if (std::chrono::duration_cast<std::chrono::milliseconds>(tmpTime-  timeofLastGlobalOpt).count() > 2000 && !g_solverRunning && copiedSolverResults) {
								timeofLastGlobalOpt = tmpTime;
								startGlobalsolve(m,solver,globalSolverInit);
								copiedSolverResults = false;
							}
#else
							m.doChunktoModelforNewestChunk();
							//m.performSparseOptimization(getInitialDofs(m));
#endif// USEGPUSOLVER

							//for debugging
							//m.generateEfficientStructures();
							//solver.constructGPUCors(); //now stuff is in gpu memory
							//solver.solve(dofs);
							//transferGlobalResults(m,solver);
							//cout <<"custom costs after opt " <<  m.getCost2() << endl;

							bool highresapplied =true; //todo really check
							if (highresapplied) {
								ChunkhasCors = checkValid(m);
								if (!ChunkhasCors) {
									utility::LogError("Chunk number {} is invalid after high res filter \n", cci+1);
								}
							}
						}
//####################################### chunk is accepted ######## set world trans and mark frames for integration and update reintegration buffer
						if (ChunkhasCors) {
							utility::LogDebug("Chunk added to model \n");
							lastChunkValid = true;

							//check if global solve has finished and update accordingly
							//make sure solve is not running parallel here
#ifdef USEGPUSOLVER
							if (!copiedSolverResults && !g_solverRunning) {
								transferGlobalResults(m,solver);
								copiedSolverResults =true;
							}
#endif

							//integration ##################
							pandia_integration::integrationlock.lock();//for thread safety. Changing position during integration can result in invalid matrices (has never actually happended)
							m.setWorldTransforms();
							for (int j = 0; j < nchunk; j++) {
								auto & f = m.chunks.back()->frames[j];
								if (!f->duplicate && !f->pushedIntoIntegrationBuffer){
									pandia_integration::integrationBuffer.push_back(f);
								}
							}
							pandia_integration::updateReintegrationBuffer();
							pandia_integration::integrationlock.unlock();

						} else { //no valid new chunk 
							lastChunkValid = false;
							if (m.chunks.size() > 1) { //mature model
								m.invalidChunks.push_back(m.chunks.back());
								//erase chunk from model
								for (int j = 0; j < m.chunks.back()->frames.size(); j++) {
									if (m.chunks.back()->frames[j]->pushedIntoIntegrationBuffer)
										pandia_integration::removeFrameFromIntegration(m.chunks.back()->frames[j]);
								}
								m.chunks.erase(m.chunks.end() - 1);
								utility::LogDebug("Put chunk on invalid list \n");
							} else { //young model kill everything todo make this stable for live integration
								//m = Model();
								//m.invalidChunks = vector<shared_ptr<Chunk>>();
								//firstframe = true;
								//firstchunk = true;
								//nnv = 0;
								//nvalidFrameIndices.clear();
								//cci = 0;
								//cfi = 0;
								//completeChunk = false;
								//frame_id_counter.id = 0;
								//utility::LogWarning("First Chunks not all valid. Restarted Model! \n");
							}
						}

					} else { //its the first chunk
						firstchunk = false;
						lastChunkValid = true;
						utility::LogDebug("First Chunk added to model \n");
						m.chunks[0]->chunktoworldtrans = getIdentity();
						pandia_integration::integrationlock.lock(); 
						for (int k = 0; k < m.chunks[0]->frames.size(); k++) {
							auto& f = m.chunks[0]->frames[k];
							f->setFrametoWorldTrans(f->chunktransform);
							f->worldtransset = true;
						}
						for (int k = 0; k < m.chunks[0]->frames.size(); k++) {
							if (!m.chunks[0]->frames[k]->pushedIntoIntegrationBuffer) {
								pandia_integration::integrationBuffer.push_back(m.chunks[0]->frames[k]);
							}
						}
						pandia_integration::updateReintegrationBuffer();
						pandia_integration::integrationlock.unlock();

					}

					utility::LogDebug("End Model stuff \n");
				}
			} else {
				nnv += nvalidFrameIndices.size();
				for (int j = 0; j < nvalidFrameIndices.size(); j++) {
					utility::LogWarning("Frame number {} is invalid\n", nvalidFrameIndices[j]);
				}
			}
		}
		i++;

	pauseStateLabel:
		bool frames_removed_inPause = false;

		while (g_pause) {

			//remove kabsch integrated frames
			if(!frames_removed_inPause){
				if (currentChunk->frames.size() != nchunk || nvalidFrameIndices.size() != 0) {
					for (auto f : currentChunk->frames) {
						if (f->pushedIntoIntegrationBuffer) {
							pandia_integration::removeFrameFromIntegration(f);
						}
					}
				}
				frames_removed_inPause = true;
			}
#ifdef USEGPUSOLVER
			//solver
			if (!copiedSolverResults && !g_solverRunning) {
				transferGlobalResults(m,solver);
				copiedSolverResults =true;
				//set new world coords
				pandia_integration::integrationlock.lock();//for thread safety. Changing position during integration can result in invalid matrices (has never actually happended)
				m.setWorldTransforms();
				pandia_integration::updateReintegrationBuffer();
				pandia_integration::integrationlock.unlock();
			}

			//there are chunks which are not part of the last optimization
			if (!g_solverRunning && m.chunks.size() != solver.nChunks && m.chunks.size() > 1) { 
				//optimize with new chunk
				startGlobalsolve(m,solver,globalSolverInit);
				copiedSolverResults = false;
			}
#endif 
			std::this_thread::sleep_for(20ms); 
		}

	}


#ifdef	USEGPUSOLVER
	//############################ doing final opt stuff #########################
	//check if current opt is recent otherwise do new opt
	while (g_solverRunning) {
		this_thread::sleep_for(20ms);
	}
	if (!copiedSolverResults) {
		transferGlobalResults(m,solver);
		copiedSolverResults =true;
	}
	cout << "Final number of chunks in solver is "<< solver.nChunks << endl;
	//if (!g_solverRunning && m.chunks.size() != solver.nChunks && m.chunks.size() > 1) { 
	//	utility::LogWarning("Final solve called after slam ended. might be indicative of unwanted behaviour \n");
	//	//run optimization in this thread
	//	vector<Eigen::Vector6d> localDofs;
	//	for (int i = 0; i < m.chunks.size(); i++) {
	//		localDofs.push_back(MattoDof(m.chunks[i]->chunktoworldtrans));
	//	}
	//	m.generateEfficientStructures();
	//	solver.constructGPUCors(); //now stuff is in gpu memory
	//	solver.solve(localDofs);
	//	transferGlobalResults(m,solver);
	//}

	//there are chunks which are not part of the last optimization
	if (m.chunks.size() != solver.nChunks && m.chunks.size() > 1) { 
		//optimize with new chunk
		//m.chunks[15]->chunktoworldtrans = Eigen::Matrix4d::Identity();
		startGlobalsolve(m,solver,globalSolverInit);
		copiedSolverResults = false;
	}

	cout << "Final number of chunks in solver after final solve is "<< solver.nChunks << endl;
	while (g_solverRunning) {
		this_thread::sleep_for(20ms);
	}

	if (!copiedSolverResults) {
		transferGlobalResults(m,solver);
		copiedSolverResults =true;
	}
#endif
	pandia_integration::integrationlock.lock(); 
	m.setWorldTransforms(); //superflous if no new global solve after slam finish
	pandia_integration::updateReintegrationBuffer();
	pandia_integration::integrationlock.unlock(); 
	
	g_current_slam_finished = true;


	//remove the frames which have been integrated last and don't fill a chunk
	if (currentChunk->frames.size() != nchunk || nvalidFrameIndices.size() != 0) {
		for (auto f : currentChunk->frames) {
			if (f->pushedIntoIntegrationBuffer) {
				pandia_integration::removeFrameFromIntegration(f);
			}
		}
	}

	//################## ending threads ######################
	stopRecording = true;
	stopSegment = true;
	cameraThread->join();
	if (g_segment)
		segThread->join();

#ifdef USEGPUSOLVER
	globalSolverThread.join();
#endif // USEGPUSOLVER



	pandia_integration::stopintegrating = true;
	if (integrate)
		integrationThread.join();

	//reset

	//for deallocation
	pandia_integration::integratedframes.clear(); //doesnt happen otherwise
	pandia_integration::deintegrationBuffer.clear();
	pandia_integration::reintegrationBuffer.clear();
	pandia_integration::integrationBuffer.clear();


	m.recordbuffer.clear(); //only ever used in pause mode, so clearing is allowed

	stopVisualizing = true;
	if(livevis)
		visThread.join();

	//reset


	std::cout << "All recon threads joined \n";

	resetThreadVars(); //for possible next launch

	g_reconThreadFinished = true;


	return 0;

}