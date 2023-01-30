#pragma once
#include <mutex>
#include <atomic>
//#include "../Server/Server.h" //this must be included before CameraKinect.h since camerakinect includes open3d.h which includes  windows.h which must not be included before winsock2.h in Server todo

//this file contains mutex lock and atomic variables for thread communication
extern std::mutex seglock; //segmentation lock
extern std::mutex meshlock; //mesh lock
extern std::mutex solverlock; //gpu solver lock
extern std::mutex currentposlock; //just for the current position


extern std::atomic<bool> stopMeshing; //signaling bool to stop the meshing thread
extern std::atomic<bool> g_current_slam_finished;
extern std::atomic<bool> g_trackingLost;
extern std::atomic<bool> g_reconThreadFinished;
extern std::atomic<bool> g_wholeMesh; //indicates that the recon thread and all its subthreads finished

//solver communication
//indicates that the global solver is running. is set to start the solver
extern std::atomic<bool> g_solverRunning;
extern std::atomic<bool> g_take_dataCam; //take hardrive data

//programm state enum, used for gui-recon communication
const enum programStates {
	gui_READY = 0, // we see the start button#
	gui_RUNNING, // We see the stop button, stuff is running!
	gui_PAUSE // Pause State, choose to savem mesh, resume, etc.
};
extern std::atomic<int> g_programState;


