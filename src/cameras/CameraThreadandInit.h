#pragma once
//#include "Server/Server.h"
#include "../cameras/CameraKinect.h"
#include "../cameras/CameraRealsense.h"
#include "../utils/coreutil.h"
#include <mutex>
#include <atomic>
#include <k4a/k4a.hpp>
#include <configvars.h>

extern std::atomic<bool> g_cameraConnected;
extern std::atomic<int> kinectDeviceCount;

void cameraConnectionThreadFunction();
void StartClientThreads(list <shared_ptr<Frame>>& framebuffer, std::atomic<bool>& stop);


void initialiseCamera();

