#pragma once

#undef UNICODE
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX //need this because windows.h has macros for min/max which are interfering with open3d RenderOption.h
#include <winsock2.h>
#include <windows.h>
#include <ws2tcpip.h>
#include <stdlib.h>
#include <stdio.h>
#include <atomic>
#include <Open3D/Open3D.h>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "core/Frame.h"


#include <mutex>
#include <atomic>
#include <k4a/k4a.hpp>
#include <configvars.h>


// Need to link with Ws2_32.lib
#pragma comment (lib, "Ws2_32.lib")
// #pragma comment (lib, "Mswsock.lib")
#define DEFAULT_PORT "31415"




void StartClientThreads(list <shared_ptr<Frame>>& framebuffer, std::atomic<bool>& stop);
//inits socket and calibration
int initializeServer();


