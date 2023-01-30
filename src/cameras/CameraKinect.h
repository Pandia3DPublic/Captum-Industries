#pragma once

#include <iostream>
#include <Open3D/Open3D.h>
#include <opencv2/opencv.hpp>
#include "core/Frame.h"
#include "Gui/guiutil.h"
#include "opencv2/core.hpp"
#include "utils/imageutil.h"
#include <atomic>
#include <k4a/k4a.hpp> //Azure Kinect C++ wrapper
#include "CameraThreadandInit.h"

void KinectThread(list <shared_ptr<Frame>>& framebuffer, std::atomic<bool>& stop, std::atomic<bool>& cameraParameterSet);
