#pragma once

#include <iostream>
#include <Open3D/Open3D.h>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>
#include "core/Frame.h"
#include "opencv2/core.hpp"
#include "utils/imageutil.h"
#include <atomic>

void RealsenseThread(list <shared_ptr<Frame>>& framebuffer, std::atomic<bool>& stop, std::atomic<bool>& cameraParameterSet);