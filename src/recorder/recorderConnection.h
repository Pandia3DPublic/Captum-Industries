#pragma once
#include <iostream>
#include <Open3D/Camera/PinholeCameraIntrinsic.h>
#include "k4a/k4a.h"
#include "k4a/k4a.hpp"
#include "utils/coreutil.h"
#include "Gui/imgui-docking/FileBrowser/ImGuiFileBrowser.h"


namespace recorder{
	
extern int recorderDeviceCount;
extern k4a::device recorderDevice;
extern k4a::calibration recorderCalibration;
extern open3d::camera::PinholeCameraIntrinsic recorderIntrinsic;
extern std::atomic<bool> parameterSet;
extern std::atomic<bool> closeProgram;
//extern std::list<k4a::capture> captureBuffer;
extern std::mutex bufferLock;
extern std::mutex colorBufferLock;
extern std::mutex depthBufferLock;
extern std::atomic<bool> warmingUp;

extern imgui_addons::ImGuiFileBrowser saveImages;
extern string imagePath;
extern int choose_PNG_JPG;






void recorderConnectionThreadFunction();
void saveImageThreadFunction(std::list<shared_ptr<k4a::capture>>& captureBuffer);
void saveColorImageThreadFunction(std::list<shared_ptr<k4a::image>>& colorBuffer);
void saveDepthImageThreadFunction(std::list<shared_ptr<k4a::image>>& depthBuffer);

////void initialiseRecorder();

}