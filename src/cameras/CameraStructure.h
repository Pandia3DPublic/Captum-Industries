#pragma once

#include <iostream>
#include <Eigen/Geometry>
#include <ST/CaptureSession.h>
#include <Open3D/Open3D.h>
#include "core/Frame.h"
#include "utils/imageutil.h"
#include "utils/matrixutil.h"
#include <atomic>

class CameraStructure
{
public:
	CameraStructure();
	~CameraStructure();

	struct SessionDelegate : ST::CaptureSessionDelegate {  //we need a delegate to access frames (sample) from capturesession

		void captureSessionEventDidOccur(ST::CaptureSession* session, ST::CaptureSessionEventId event) override; //For status information of camera
		void captureSessionDidOutputSample(ST::CaptureSession*, const ST::CaptureSessionSample& sample) override; //Get sample (similary to waitforframes in realsense), push to samplebuffer

		list <shared_ptr<ST::CaptureSessionSample>> samplebuffer; //Buffer for raw samples

	};

	void align_color_to_depth(const ST::DepthFrame& depth_frame, const ST::ColorFrame& color_frame, shared_ptr<geometry::Image> color, geometry::Image& output);
	void processFrames(shared_ptr<ST::CaptureSessionSample>& sample, list <shared_ptr<Frame>>& framebuffer); //process frames (align, pcd, ...), push processed frames to framebuffer
	void startCameraThread(list <shared_ptr<Frame>>& framebuffer, std::atomic<bool>& stop, std::atomic<bool>& cameraParameterSet); //main camera loop, init camera and settings
	std::shared_ptr<Frame> getSingleFrame(list <shared_ptr<Frame>>& framebuffer); //return single frame from framebuffer

};