#include "CameraRealsense.h"
#include <mutex>
#include "configvars.h"
#include "utils/coreutil.h"
#include "utils/matrixutil.h"
#include <atomic>
using namespace cv;

static mutex corelock;


enum Preset
{
	Custom = 0,
	Default = 1,
	Hand = 2,
	HighAccuracy = 3,
	HighDensity = 4,
	MediumDensity = 5
};

void checkmaxmin(int max, int min, int& test) {
	if (test > max) {
		test = max;
		utility::LogWarning("Res too high set to maximum res.\n");
	}
	if (test < min) {
		utility::LogWarning("Res too low. Will probably result in error\n");
	}
}

//corethread to push raw framesets to buffer
void coreThread(rs2::pipeline* pipe, list <shared_ptr<rs2::frameset>>* rsframebuffer, std::atomic<bool>* stop) {
	while (!(*stop)) 
	{
		auto frames = pipe->wait_for_frames();
		corelock.lock();
		rsframebuffer->push_back(make_shared<rs2::frameset>(frames));
		corelock.unlock();
	}
}

//todo adjust for different color and depth res
void RealsenseThread(list <shared_ptr<Frame>> & framebuffer, std::atomic<bool>&stop, std::atomic<bool>& cameraParameterSet)
{
	g_warmupCamera = true;
	// Use preset options for depth camera
	auto depth_sensor = g_profileRealsense.get_device().first<rs2::depth_sensor>();
	depth_sensor.set_option(rs2_option::RS2_OPTION_VISUAL_PRESET, Default);

	rs2::disparity_transform depth_to_disparity(true);
	rs2::disparity_transform disparity_to_depth(false);
	// Declare filters
	rs2::decimation_filter dec_filter;  // Decimation - reduces depth frame density
	rs2::threshold_filter thr_filter;   // Threshold  - removes values outside recommended range
	rs2::spatial_filter spat_filter;    // Spatial    - edge-preserving spatial smoothing
	rs2::temporal_filter temp_filter;   // Temporal   - reduces temporal noise
										// Declare disparity transform from depth to disparity and vice versa
	rs2::align align_to_color(RS2_STREAM_COLOR);
	//start core thread, rsframebuffer for raw camera frames
	list <shared_ptr<rs2::frameset>> rsframebuffer;

	shared_ptr<rs2::frameset> frames = make_shared<rs2::frameset>();
	for (int i = 0; i < 30; i++) {
		*frames = g_pipeRealsense.wait_for_frames();
		g_warmupProgress++;
	}

	std::thread coreThread(coreThread, &g_pipeRealsense, &rsframebuffer, &stop); //startet neuen thread und liest bidler ein
	g_warmupCamera = false;
	g_warmupProgress = 0;


	// Camera Loop with processing
	while (!stop) 
	{
		while (rsframebuffer.empty()) {
			std::this_thread::sleep_for(20ms);
		}
		//Get frameset from raw image buffer
		corelock.lock();
		frames = rsframebuffer.front();
		rsframebuffer.pop_front();
		corelock.unlock();
		
		//Alignment
		*frames = align_to_color.process(*frames);

		//Get each frame
		auto color_frame = frames->get_color_frame();
		auto depth_frame = frames->get_depth_frame();

		// Apply filters
		//depth_frame = dec_filter.process(depth_frame); //todo write reasonable decimation filter ourselves
		depth_frame = thr_filter.process(depth_frame);
		depth_frame = depth_to_disparity.process(depth_frame);
		depth_frame = spat_filter.process(depth_frame);
		//depth_frame = temp_filter.process(depth_frame);
		depth_frame = disparity_to_depth.process(depth_frame);

		// Create OpenCV Matrix from frame data
		Mat col = Mat(Size(color_frame.get_width(), color_frame.get_height()), CV_8UC3, (void*)color_frame.get_data(), Mat::AUTO_STEP);
		Mat cvDepth(Size(depth_frame.get_width(), depth_frame.get_height()), CV_16UC1, (void*)depth_frame.get_data(), Mat::AUTO_STEP);
		
		geometry::Image color_image_8bit;
		geometry::Image depth_image_16bit;
		std::shared_ptr<geometry::Image> color_image_pointer;
		std::shared_ptr<geometry::Image> depth_image_pointer;

		// Convert OpenCV Mat to Open3d Image
		// Color
		color_image_8bit.Prepare(color_frame.get_width(), color_frame.get_height(), 3, 1);
#pragma omp parallel for
		for (int y = 0; y < color_frame.get_height(); y++) {
			uint8_t* pixel = col.ptr<uint8_t>(y); // point to first color in row
			for (int x = 0; x < color_frame.get_width(); x++) {
				*color_image_8bit.PointerAt<uint8_t>(x, y, 0) = *pixel++;
				*color_image_8bit.PointerAt<uint8_t>(x, y, 1) = *pixel++;
				*color_image_8bit.PointerAt<uint8_t>(x, y, 2) = *pixel++;
			}
		}
		// Depth
		depth_image_16bit.Prepare(depth_frame.get_width(), depth_frame.get_height(), 1, 2);
#pragma omp parallel for
		for (int y = 0; y < depth_frame.get_height(); y++) {
			uint16_t* pixel_d = cvDepth.ptr<uint16_t>(y); //point to first pixel in row
			for (int x = 0; x < depth_frame.get_width(); x++) {
				*depth_image_16bit.PointerAt<uint16_t>(x, y) = *pixel_d++;
			}
		}

		color_image_pointer = std::make_shared<geometry::Image>(color_image_8bit);
		depth_image_pointer = std::make_shared<geometry::Image>(depth_image_16bit);

		auto tmp = make_shared<Frame>();
		generateFrame(color_image_pointer, depth_image_pointer, tmp);

		g_bufferlock.lock();
		framebuffer.push_back(tmp);
		g_bufferlock.unlock();

	}
	coreThread.join();
}

