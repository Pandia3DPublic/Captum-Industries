#include "CameraKinect.h"
#include <mutex>
#include "configvars.h"
#include "utils/coreutil.h"
#include "utils/matrixutil.h"
#include <Cuda/Camera/PinholeCameraIntrinsicCuda.h>
#include "genKps.h"

using namespace cv;
using namespace std;

static mutex corelock;


void printCalibrationDepth(k4a::calibration& calib)
{
	auto color = calib.color_camera_calibration;
	auto depth = calib.depth_camera_calibration;
	cout << "\n===== Device Calibration information (depth) =====\n";
	cout << "color resolution width: " << color.resolution_width << endl;
	cout << "color resolution height: " << color.resolution_height << endl;
	cout << "depth resolution width: " << depth.resolution_width << endl;
	cout << "depth resolution height: " << depth.resolution_height << endl;
	cout << "principal point x: " << depth.intrinsics.parameters.param.cx << endl;
	cout << "principal point y: " << depth.intrinsics.parameters.param.cy << endl;
	cout << "focal length x: " << depth.intrinsics.parameters.param.fx << endl;
	cout << "focal length y: " << depth.intrinsics.parameters.param.fy << endl;
	cout << "radial distortion coefficients:" << endl;
	cout << "k1: " << depth.intrinsics.parameters.param.k1 << endl;
	cout << "k2: " << depth.intrinsics.parameters.param.k2 << endl;
	cout << "k3: " << depth.intrinsics.parameters.param.k3 << endl;
	cout << "k4: " << depth.intrinsics.parameters.param.k4 << endl;
	cout << "k5: " << depth.intrinsics.parameters.param.k5 << endl;
	cout << "k6: " << depth.intrinsics.parameters.param.k6 << endl;
	cout << "center of distortion in Z=1 plane, x: " << depth.intrinsics.parameters.param.codx << endl;
	cout << "center of distortion in Z=1 plane, y: " << depth.intrinsics.parameters.param.cody << endl;
	cout << "tangential distortion coefficient x: " << depth.intrinsics.parameters.param.p1 << endl;
	cout << "tangential distortion coefficient y: " << depth.intrinsics.parameters.param.p2 << endl;
	cout << "metric radius: " << depth.intrinsics.parameters.param.metric_radius << endl;
	cout << endl;

}
void printCalibrationColor(k4a::calibration& calib)
{
	auto color = calib.color_camera_calibration;
	auto depth = calib.depth_camera_calibration;
	cout << "\n===== Device Calibration information (color) =====\n";
	cout << "color resolution width: " << color.resolution_width << endl;
	cout << "color resolution height: " << color.resolution_height << endl;
	cout << "depth resolution width: " << depth.resolution_width << endl;
	cout << "depth resolution height: " << depth.resolution_height << endl;
	cout << "principal point x: " << color.intrinsics.parameters.param.cx << endl;
	cout << "principal point y: " << color.intrinsics.parameters.param.cy << endl;
	cout << "focal length x: " << color.intrinsics.parameters.param.fx << endl;
	cout << "focal length y: " << color.intrinsics.parameters.param.fy << endl;
	cout << "radial distortion coefficients:" << endl;
	cout << "k1: " << color.intrinsics.parameters.param.k1 << endl;
	cout << "k2: " << color.intrinsics.parameters.param.k2 << endl;
	cout << "k3: " << color.intrinsics.parameters.param.k3 << endl;
	cout << "k4: " << color.intrinsics.parameters.param.k4 << endl;
	cout << "k5: " << color.intrinsics.parameters.param.k5 << endl;
	cout << "k6: " << color.intrinsics.parameters.param.k6 << endl;
	cout << "center of distortion in Z=1 plane, x: " << color.intrinsics.parameters.param.codx << endl;
	cout << "center of distortion in Z=1 plane, y: " << color.intrinsics.parameters.param.cody << endl;
	cout << "tangential distortion coefficient x: " << color.intrinsics.parameters.param.p1 << endl;
	cout << "tangential distortion coefficient y: " << color.intrinsics.parameters.param.p2 << endl;
	cout << "metric radius: " << color.intrinsics.parameters.param.metric_radius << endl;
	cout << endl;

}

//corethread to push raw framesets and imu data to respective buffers 
void coreThread(k4a::device* kinect, list <shared_ptr<k4a::capture>>* rawbuffer, list <shared_ptr<k4a_imu_sample_t>>* imubuffer, std::atomic<bool>* stop) {
	bool success = true;
	while (!(*stop))
	{
		k4a::capture capture;
		k4a_imu_sample_t imu;
		//do not check for connection or parameterSet, else you will be stuck in this loop
		while (g_pause) {
			this_thread::sleep_for(20ms);
		}
		try {//prevents capturing from invalid/disconnected device

			//retrieves capture and imu data from device
			g_deviceKinect.get_capture(&capture);
			g_deviceKinect.get_imu_sample(&imu); //currently blocking

			//retrieves imu data which has roughly the same timestamp as the capture (semi-synchronized)
			//roughly between 0 and 400 microseconds disparity of capture and imu timestamp
			while ((int)capture.get_color_image().get_device_timestamp().count() > (int)imu.acc_timestamp_usec) {
				g_deviceKinect.get_imu_sample(&imu); //currently blocking
			}
			success = true;
		}
		catch (...) {
			cout << "connection lost \n";
			success = false;
		}
		
		//only writes non-empty captures/imus in buffer
		if (success) {
			corelock.lock();
			rawbuffer->push_back(make_shared<k4a::capture>(capture));
			imubuffer->push_back(make_shared<k4a_imu_sample_t>(imu));
			corelock.unlock();
		}
	}
}
//this function takes the heavy load of data preparation
//todo kinect conversion is complicated and there should be some pcd generation in there somewhere. 
//		For really high performance this can be used.
//todo check if gpu support is enabled for tranform functiosn of azure sdk
//note: init kamera must be called beforehand
void KinectThread(list <shared_ptr<Frame>>& framebuffer, std::atomic<bool>& stop, std::atomic<bool>& cameraParameterSet)
{

	using namespace cv;
	g_warmupCamera = true; //for gui-indicator
	//start record thread, rawbuffer for raw camera frames
	list <shared_ptr<k4a::capture>> rawbuffer;
	list <shared_ptr<k4a_imu_sample_t>> imubuffer;
	k4a::transformation transform(g_calibrationKinect);

	//Warmup for ca. 1 second
	k4a::capture captureWarmup;
	for (int i = 0; i < 60; i++) {
		g_deviceKinect.get_capture(&captureWarmup); //dropping several frames for auto-exposure
		g_warmupProgress++;
	}
	captureWarmup.reset();
	g_warmupCamera = false;
	g_warmupProgress = 0;
	
	//coreThreads must be startetd after captureWarmup to avoid that first 30 frames get pushed in rawbuffer
	shared_ptr<std::thread> cThread = make_shared<std::thread>(coreThread, &g_deviceKinect, &rawbuffer, &imubuffer, &stop); //startet neuen thread und liest bidler ein
	
	//Camera Loop
	while (!stop)
	{
		while (rawbuffer.empty() && !stop) {
			std::this_thread::sleep_for(20ms);
		}

		if (stop) {
			break;
		}

		
		//Get capture from raw image buffer and from imu-camera data
		corelock.lock();
		auto capture = rawbuffer.front();
		rawbuffer.pop_front();
		auto imu = imubuffer.front();
		imubuffer.pop_front();
		corelock.unlock();

		//get color and aligned depth image
		//this produces an image that is distorted accoring to the color camera distortion, which should be near to pinhole
		k4a::image color = capture->get_color_image();
		auto tmpdepth = capture->get_depth_image();
		//takes 3-6ms long, should be gpu supported (untested)
		k4a::image transDepth = transform.depth_image_to_color_camera(tmpdepth); //depth image now aligned to color! 
		//todo: Note the color image is not a perfect pinhole camera. For ideal results there should be one more distortion step for both images
		//To get really fast this should be done on downsized images (optional).
		//To get even faster, skip the opencv images and work directly on k4a images (very optional)

		//############################ transform to scaled open3d images #################
		// Create OpenCV Mat from Kinect Image
		//opencv stuff takes 1.5ms
		//note this goes out of scope
		Mat cvColor4 = Mat(Size(color.get_width_pixels(), color.get_height_pixels()), CV_8UC4, (void*)color.get_buffer(), Mat::AUTO_STEP);
		Mat cvCol3 = Mat(Size(color.get_width_pixels(), color.get_height_pixels()), CV_8UC3);
		cvtColor(cvColor4, cvCol3, COLOR_BGRA2BGR); //remove alpha channel
		Mat cvDepth = Mat(Size(transDepth.get_width_pixels(), transDepth.get_height_pixels()), CV_16U, (void*)transDepth.get_buffer(), Mat::AUTO_STEP);
		//note the global intrinsics have to be adjusted to these numbers
		resize(cvCol3, cvCol3, Size(g_resx, g_resy));
		resize(cvDepth, cvDepth, Size(g_resx, g_resy), 0, 0, INTER_NEAREST); //no interpol since this will give artifacts for depth images

		shared_ptr<geometry::Image> color_image = make_shared<geometry::Image>();
		shared_ptr<geometry::Image> depth_image = make_shared<geometry::Image>();

		//convert to open3d image
		color_image->Prepare(cvCol3.cols, cvCol3.rows, 3, 1);
		//memcpy doesnt work here since color channels are the wrong way round
		//if (cvCol3.isContinuous()) {
		//	memcpy(color_image_8bit.data_.data(), cvCol3.data, cvCol3.total() * cvCol3.elemSize());
		//}
		//else {
#pragma omp parallel for 
		for (int y = 0; y < cvCol3.rows; y++) {
			uint8_t* pixel = cvCol3.ptr<uint8_t>(y); // point to first color in row
			for (int x = 0; x < cvCol3.cols; x++) {
				*color_image->PointerAt<uint8_t>(x, y, 2) = *pixel++;
				*color_image->PointerAt<uint8_t>(x, y, 1) = *pixel++;
				*color_image->PointerAt<uint8_t>(x, y, 0) = *pixel++;
			}
		}
		//}

		depth_image->Prepare(cvDepth.cols, cvDepth.rows, 1, 2);
		if (cvDepth.isContinuous()) {
			memcpy(depth_image->data_.data(), cvDepth.data, cvDepth.total() * cvDepth.elemSize());
		}
		else {
#pragma omp parallel for
			for (int y = 0; y < cvDepth.rows; y++) {
				uint16_t* pixel_d = cvDepth.ptr<uint16_t>(y); //point to first pixel in row
				for (int x = 0; x < cvDepth.cols; x++) {
					*depth_image->PointerAt<uint16_t>(x, y) = *pixel_d++;
				}
			}
		}

//################## conversion done

		auto tmp = std::make_shared<Frame>();
		tmp->imuSample = imu;
		//6-15ms
		generateFrame(color_image, depth_image, tmp);
		//time
		generateOrbKeypoints(tmp,cvCol3);

		g_bufferlock.lock();
		framebuffer.push_back(tmp);
		g_bufferlock.unlock();

	}
	
	cThread->join();

}