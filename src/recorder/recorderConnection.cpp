#include "recorder/recorderConnection.h"
#include <thread>
#include <chrono>
#include <direct.h> //mkdir

#include "ptimer.h"

using namespace std;
using namespace open3d;


namespace recorder {

	int recorderDeviceCount = 0;
	k4a::device recorderDevice;
	k4a::calibration recorderCalibration;
	open3d::camera::PinholeCameraIntrinsic recorderIntrinsic;
	int resx = 640;
	int resy = 480;
	std::atomic<bool> parameterSet(false);
	std::atomic<bool> closeProgram(false);
	//std::list<shared_ptr<k4a::capture>> captureBuffer;
	std::mutex bufferLock;
	std::mutex colorBufferLock;
	std::mutex depthBufferLock;
	std::atomic<bool> warmingUp = true;


	imgui_addons::ImGuiFileBrowser saveImages;
	string imagePath = "";
	int choose_PNG_JPG = 0;




void initialiseRecorder() {
	//recorderDevice.stop_cameras();
	recorderDevice.close();
	bool connected = false;

	while (!connected) {

		try {
			recorderDevice = recorderDevice.open(K4A_DEVICE_DEFAULT); //throws exception if there is no device
			//executes next lines if there is no exception
			cout << "Azure Kinect device opened!" << endl;
			connected = true;
		}
		catch (k4a::error e) {
			//cerr << e.what() << endl;
			cout << "Retry connecting Kinect camera" << endl;
			this_thread::sleep_for(1000ms);
		}

	}


	//Configuration parameters
	//note: nfov and wfov have different field of views. Binned and unbinned have the same, but with different resolutions and more/less jitter and noise
	k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
	config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32; //note this takes processing power since some stupid mjpeg format is native.
	config.color_resolution = K4A_COLOR_RESOLUTION_1536P; //this is the smallest 4:3 resolution which has max overlap with depth
	config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED; //res is 640x576 for K4A_DEPTH_MODE_NFOV_UNBINNED
	config.camera_fps = K4A_FRAMES_PER_SECOND_30;
	config.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE;
	config.synchronized_images_only = true;

	recorderDevice.start_cameras(&config);
	//recorderDevice.start_imu();

	recorderCalibration = recorderDevice.get_calibration(config.depth_mode, config.color_resolution);

	auto ccc = recorderCalibration.color_camera_calibration;
	auto param = ccc.intrinsics.parameters.param;
	camera::PinholeCameraIntrinsic intrinsic(ccc.resolution_width, ccc.resolution_height, param.fx, param.fy, param.cx, param.cy);

	//note that the input pictures have to be scalled to this resolution
	recorderIntrinsic = getScaledIntr(intrinsic, resx, resy);
	recorderIntrinsic.width_ = resx;
	recorderIntrinsic.height_ = resy;
	//g_lowIntr = getScaledIntr(g_intrinsic, g_lowx, g_lowy); //todo get rid of getlowintr


	//save intrinsic
	/*ofstream intrinsicFile("intrinsic.txt");
	intrinsicFile << intrinsic.width_ << endl << intrinsic.height_ << endl << intrinsic.intrinsic_matrix_;
	intrinsicFile.close();*/

	//cout << "INTRINSIC:" << endl;
	//cout << recorderIntrinsic.width_ << endl << recorderIntrinsic.height_ << endl << recorderIntrinsic.intrinsic_matrix_ << endl;

	parameterSet = true;
}


void recorderConnectionThreadFunction() {

	while (!closeProgram) {
		try {
			recorderDeviceCount = recorderDevice.get_installed_count();
		}
		catch (...) {
			std::cout << "cannot access k4a-device" << std::endl;
			recorderDeviceCount = 0;
		}


		if (recorderDeviceCount == 0) {
			parameterSet = false;
		}

		if (recorderDeviceCount > 0 && !parameterSet) {
			initialiseRecorder();
		}

	
		if(!closeProgram)
			std::this_thread::sleep_for(1000ms);
		
	}

}



void saveColorImageThreadFunction(std::list<shared_ptr<k4a::image>>& colorBuffer) {

	using namespace cv;
	int res_x = 640;
	int res_y = 480;
	int counter = 0;
	//resolution from azure kinect. Set in device_configuration
	int camera_x = 2048;
	int camera_y = 1536;


	while (!recorder::closeProgram) {

		while (colorBuffer.empty() && !recorder::closeProgram) {
			std::this_thread::sleep_for(20ms);
		}

		if (recorder::closeProgram) {
			break;
		}


		colorBufferLock.lock();
		auto color = colorBuffer.front();
		colorBuffer.pop_front();
		colorBufferLock.unlock();

		Mat cvColor4 = Mat(Size(color->get_width_pixels(), color->get_height_pixels()), CV_8UC4, (void*)color->get_buffer(), Mat::AUTO_STEP);
		static Mat cvCol3 = Mat(Size(color->get_width_pixels(), color->get_height_pixels()), CV_8UC3);
		cvtColor(cvColor4, cvCol3, COLOR_BGRA2BGR); //remove alpha channel
		resize(cvCol3, cvCol3, Size(res_x, res_y));

		shared_ptr<geometry::Image> color_image = make_shared<geometry::Image>();

		//convert to open3d image
		color_image->Prepare(cvCol3.cols, cvCol3.rows, 3, 1);

#pragma omp parallel for 
		for (int y = 0; y < cvCol3.rows; y++) {
			uint8_t* pixel = cvCol3.ptr<uint8_t>(y); // point to first color in row
			for (int x = 0; x < cvCol3.cols; x++) {
				*color_image->PointerAt<uint8_t>(x, y, 2) = *pixel++;
				*color_image->PointerAt<uint8_t>(x, y, 1) = *pixel++;
				*color_image->PointerAt<uint8_t>(x, y, 0) = *pixel++;
			}
		}

		if (choose_PNG_JPG == 0) {
			io::WriteImage(imagePath + "\\color/" + getPicNumberString(counter) + ".png", *color_image);
		}
		else {
			io::WriteImage(imagePath + "\\color/" + getPicNumberString(counter) + ".jpg", *color_image);
		}

		counter++;
	}

}


void saveDepthImageThreadFunction(std::list<shared_ptr<k4a::image>>& depthBuffer) {

	using namespace cv;
	int res_x = 640;
	int res_y = 480;
	int counter = 0;
	k4a::transformation transform(recorder::recorderCalibration);


	while (!recorder::closeProgram) {
		while (depthBuffer.empty() && !recorder::closeProgram) {
			std::this_thread::sleep_for(20ms);
		}


		if (recorder::closeProgram) {
			break;
		}

		recorder::depthBufferLock.lock();
		auto depth = depthBuffer.front();
		depthBuffer.pop_front();
		recorder::depthBufferLock.unlock();

		depth = make_shared<k4a::image>(transform.depth_image_to_color_camera(*depth));


		Mat cvDepth = Mat(Size(depth->get_width_pixels(), depth->get_height_pixels()), CV_16U, (void*)depth->get_buffer(), Mat::AUTO_STEP);
		resize(cvDepth, cvDepth, Size(res_x, res_y), 0, 0, INTER_NEAREST); //no interpol since this will give artifacts for depth images

		shared_ptr<geometry::Image> depth_image = make_shared<geometry::Image>();

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

		io::WriteImage(imagePath + "\\depth/" + getPicNumberString(counter) + ".png", *depth_image);
		counter++;
	}
}


}