#include "cameras/ClientCamera/clientCamera.h" //this must be on top because win.h is stupid
#include "core/threadvars.h"
#include "CameraThreadandInit.h"
#include "ptimer.h"
#include "protos/framedata.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/util/delimited_message_util.h>
#include "genKps.h"

std::atomic<bool> g_cameraConnected(false);
std::atomic<int> kinectDeviceCount(0);

void initialiseCamera() {
	//##########################################initialize Camera ###############################################################
	
	//Create and open device
	if (g_camType == camtyp::typ_kinect) {
	
		g_deviceKinect.close(); //needs to be called to prevent errors on change of loadDataFlag
		bool connected = false;

		while (!connected) {

			if (g_closeProgram || g_take_dataCam) return;

			try {
				g_deviceKinect = g_deviceKinect.open(K4A_DEVICE_DEFAULT); //throws exception if there is no device
				//executes next lines if there is no exception
				cout << "Azure Kinect device opened!" << endl;
				connected = true;
			}
			catch (k4a::error e) {
				cerr << e.what() << endl;
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

		//Start Cameras
		//needs to be started when reconrun starts, to prevent early disconnect
		g_deviceKinect.start_cameras(&config);
		g_deviceKinect.start_imu();

		//Calibration object contains camera specific information and is used for all transformation functions
		g_calibrationKinect = g_deviceKinect.get_calibration(config.depth_mode, config.color_resolution);
		//printCalibrationColor(calibration);

		//take the color intrinsic since depth img is warped to color. Note: This assumes the camera image is a pinhole camera which is an approximation (todo maybe)
		auto ccc = g_calibrationKinect.color_camera_calibration;
		auto param = ccc.intrinsics.parameters.param;
		camera::PinholeCameraIntrinsic intrinsic(ccc.resolution_width, ccc.resolution_height, param.fx, param.fy, param.cx, param.cy);

		//note that the input pictures have to be scalled to this resolution
		g_intrinsic = getScaledIntr(intrinsic, g_resx, g_resy);
		g_intrinsic.width_ = g_resx;
		g_intrinsic.height_ = g_resy;
		g_intrinsic_cuda = open3d::cuda::PinholeCameraIntrinsicCuda(g_intrinsic);
		g_lowIntr = getScaledIntr(g_intrinsic, g_lowx, g_lowy); //todo get rid of getlowintr

		g_cameraParameterSet = true;

	}

	if (g_camType == camtyp::typ_realsense) {

		//rs2::context context_realsense;
		//rs2::device_list device_realsense;

		bool connected = false;

		rs2::config config_realsense;
		config_realsense.enable_stream(RS2_STREAM_COLOR, g_resx, g_resy, RS2_FORMAT_RGB8, 30);
		config_realsense.enable_stream(RS2_STREAM_DEPTH, g_resx, g_resy, RS2_FORMAT_Z16, 30);


		while (!connected) {

			if (g_closeProgram || g_take_dataCam) return;

			try {
				g_profileRealsense = g_pipeRealsense.start(config_realsense);
				//executes next lines if there is no exception
				cout << "Intel Realsense device opened!" << endl;
				connected = true;
			}
			catch (rs2::error e) {
				cerr << e.what() << endl;
				cout << "Retry connecting Realsense camera" << endl;
				this_thread::sleep_for(1000ms);
			}
		}

		//get intrinsics
		auto intr = g_pipeRealsense.get_active_profile().get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();
		g_intrinsic = camera::PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy);
		g_intrinsic_cuda = open3d::cuda::PinholeCameraIntrinsicCuda(g_intrinsic);
		g_lowIntr = getLowIntr(g_intrinsic);

		g_cameraParameterSet = true;

	}


	if (g_camType == camtyp::typ_data) {
		cout << "found data cam \n";
		if (fileExists(g_readimagePath + "\\intrinsic.txt")) {
			utility::LogDebug("Set intrinsics from file\n");
			setFromIntrinsicFile(g_readimagePath + "\\intrinsic.txt");
		}
		else {
			utility::LogWarning("No intrinsic file found. Taking default intrinsic\n");
			setDefaultIntrinsic();
		}

		g_cameraParameterSet = true;

	}


	if (g_camType == camtyp::typ_client) {
			cout << "taking client data cam \n";
			initializeServer();
			g_cameraParameterSet = true;
	}


	cout << "camera initialized" << endl;

}



// check if there is a camera
void cameraConnectionThreadFunction() {

	//Create and open device
	rs2::context context_realsense;
	rs2::device_list device_realsense;

	while (!g_closeProgram) {//Loops as long as there is no device found
		g_cameraConnected = false;
		if (!g_clientdata) { // to prevent conflict with local camera if client is localhost
			//kinect
			try {
				kinectDeviceCount = g_deviceKinect.get_installed_count();
			}
			catch (...) {
				cout << "cannot access k4a-device" << endl;
				kinectDeviceCount = 0;
			}

			if (kinectDeviceCount > 0) {
				g_cameraConnected = true;
				g_camType = camtyp::typ_kinect;
				//utility::LogInfo("Found {} connected Azure Kinect device(s)\n", k4a_device_get_installed_count());
			}
		}

		//realsense
		device_realsense = context_realsense.query_devices();
		if (device_realsense.size() > 0) {
			g_cameraConnected = true;
			g_camType = camtyp::typ_realsense;
			//utility::LogInfo("Found {} connected Intel Realsense device(s)\n", device_realsense.size());
		}


		//clientCamera
		if (g_clientdata) { //todo client call
			g_camType = camtyp::typ_client;
			g_cameraConnected = true;
		}

		//data cam
		if (g_take_dataCam) {
			g_camType = camtyp::typ_data;
			g_cameraConnected = true;
			//utility::LogInfo("Taking data from harddrive \n");
		}



		if (!g_cameraConnected) { //when there is no camera, there cannot be valid cameraparameters
			g_cameraParameterSet = false;
		}


		if (g_cameraConnected && !g_cameraParameterSet) { //initialise on successful connection parameters one time
			initialiseCamera();
		}

		//what if cameraParameters are not set after program by cameraInitialise?
		while (g_take_dataCam && !g_closeProgram) { // prevent locking if user wants to quit program if datacam is true
			this_thread::sleep_for(20ms);
		}

		if(!g_closeProgram)
			this_thread::sleep_for(1000ms);
	}

}

