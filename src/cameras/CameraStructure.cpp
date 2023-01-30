#include "CameraStructure.h"
#include <mutex>
#include "configvars.h"
#include "utils/coreutil.h"
#include <atomic>

using namespace std;

//static mutex g_bufferlock;
static mutex samplelock;

bool isProcessing = false;

CameraStructure::CameraStructure()
{
}

CameraStructure::~CameraStructure()
{
}

void CameraStructure::align_color_to_depth(const ST::DepthFrame& depth_frame, const ST::ColorFrame& color_frame, shared_ptr<geometry::Image> color, geometry::Image& output)
{
	output.Prepare(depth_frame.width(), depth_frame.height(), 3, 1);

	//get pose mat, convert to Eigen
	const ST::Matrix4 pose_tmp = depth_frame.visibleCameraPoseInDepthCoordinateFrame();
	Eigen::Matrix4d pose = Eigen::Matrix4d::Zero();
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			pose(i, j) = pose_tmp.atRowCol(i, j);
		}
	}
	//cout << "pose \n" << pose << endl;

	// Depth Intrinsics
	Eigen::Matrix3d intr = Eigen::Matrix3d::Zero();
	intr(0, 0) = depth_frame.intrinsics().fx;
	intr(1, 1) = depth_frame.intrinsics().fy;
	intr(0, 2) = depth_frame.intrinsics().cx;
	intr(1, 2) = depth_frame.intrinsics().cy;
	intr(2, 2) = 1;
	//cout << "intr depth\n" << intr << endl;

	// Color Intrinsics
	Eigen::Matrix3d intr_col = Eigen::Matrix3d::Zero();
	intr_col(0, 0) = color_frame.intrinsics().fx;
	intr_col(1, 1) = color_frame.intrinsics().fy;
	intr_col(0, 2) = color_frame.intrinsics().cx;
	intr_col(1, 2) = color_frame.intrinsics().cy;
	intr_col(2, 2) = 1;
	//cout << "intr col\n" << intr_col << endl;

	float* d_raw = (float*)depth_frame.depthInMillimeters(); //pointer to raw depth values
	for (int y = 0; y < depth_frame.height(); y++)
	{
		for (int x = 0; x < depth_frame.width(); x++, d_raw++)
		{
			if (!isnan(*d_raw)) {
				double d = *d_raw * 0.001; //depth in meters
				//cout << "d in meters: " << d << endl;

				// Reproject (x,y,Z) to (X,Y,Z,1) in depth camera frame
				Eigen::Vector4d vdepth;
				vdepth[0] = (x - intr(0, 2)) * d / intr(0, 0);
				vdepth[1] = (y - intr(1, 2)) * d / intr(1, 1);
				vdepth[2] = d;
				vdepth[3] = 1;

				// Transform to RGB camera frame
				Eigen::Vector4d vrgb = pose * vdepth;

				// project transformed points to rgb image coords
				int x_rgb = vrgb(0) * intr_col(0, 0) / vrgb(2) + intr_col(0, 2) + 0.5; //value is often -3258 (invalid) -> if d_raw is 0
				int y_rgb = vrgb(1) * intr_col(1, 1) / vrgb(2) + intr_col(1, 2) + 0.5;


				//// depth registration
				//if (x_rgb >= 0 && x_rgb <= color_frame.width() && y_rgb >= 0 && y_rgb <= color_frame.height()) { //check if in range of output image
				//	uint16_t reg_depth = *output.PointerAt<uint16_t>(x_rgb, y_rgb);
				//	uint16_t new_depth = vrgb(2) * 1000.0;
				//	// write depth values to output image
				//	if (reg_depth == 0 || reg_depth > new_depth) {
				//		*output.PointerAt<uint16_t>(x_rgb, y_rgb) = new_depth;
				//	}
				//}

				//col registration
				if (x_rgb >= 0 && x_rgb < color->width_ && y_rgb >= 0 && y_rgb < color->height_) { //we hit the picture
					*output.PointerAt<uint8_t>(x, y, 0) = *color->PointerAt<uint8_t>(x_rgb, y_rgb, 0); //todo bilinear interpolation
					*output.PointerAt<uint8_t>(x, y, 1) = *color->PointerAt<uint8_t>(x_rgb, y_rgb, 1);
					*output.PointerAt<uint8_t>(x, y, 2) = *color->PointerAt<uint8_t>(x_rgb, y_rgb, 2);
				}
			}

		}
	}
}

void CameraStructure::processFrames(shared_ptr<ST::CaptureSessionSample>& sample, list <shared_ptr<Frame>>& framebuffer)
{
	geometry::Image color_image_8bit;
	geometry::Image depth_image_16bit;
	geometry::Image color_image_aligned;
	std::shared_ptr<geometry::Image> color_image_pointer;
	std::shared_ptr<geometry::Image> depth_image_pointer;

	// Convert frame data to open3d image
	// Color
	color_image_8bit.Prepare(sample->visibleFrame.width(), sample->visibleFrame.height(), 3, 1);
	uint8_t* pixel = (uint8_t*)sample->visibleFrame.rgbData();
#pragma omp parallel for
	for (int y = 0; y < sample->visibleFrame.height(); y++) {
		for (int x = 0; x < sample->visibleFrame.width(); x++) {
			*color_image_8bit.PointerAt<uint8_t>(x, y, 0) = *pixel++;
			*color_image_8bit.PointerAt<uint8_t>(x, y, 1) = *pixel++;
			*color_image_8bit.PointerAt<uint8_t>(x, y, 2) = *pixel++;
		}
	}

	// Depth
	depth_image_16bit.Prepare(sample->depthFrame.width(), sample->depthFrame.height(), 1, 2);
	float* pixel_d = (float*)sample->depthFrame.depthInMillimeters();
#pragma omp parallel for
	for (int y = 0; y < sample->depthFrame.height(); y++) {
		for (int x = 0; x < sample->depthFrame.width(); x++) {
			*depth_image_16bit.PointerAt<uint16_t>(x, y) = (uint16_t)* pixel_d++;
		}
	}

	//Alignment
	align_color_to_depth(sample->depthFrame, sample->visibleFrame, make_shared<geometry::Image>(color_image_8bit), color_image_aligned);

	//Assign aligned color image and depth image to pointer
	color_image_pointer = make_shared<geometry::Image>(color_image_aligned);
	depth_image_pointer = make_shared<geometry::Image>(depth_image_16bit);

	//Create cvColor Mat for keypoint detection
	cv::Mat col;
	topencv(color_image_pointer, col);
	auto tmp = make_shared<Frame>();

	//get point cloud frames from image data
	auto rgbd = geometry::RGBDImage::CreateFromColorAndDepth(color_image_aligned, depth_image_16bit, 1000.0, 3.0, false);
	auto pcd = geometry::PointCloud::CreateFromRGBDImage(*rgbd, g_intrinsic);

	//tmp->rgbd = rgbd;
	//NOTE: THIS IS NOT UP TO DATE. USE GENERATEFRAMES FOR A TIMELY IMPLEMENTATION
	///tmp->depth = depth_image_pointer; //todo remove
	tmp->chunktransform = getIdentity();

	//get smaller images for costly calculations 
	tmp->rgblow = resizeImage(color_image_pointer, g_lowx, g_lowy, "nointerpol"); //todo do bilinear interpolate
	tmp->depthlow = resizeImage(depth_image_pointer, g_lowx, g_lowy, "nointerpol");
	auto rgbdlow = geometry::RGBDImage::CreateFromColorAndDepth(*tmp->rgblow, *tmp->depthlow, 1000.0, 3.0, false);
	tmp->lowpcd = geometry::PointCloud::CreateFromRGBDImage(*rgbdlow, g_lowIntr);
	if (tmp->lowpcd->HasNormals() == false) {
		tmp->lowpcd->EstimateNormals();
	}
	tmp->lowpcd->NormalizeNormals();
	tmp->lowpcd->OrientNormalsTowardsCameraLocation();

	g_bufferlock.lock();
	framebuffer.push_back(tmp);
	g_bufferlock.unlock();

}

void CameraStructure::SessionDelegate::captureSessionEventDidOccur(ST::CaptureSession* session, ST::CaptureSessionEventId event) {
	printf("Received capture session event %d (%s)\n", (int)event, ST::CaptureSessionSample::toString(event));
	switch (event) {
	case ST::CaptureSessionEventId::Booting: break;
	case ST::CaptureSessionEventId::Ready:
		printf("Starting streams...\n");
		printf("Sensor Serial Number is %s \n ", session->sensorSerialNumber());
		session->startStreaming();
		break;
	case ST::CaptureSessionEventId::Disconnected:
	case ST::CaptureSessionEventId::Error:
		printf("Capture session error\n");
		exit(1);
		break;
	default:
		printf("Capture session event unhandled\n");
	}
}

void CameraStructure::SessionDelegate::captureSessionDidOutputSample(ST::CaptureSession*, const ST::CaptureSessionSample& sample) {
	//printf("Received capture session sample of type %d (%s)\n", (int)sample.type, ST::CaptureSessionSample::toString(sample.type));
	switch (sample.type) {
	case ST::CaptureSessionSample::Type::DepthFrame:
		printf("Depth frame: size %dx%d\n", sample.depthFrame.width(), sample.depthFrame.height());
		break;
	case ST::CaptureSessionSample::Type::VisibleFrame:
		printf("Visible frame: size %dx%d\n", sample.visibleFrame.width(), sample.visibleFrame.height());
		break;
	case ST::CaptureSessionSample::Type::InfraredFrame:
		printf("Infrared frame: size %dx%d\n", sample.infraredFrame.width(), sample.infraredFrame.height());
		break;
	case ST::CaptureSessionSample::Type::SynchronizedFrames: // We only want synchronized depth and color frames
		//printf("Synchronized frames: depth %dx%d visible %dx%d infrared %dx%d\n", sample.depthFrame.width(), sample.depthFrame.height(), sample.visibleFrame.width(), sample.visibleFrame.height(), sample.infraredFrame.width(), sample.infraredFrame.height());
		samplelock.lock();
		samplebuffer.push_back(make_shared<ST::CaptureSessionSample>(sample));
		samplelock.unlock();
		break;
	case ST::CaptureSessionSample::Type::AccelerometerEvent:
		printf("Accelerometer event: [% .5f % .5f % .5f]\n", sample.accelerometerEvent.acceleration().x, sample.accelerometerEvent.acceleration().y, sample.accelerometerEvent.acceleration().z);
		break;
	case ST::CaptureSessionSample::Type::GyroscopeEvent:
		printf("Gyroscope event: [% .5f % .5f % .5f]\n", sample.gyroscopeEvent.rotationRate().x, sample.gyroscopeEvent.rotationRate().y, sample.gyroscopeEvent.rotationRate().z);
		break;
	default:
		printf("Sample type unhandled\n");
	}
}


void CameraStructure::startCameraThread(list <shared_ptr<Frame>>& framebuffer, std::atomic<bool>& stop, std::atomic<bool>& cameraParameterSet)
{
	ST::CaptureSessionSettings settings;
	settings.source = ST::CaptureSessionSourceId::StructureCore;
	settings.structureCore.depthEnabled = true;
	settings.structureCore.visibleEnabled = true;
	settings.structureCore.depthResolution = ST::StructureCoreDepthResolution::VGA;
	settings.structureCore.visibleResolution = ST::StructureCoreVisibleResolution::VGA;
	utility::LogWarning("Variable Resolution not supported for Structure Sensor at the moment. Resolution is set to 640x480.\n");
	settings.structureCore.depthFramerate = 30;
	settings.structureCore.visibleFramerate = 30;
	settings.structureCore.infraredAutoExposureEnabled = true;
	settings.structureCore.initialVisibleExposure = 0.016f; //todo nice, code own auto exposure feature
	settings.structureCore.initialVisibleGain = 2.0f;
	settings.applyExpensiveCorrection = true;

	SessionDelegate delegate; //delegate receives frames from session, we need a delegate to access frames from session
	ST::CaptureSession session;
	session.setDelegate(&delegate);

	if (!session.startMonitoring(settings)) { //start session with specified settings
		utility::LogError("Failed to initialize capture session!\n");
		exit(1);
	}

	//get intrinsics
	while (!cameraParameterSet) {
		while (delegate.samplebuffer.empty()) { //wait for first sample
			std::this_thread::sleep_for(20ms);
		}
		if (delegate.samplebuffer.front()->depthFrame.isValid()) {
			samplelock.lock();
			ST::Intrinsics intr = delegate.samplebuffer.front()->depthFrame.intrinsics();
			samplelock.unlock();
			camera::PinholeCameraIntrinsic intrinsic = camera::PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy);
			g_intrinsic = intrinsic;
			g_intrinsic_cuda = open3d::cuda::PinholeCameraIntrinsicCuda(g_intrinsic);
			g_lowIntr = getLowIntr(intrinsic);
			cameraParameterSet = true;
		}
		else {
			utility::LogWarning("Failed to get Intrinsics! Depth Frame is not valid.\n");
			samplelock.lock();
			delegate.samplebuffer.pop_front();
			samplelock.unlock();
		}
	}

	// The SessionDelegate receives samples on a background thread while streaming.
	while (!stop) {
		if (!delegate.samplebuffer.empty()) {
			samplelock.lock();
			auto tmpSample = delegate.samplebuffer.front();
			delegate.samplebuffer.pop_front();
			samplelock.unlock();

			processFrames(tmpSample, framebuffer); //process frames, push to buffer

		}
		else {
			std::this_thread::sleep_for(20ms);
		}
	}
	utility::LogInfo("Stop streaming...\n");
	session.stopStreaming();
}

std::shared_ptr<Frame> CameraStructure::getSingleFrame(list<shared_ptr<Frame>>& framebuffer)
{
	while (framebuffer.empty()) {
		std::this_thread::sleep_for(20ms);
	}

	g_bufferlock.lock();
	auto tmp = std::make_shared<Frame>();
	tmp = framebuffer.front();
	framebuffer.pop_front();
	g_bufferlock.unlock();

	return tmp;
}
