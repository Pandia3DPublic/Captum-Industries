#include "match.h"
#include "Open3D/Camera/PinholeCameraIntrinsic.h"
#include <mutex>
#include <Cuda/Camera/PinholeCameraIntrinsicCuda.h>
#include <atomic>
#include "k4a/k4a.h"
#include "k4a/k4a.hpp"
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include "configvars.h"


double g_td = -1;
double g_tc = -1;
double g_tn = -1;
int g_nread = -1;
double g_mergeradius = -1;
double g_conditionThres = -1;
double g_minArea = -1;
double g_reprojection_threshold = -1;
int g_nopt = -1; //number of iterations in optimize
int g_nLocalGroup = -1; //number of max chunks in one local group
int g_nkeypoints = -1;
bool g_segment = false; //untested in checkConfigVariables
double g_cutoff = -1; //cutoff distance for depth data
double g_mincutoff = -1; //cutoff distance for depth data
int g_verbosity = -1;
double g_voxel_length = -1;
double g_treint = -1;

open3d::camera::PinholeCameraIntrinsic g_cameraIntrinsic;
open3d::camera::PinholeCameraIntrinsic g_intrinsic;
open3d::cuda::PinholeCameraIntrinsicCuda g_intrinsic_cuda;
open3d::camera::PinholeCameraIntrinsic g_lowIntr;
//virtual camera intrinsics
open3d::camera::PinholeCameraIntrinsic virtualIntrinsic;
open3d::cuda::PinholeCameraIntrinsicCuda virtualIntrinsic_cuda;
//camera handles
//kinect
k4a::device g_deviceKinect;
k4a::calibration g_calibrationKinect;
//realsense
rs2::pipeline g_pipeRealsense;
rs2::pipeline_profile g_profileRealsense;
std::atomic<bool> g_warmupCamera(false);
int g_warmupProgress = 0;
std::atomic<bool> g_cameraParameterSet(false);

camtyp g_camType = camtyp::typ_kinect;
int nchunk = 11;


//data set stuff
int g_nstart = -1;
string g_readimagePath = "C:/dev/"; //read images from here



//internal global vars
unique_id_counter frame_id_counter;
unique_id_counter chunk_id_counter;
int g_lowx = 80;
int g_lowy = 60;
int g_nOrb = 32;
int g_nFPFH = 33;
std::mutex g_bufferlock;
std::mutex g_protobufflock;
std::mutex g_imagebufflock;
int g_resx = 640;
int g_resy = 480;
bool g_clientdata = false;

int g_initial_width= 1440;
int g_initial_height = 800;

//gui variables
std::atomic<bool> g_pause(false); //indicates that reconthread should pause
std::atomic<bool> g_clear_button(false); //indicates that reconthread should stop
std::atomic<bool> g_closeProgram(false);
//std::atomic<bool> g_PostProcessThreadFinished(false);