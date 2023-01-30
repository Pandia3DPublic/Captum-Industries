#pragma once
#include "match.h"
#include "Open3D/Camera/PinholeCameraIntrinsic.h"
#include <Cuda/Camera/PinholeCameraIntrinsicCuda.h>
#include <mutex>
#include <atomic>
#include <k4a/k4a.h>
#include <k4a/k4a.hpp>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API


extern double g_td;
extern double g_tc;
extern double g_tn;
extern int g_nread;
extern double g_mergeradius;
extern double g_conditionThres;
extern double g_minArea;
extern double g_reprojection_threshold;
extern int g_nkeypoints;
extern int g_nLocalGroup;
extern int g_nopt;
extern open3d::camera::PinholeCameraIntrinsic g_cameraIntrinsic;
extern open3d::camera::PinholeCameraIntrinsic g_intrinsic;
extern open3d::cuda::PinholeCameraIntrinsicCuda g_intrinsic_cuda;
extern open3d::camera::PinholeCameraIntrinsic g_lowIntr;
//virtual camera intrinsics
extern open3d::camera::PinholeCameraIntrinsic virtualIntrinsic;
extern open3d::cuda::PinholeCameraIntrinsicCuda virtualIntrinsic_cuda;

extern bool g_segment;
extern double g_cutoff;
extern double g_mincutoff;
extern int g_verbosity;
extern double g_voxel_length;
extern double g_treint;
//camera handles
	//kinect
extern k4a::device g_deviceKinect;
extern k4a::calibration g_calibrationKinect;
	//realsense
extern rs2::pipeline g_pipeRealsense;
extern rs2::pipeline_profile g_profileRealsense;
	//other
extern std::atomic<bool> g_cameraParameterSet;

//extern enum class camtyp;
enum class camtyp {typ_kinect, typ_realsense, typ_data, typ_client};


extern camtyp g_camType;
extern std::atomic<bool> g_warmupCamera;
extern int g_warmupProgress;

//data set stuff
extern int g_nstart;
extern string g_readimagePath;





extern unique_id_counter frame_id_counter;
extern unique_id_counter chunk_id_counter;
extern int g_lowx;
extern int g_lowy;
extern int g_nOrb; //number of bytes in a descriptor of orb
extern int g_nFPFH; 
extern std::mutex g_bufferlock;
extern std::mutex g_protobufflock;
extern std::mutex g_imagebufflock;
extern int g_resx;
extern int g_resy;
extern bool g_clientdata;

extern int g_initial_width;
extern int g_initial_height;

extern std::atomic<bool> g_pause;
extern std::atomic<bool> g_clear_button;
extern std::atomic<bool> g_closeProgram;

extern int nchunk;
//extern std::atomic<bool> g_PostProcessThreadFinished;