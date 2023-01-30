#pragma once
#include "GlobalDefines.h"
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING //need this since VS 2019 16.3 throws error for deprecated <experimental/filesystem> header.
	//this define has no effect for other platforms or toolset versions. in the long term, switch to C++17 <filesystem> header.
#include <experimental/filesystem>
#define NOMINMAX //need this for opencv and pybind to avoid define clashes

//#include "cameras/ClientCamera/clientCamera.h"

#include <iostream>
#include <memory>
#include <thread>
#include <list>
#include <atomic>
#include "core/threadvars.h" //contains the necesssary mutex locks. must be included before integrate.h (todo)
#include "utils/visutil.h"
#include "utils/coreutil.h"
#include "filters/kabschfilter.h"
#include "filters/reprojection.h"
#include "core/Frame.h"
#include "core/Chunk.h"
#include "core/Model.h"
#include "cmakedefines.h"
#include "genKps.h"
#include "genCors.h"
#include "readconfig.h"
#include "configvars.h"
#include "integrate.h"
#include <Cuda/Open3DCuda.h>
#include "cameras/CameraThreadandInit.h"
#include "GPUSolver/solverWrapper.h"
#include "cameras/DataCam.h"
#include "Gui/guiutil.h"
#include "semantics/segmentation.h"

//todo speed up compile time

int reconrun(Model &m, bool livevis, bool integrate);
