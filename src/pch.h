#pragma once
#include "GlobalDefines.h"

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
