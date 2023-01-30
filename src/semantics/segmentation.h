#pragma once
#include <list>
#include "core/Frame.h"
#include "labelutil.h"
#include "../core/threadvars.h"

void segmentationThread(list <shared_ptr<Frame>>* segframebuffer, std::atomic<bool>* stop, std::atomic<bool>* loaded);