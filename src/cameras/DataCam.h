#pragma once
#include <list>
#include "core/Frame.h"
#include <atomic>

void DataCamThreadFunction(list <shared_ptr<Frame>>& framebuffer, std::atomic<bool>& stop);