#pragma once
#include "match.h"
#include "core/KeypointUnit.h"
#include  <vector>
#include "opencv2/features2d/features2d.hpp"


bool matchsort(match& i, match& j);
void getCors(std::shared_ptr<KeypointUnit> f1,std::shared_ptr<KeypointUnit> f2, std::vector<match> &out);