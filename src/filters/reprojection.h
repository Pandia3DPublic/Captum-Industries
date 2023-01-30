#include "match.h"
#include "core/Frame.h"
//#include "configvars.h"
using namespace std;

void reprojectionfilter(shared_ptr<Frame> f1, shared_ptr<Frame> f2,  pairTransform& trans, std::vector<match>& matches);
