#include "match.h"
#include <Open3D/Open3D.h>

std::ostream& operator<<(std::ostream& os, const rmatch& r)
{
	os << "frame index_1: " << r.fi1 << "; frame index_2: "<< r.fi2 << "; index_1: " << r.i1 << "; index_2: " << r.i2 << std::endl;;
	return os;
}

std::ostream& operator<<(std::ostream& os, const match& r)
{
	os << "p1: " << r.p1 << "; p_2: "<< r.p2 << "; distance: " << r.d << "; indeces: " << r.indeces << std::endl;;
	return os;
}

bool c_keypoint::operator==(const c_keypoint& k)
{
	return k.p == this->p;
}


void c_keypoint::transform(Eigen::Matrix4d& m) {
	p = m * p;
}
