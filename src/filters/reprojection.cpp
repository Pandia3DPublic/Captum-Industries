#include "reprojection.h"
#include "configvars.h"

//todo check if camera transformation is vernachlaessigt
//todo use intensity pics for color (maybe)
void reprojectionfilter(shared_ptr<Frame> f1, shared_ptr<Frame> f2, pairTransform& trans, std::vector<match>& matches) {
	double err = 0;
	int count = 0;
	geometry::PointCloud tmp = *f1->lowpcd;
	tmp.Transform(trans.kabschtrans);
	auto lowintr = g_lowIntr.intrinsic_matrix_; 
	for (int i = 0; i < tmp.points_.size(); i++) {
		auto& p = tmp.points_[i];
		//project transformed points to image coords
		int x = p(0) * lowintr(0, 0) / p(2) + lowintr(0, 2);
		int y = p(1) * lowintr(1, 1) / p(2) + lowintr(1, 2);
		if (x < 80 && y < 60 && x>-1 && y>-1) { // todo make more general and solve via intrinsic
			if (f2->lowpcd->indeces(x, y) != -1) {
				//find the corresponding point
				auto a = f2->lowpcd->indeces(x, y);
				auto p2 = f2->lowpcd->points_[a];
				//do tests if the point is valid
				auto diff = (p2 - p).norm();
				if (diff < g_td) { // depth
					if ((tmp.colors_[i] - f2->lowpcd->colors_[f2->lowpcd->indeces(x, y)]).sum() < g_tc) { // color
						if (tmp.normals_[i].transpose() * f2->lowpcd->normals_[f2->lowpcd->indeces(x, y)] > g_tn) { // normals
							//point correspondence is valid 
							err += diff;
							count++;
						}
					}
				}
			}
		}
	}
	if (count > 0)
		err /= count;

	if (err > g_reprojection_threshold || count < f2->depthlow->width_* f2->depthlow->height_* 0.02){
		matches = vector<match>();
		trans.set = false;
		utility::LogDebug("All matches removed by reprojection filter. Error was: {} \n", err);
	}
}

//p2(0) = (x - lowintr(0, 2)) * d / lowintr(0, 0);
//p2(1) = (y - lowintr(1, 2)) * d / lowintr(1, 1);
//p2(2) = d;
////do tests if the point is valid
//if ((p2 - p).norm() < 0.15) { // depth
//	auto color = Eigen::Vector3d();
//	color(0) = *f2->rgblow->PointerAt<uint>(x, y, 0);
//	color(1) = *f2->rgblow->PointerAt<uint>(x, y, 1);
//	color(2) = *f2->rgblow->PointerAt<uint>(x, y, 2);
//	if ((tmp.colors_[i] - color).norm() < 0.1) { //color todo 1-norm
//	}
//}