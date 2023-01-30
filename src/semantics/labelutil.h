#pragma once
#include <Open3D/Open3D.h>
#include "LabeledClasses.h"
#include "Python/open3d_pybind.h"
#include "pybind11/embed.h"

//Struct for calculating the intersection over union
struct IOU {
	int true_positive = 0;
	int false_positive = 0;
	int false_negative = 0;
	double value = 0;
	uint8_t label;
};



//mesh should be ground truth
double determineAccuracy(LabeledPointCloud& lpcd, LabeledTriangleMesh& lmesh);
//returns accurarcy
double determineAccuracy(LabeledTriangleMesh& lmeshGround, LabeledTriangleMesh& lmeshpred);
//unknown must be label 255 for this mehtod to work
double determineunkown(LabeledTriangleMesh& lmesh);
//utility function to calculate ious on multiple meshes with ground truth data.
void determineIOU(LabeledTriangleMesh& lmeshGround, LabeledTriangleMesh& lmeshpred, std::vector<IOU>& IOUs);



//creates a label to color map. This map can be used by pcd and mesh functions.
std::pair< std::vector<uint8_t>, std::vector<Eigen::Vector3d>> createColorMap(int numofValues, bool range);
//write colormap to file to plot it in python
void saveColorMap(const std::pair< std::vector<uint8_t>, std::vector<Eigen::Vector3d>>& colormap, const std::vector<std::string>& names, const std::string& filename);
//color an image with labels. takes one channel one byte img. returns 3 channel img
std::shared_ptr<geometry::Image> colorImageLabels(std::shared_ptr<geometry::Image> image_ptr, std::pair< std::vector<uint8_t>, std::vector<Eigen::Vector3d>> colormap);

//core function to retrieve a labeled image given an rgb image (3 channel) and a neural net model and a module from which to call it from
//return size is always 513 (current deeplab v3+ model) for some weird reason todo
std::shared_ptr<geometry::Image> getSegmentedImagePython(geometry::Image& img_ptr, py::module segment, py::object model);

//knn filter for labeled images (one channel grayscale)
std::shared_ptr<geometry::Image> majorityVote(std::shared_ptr<geometry::Image> image_ptr, int kernel_size);
//returns the most frequent element in an uint8_t vector as int
int findmaxfreqelem(std::vector<uint8_t>& vec);
//generate a labeled mesh from a labeled pcd and a fitting mesh. defaultlable determines the lable in case no lable is found. lmesh should not be labeled but normal triangle mesh
std::shared_ptr<LabeledTriangleMesh> getLabeledMesh(LabeledPointCloud& lpcd, LabeledTriangleMesh& lmesh, uint8_t defaultlabel = -1);

void RGBtoHSV(float& fR, float& fG, float fB, float& fH, float& fS, float& fV);
/*! \brief Convert HSV to RGB color space

  Converts a given set of HSV values `h', `s', `v' into RGB
  coordinates. The output RGB values are in the range [0, 1], and
  the input HSV values are in the ranges h = [0, 360], and s, v =
  [0, 1], respectively.

  \param fR Red component, used as output, range: [0, 1]
  \param fG Green component, used as output, range: [0, 1]
  \param fB Blue component, used as output, range: [0, 1]
  \param fH Hue component, used as input, range: [0, 360]
  \param fS Hue component, used as input, range: [0, 1]
  \param fV Hue component, used as input, range: [0, 1]

*/
void HSVtoRGB(float& fR, float& fG, float& fB, float& fH, float& fS, float& fV);


