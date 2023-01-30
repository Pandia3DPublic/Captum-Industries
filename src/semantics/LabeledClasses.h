#define _USE_MATH_DEFINES
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING

#pragma once
#include "Open3D/Open3D.h"
 
#include <Open3D/Integration/MarchingCubesConst.h>
#include <chrono>

//this file contains inherited classes for mesh, pcd and tsdf that contain lables.
using namespace open3d;

class LabeledPointCloud : public geometry::PointCloud {
public:
	std::vector<uint8_t> labels_;
	std::vector<Eigen::Vector3d> original_colors;
	std::vector<float> confidences_;
	

	// labelimg has to be uint8_t values
	// depthimg has to be float values
	// color has to be float values
	static std::shared_ptr<LabeledPointCloud> CreateFromRGBDImageAndLabel(
		const geometry::Image& colorimage,
		const geometry::Image& depthimage,
		const geometry::Image& labelImg,
		const camera::PinholeCameraIntrinsic& intrinsic,
		const Eigen::Matrix4d& extrinsic);
	//removes points in classes based on variancen. Inferior to knn-filter. Not real-time capable.
	void RemoveStatisticalOutliers(size_t nb_neighbors, double std_ratio);
	//knn Filter. Uses a weighted majority vote in a neighbourhood to improve accuracy. Almost always works but is not rt. 20 is usual paramter
	void LabeledPointCloud::NearestNeighborFilter(int nb_neighbors);
	//first parameter in pair is labels. Second parameter is colors. Generates a colormap automatically and subsequently return the used map. 
	//Checks how many different labels are present and colors the pcd.
	std::pair< std::vector<uint8_t>, std::vector<Eigen::Vector3d>> colorLabels();
	//uses a colormap to color the pcd.
	void colorLabels(std::pair< std::vector<uint8_t>, std::vector<Eigen::Vector3d>> colormap);
	//rewrites the original rgb data
	void showOriginalColors();
};

class LabeledTriangleMesh : public geometry::TriangleMesh {

public:
	std::vector<Eigen::Vector3d> vertex_colors_org;
	std::vector<uint8_t> vertex_labels;
	std::vector<float> vertex_labels_confidence;
	// uses the labels to color the mesh
	void ColorLabels(std::pair< std::vector<uint8_t>, std::vector<Eigen::Vector3d>> colormap);
	//return to original rgb data
	void showOriginalColors();
	//map one label set to another given a conversion table.
	void changeLabels(std::vector < std::pair<uint8_t, uint8_t>> convertTable);
};

class LabeledScalTSDFVolume : public integration::ScalableTSDFVolume {

public:
	//probably necessary for default paras. Is standard tsdfvolume constructor
	LabeledScalTSDFVolume(double voxel_length,
							double sdf_trunc,
							integration::TSDFVolumeColorType color_type,
							int volume_unit_resolution = 16,
							int depth_sampling_stride = 4);

	std::shared_ptr<LabeledPointCloud> ExtractPointCloudwithLabels();
	std::shared_ptr<LabeledTriangleMesh> ExtractLabeledTriangleMesh();

	//colorimg has to be 3 channel uint8_t´or 1 channel float 
	// depthimg has to be float
	// segmap has to be uint8_t
	//uses a custom version of IntegrateWithDepthToCameraDistanceMultiplier in uniform tsdf volume that takes a segmap
	void IntegratewithLabels(
		const geometry::Image& colorimg,
		const geometry::Image& depthimg,
		const geometry::Image& segmap,
		const camera::PinholeCameraIntrinsic& intrinsic,
		const Eigen::Matrix4d& extrinsic);

		// depthimg has to be float
		// segmap has to be uint8_t
	void IntegrateonlyLabels(
		const geometry::Image& depthimg,
		const geometry::Image& segmap,
		const camera::PinholeCameraIntrinsic& intrinsic,
		const Eigen::Matrix4d& extrinsic);

	//performs usual integration while counting the number of lables per voxel. If the average is lower than ratio the method returns true, otherwise false
	//should be used prior to integrateonlylables
	bool IntegratewhileCounting(const geometry::Image& colorimg,
		const geometry::Image& depthimg,
		const camera::PinholeCameraIntrinsic& intrinsic,
		const Eigen::Matrix4d& extrinsic,
		double ratio);

};













