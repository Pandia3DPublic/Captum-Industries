#include "LabeledClasses.h"
#include "labelutil.h"


using namespace open3d;
using namespace geometry;
using namespace integration;
using namespace std::chrono;

//############################################################ headerless helper functions #############################################
int CountValidDepthPixels(const Image& depth, int stride) {
	int num_valid_pixels = 0;
	for (int i = 0; i < depth.height_; i += stride) {
		for (int j = 0; j < depth.width_; j += stride) {
			const float* p = depth.PointerAt<float>(j, i);
			if (*p > 0) num_valid_pixels += 1;
		}
	}
	return num_valid_pixels;
}

// find weighted most common element
int findmaxfreqelement(std::vector<uint8_t>& vec, std::vector<float>& w) {
	std::vector<float> histogramcount;
	std::vector<uint8_t> histogramvalues;
	int maxvalue = 0;
	int maxclass = -1;


	for (int i = 0; i < vec.size(); i++) {
		bool found = false;
		auto& element = vec[i];
		for (int a = 0; a < histogramvalues.size(); a++) {
			if (histogramvalues[a] == element) {
				histogramcount[a] += w[i];
				if (maxvalue < histogramcount[a]) {
					maxvalue = histogramcount[a];
					maxclass = histogramvalues[a];
				}
				found = true;
				break;
			}
		}
		if (!found) {
			histogramvalues.push_back(element);
			histogramcount.push_back(w[i]);
		}
	}

	return maxclass;
}


//############################################################ header functions #############################################

void LabeledTriangleMesh::changeLabels(std::vector < std::pair<uint8_t, uint8_t>> convertTable) {
#pragma omp parallel for
	for (int i = 0; i < vertex_labels.size(); i++) {
		bool found = false;
		for (auto pair : convertTable) {
			auto from = pair.first;
			if (vertex_labels[i] == from) {
				vertex_labels[i] = pair.second;
				found = true;
				break;
			}
		}
		if (!found) {
			vertex_labels[i] = 39;
		}
	}
}



std::pair< std::vector<uint8_t>, std::vector<Eigen::Vector3d>> LabeledPointCloud::colorLabels() {
	std::vector<uint8_t> histogramvalues;
	int num_unique_elements = 0;
	original_colors = colors_;
	for (auto x : labels_) {
		bool found = false;
		for (int a = 0; a < histogramvalues.size(); a++) {
			if (histogramvalues[a] == x) {
				found = true;
				break;
			}
		}
		if (!found) {
			num_unique_elements++;
			histogramvalues.push_back(x);
		}
	}


	std::vector<Eigen::Vector3d> colorvalues;
	for (int i = 0; i < num_unique_elements; i++) {
		/* double color1 = i * 256 / num_unique_elements;
		 double color2 = 255.0 - (i * 2 * 256 / num_unique_elements) % 256;
		 double color3 = 255.0 - (i * 3 * 256 / num_unique_elements) % 256;
		 Eigen::Vector3d colorvec(color1 / 255.0, color2 / 255.0, color3 / 255.0);*/
		float hue = 360.0 * i / num_unique_elements;
		float saturation = 0.5 + (i % 2) / 2.0;
		float value = 0.5 + (i % 2) / 2.0;
		float r, g, b;
		HSVtoRGB(r, g, b, hue, saturation, value);
		Eigen::Vector3d colorvec(r, g, b);
		// std::cout << "rgb: " << colorvec << '\n';
		colorvalues.push_back(colorvec);
	}



#pragma omp parallel for
	for (int i = 0; i < colors_.size(); i++) {
		for (int a = 0; a < histogramvalues.size(); a++) {
			if (labels_[i] == histogramvalues[a]) {
				colors_[i] = colorvalues[a];
				break;
			}
		}
	}

	return std::pair< std::vector<uint8_t>, std::vector<Eigen::Vector3d>>(histogramvalues, colorvalues);

}

void LabeledPointCloud::colorLabels(std::pair< std::vector<uint8_t>, std::vector<Eigen::Vector3d>> colormap) {
	auto histogramvalues = colormap.first;
	auto colorvalues = colormap.second;
	original_colors = colors_;
#pragma omp parallel for
	for (int i = 0; i < colors_.size(); i++) {
		for (int a = 0; a < histogramvalues.size(); a++) {
			if (labels_[i] == histogramvalues[a]) {
				colors_[i] = colorvalues[a];
				break;
			}
		}
	}

}

void LabeledPointCloud::showOriginalColors() {

	if (!original_colors.empty())
		colors_ = original_colors;
	else  std::cout << "already showing true colors" << std::endl;

}

void LabeledTriangleMesh::showOriginalColors() {

	if (!vertex_colors_org.empty())
		vertex_colors_ = vertex_colors_org;
	else  std::cout << "already showing true colors" << std::endl;

}

std::shared_ptr<LabeledPointCloud> LabeledPointCloud::CreateFromRGBDImageAndLabel(
	const Image& colorimage,
	const Image& depthimage,
	const Image& labelImg,
	const camera::PinholeCameraIntrinsic& intrinsic,
	const Eigen::Matrix4d& extrinsic) {



	auto pointcloud = std::make_shared<LabeledPointCloud>();
	Eigen::Matrix4d camera_pose = extrinsic.inverse();
	auto focal_length = intrinsic.GetFocalLength();
	auto principal_point = intrinsic.GetPrincipalPoint();
	int num_valid_pixels = CountValidDepthPixels(depthimage, 1);
	pointcloud->points_.resize(num_valid_pixels);
	pointcloud->colors_.resize(num_valid_pixels);
	pointcloud->labels_.resize(num_valid_pixels);
	int cnt = 0;


	for (int i = 0; i < depthimage.height_; i++) {
		for (int j = 0; j < depthimage.width_; j++) {
			auto depthvalue = *depthimage.PointerAt<float>(j, i);
			Eigen::Vector3d colorvec;
			auto labelvalue = *labelImg.PointerAt<uint8_t>(j, i);
			for (int nc = 0; nc < colorimage.num_of_channels_; nc++) {
				colorvec(nc) = *colorimage.PointerAt<float>(j, i, nc);
			}
			if (depthvalue > 0) {
				double z = (double)(depthvalue);
				double x = (j - principal_point.first) * z / focal_length.first;
				double y =
					(i - principal_point.second) * z / focal_length.second;
				Eigen::Vector4d point =
					camera_pose * Eigen::Vector4d(x, y, z, 1.0);
				pointcloud->points_[cnt] = point.block<3, 1>(0, 0);
				pointcloud->labels_[cnt] = labelvalue;
				pointcloud->colors_[cnt] = colorvec;
				cnt++;
			}
		}

	}

	return pointcloud;

}

void LabeledPointCloud::NearestNeighborFilter(int nb_neighbors) {


	//takes about 70ms on 1mio points
	KDTreeFlann kdtree;
	kdtree.SetGeometry(*this);
	auto pointcloud = std::make_shared<LabeledPointCloud>();
#pragma omp parallel for 
	for (auto i = 0; i < points_.size(); i++) {
		// std::cout << i << std::endl;
		// auto start = high_resolution_clock::now();
		std::vector<int> tmp_indices;
		std::vector<double> dist;
		// cost with 20 nn about 4.4 us
		kdtree.SearchKNN(points_[i], nb_neighbors, tmp_indices, dist);
		//auto stop = high_resolution_clock::now();
		//auto duration = duration_cast<nanoseconds>(stop - start);
		//std::cout << "time for kdtree search  in us: " << duration.count() << std::endl;


		//start = high_resolution_clock::now();

		// cost with 20 nn about 2.2 us
		std::vector<uint8_t> neighbours;
		std::vector<float> confidence_neighbour;
		neighbours.push_back(labels_[i]);
		confidence_neighbour.push_back(confidences_[i]);
		for (int a = 0; a < tmp_indices.size(); a++) {
			neighbours.push_back(labels_[tmp_indices[a]]);
			confidence_neighbour.push_back(confidences_[tmp_indices[a]]);
		}

		auto oldlabel = labels_[i];
		labels_[i] = findmaxfreqelement(neighbours, confidence_neighbour);
		if (oldlabel != labels_[i]) confidences_[i] = 0.1;

		/* stop = high_resolution_clock::now();
		 duration = duration_cast<nanoseconds>(stop - start);
		 std::cout << "time for histogramm building  in us: " << duration.count() << std::endl;*/
	}

}

void LabeledPointCloud::RemoveStatisticalOutliers(size_t nb_neighbors, double std_ratio) {

	std::vector<uint8_t> histogramvalues;
	for (auto x : labels_) {
		bool found = false;
		for (int a = 0; a < histogramvalues.size(); a++) {
			if (histogramvalues[a] == x) {
				found = true;
				break;
			}
		}
		if (!found) {
			histogramvalues.push_back(x);
		}
	}


	std::vector< std::shared_ptr<geometry::PointCloud>> monolabelpcd;
	std::vector<std::vector<size_t>> monolabelpcd_indices;
	for (auto label : histogramvalues) {
		geometry::PointCloud pcd;
		std::vector<size_t> indices;
		for (int i = 0; i < points_.size(); i++) {
			if (label == labels_[i]) {
				indices.push_back(i);
			}
		}
		if (indices.size() > 20) {
			monolabelpcd_indices.push_back(indices);
			monolabelpcd.push_back(SelectDownSample(indices));
		}
	}

	for (auto& label : labels_) {
		label = -1;
	}

	for (int i = 0; i < monolabelpcd.size(); i++) {
		auto tuple = (*monolabelpcd[i]).RemoveStatisticalOutliers(nb_neighbors, std_ratio);
		std::vector<size_t>& pcd = std::get<1>(tuple);

		for (int a = 0; a < pcd.size(); a++) {
			labels_[monolabelpcd_indices[i][pcd[a]]] = histogramvalues[i];
		}
	}


}

LabeledScalTSDFVolume::LabeledScalTSDFVolume(double voxel_length,
	double sdf_trunc,
	integration::TSDFVolumeColorType color_type,
	int volume_unit_resolution,
	int depth_sampling_stride) : ScalableTSDFVolume(voxel_length,
		sdf_trunc,
		color_type,
		volume_unit_resolution,
		depth_sampling_stride)
{}


std::shared_ptr<LabeledPointCloud> LabeledScalTSDFVolume::ExtractPointCloudwithLabels() {
	auto pointcloud = std::make_shared<LabeledPointCloud>();
	double half_voxel_length = voxel_length_ * 0.5;
	float w0, w1, f0, f1;
	std::vector<uint8_t> l1, l0;
	Eigen::Vector3f c0, c1;
	for (const auto& unit : volume_units_) {
		if (unit.second.volume_) {
			const auto& volume0 = *unit.second.volume_;
			const auto& index0 = unit.second.index_;
			for (int x = 0; x < volume0.resolution_; x++) {
				for (int y = 0; y < volume0.resolution_; y++) {
					for (int z = 0; z < volume0.resolution_; z++) {
						Eigen::Vector3i idx0(x, y, z);
						w0 = volume0.voxels_[volume0.IndexOf(idx0)]
							.weight_;
						f0 = volume0.voxels_[volume0.IndexOf(idx0)]
							.tsdf_;
						if (color_type_ != integration::TSDFVolumeColorType::None)
							c0 = volume0.voxels_[volume0.IndexOf(idx0)]
							.color_.cast<float>();
						l0 = volume0
							.voxels_[volume0.IndexOf(idx0)]
							.labels_;
						if (w0 != 0.0f && f0 < 0.98f && f0 >= -0.98f) {
							Eigen::Vector3d p0 =
								Eigen::Vector3d(half_voxel_length +
									voxel_length_ * x,
									half_voxel_length +
									voxel_length_ * y,
									half_voxel_length +
									voxel_length_ * z) +
								index0.cast<double>() * volume_unit_length_;
							for (int i = 0; i < 3; i++) {
								Eigen::Vector3d p1 = p0;
								Eigen::Vector3i idx1 = idx0;
								Eigen::Vector3i index1 = index0;
								p1(i) += voxel_length_;
								idx1(i) += 1;
								if (idx1(i) < volume0.resolution_) {
									w1 = volume0.voxels_[volume0.IndexOf(idx1)]
										.weight_;
									f1 = volume0.voxels_[volume0.IndexOf(idx1)]
										.tsdf_;
									if (color_type_ !=
										integration::TSDFVolumeColorType::None)
										c1 = volume0.voxels_[volume0.IndexOf(
											idx1)]
										.color_.cast<float>();
									l1 = volume0.voxels_[volume0.IndexOf(
											idx1)]
										.labels_;
								} else {
									idx1(i) -= volume0.resolution_;
									index1(i) += 1;
									auto unit_itr = volume_units_.find(index1);
									if (unit_itr == volume_units_.end()) {
										w1 = 0.0f;
										f1 = 0.0f;
									} else {
										const auto& volume1 =
											*unit_itr->second.volume_;
										w1 = volume1.voxels_[volume1.IndexOf(
												idx1)]
											.weight_;
										f1 = volume1.voxels_[volume1.IndexOf(
												idx1)]
											.tsdf_;

										if (color_type_ !=
											integration::TSDFVolumeColorType::None)
											c1 = volume1.voxels_[volume1.IndexOf(
												idx1)]
											.color_.cast<float>();

										l1 = volume1.voxels_[volume1.IndexOf(
												idx1)]
											.labels_;
									}
								}
								if (w1 != 0.0f && f1 < 0.98f && f1 >= -0.98f &&
									f0 * f1 < 0) {
									float r0 = std::fabs(f0);
									float r1 = std::fabs(f1);
									Eigen::Vector3d p = p0;
									p(i) = (p0(i) * r1 + p1(i) * r0) /
										(r0 + r1);
									pointcloud->points_.push_back(p);
									if (color_type_ ==
										integration::TSDFVolumeColorType::RGB8) {
										pointcloud->colors_.push_back(
											((c0 * r1 + c1 * r0) /
											(r0 + r1) / 255.0f)
											.cast<double>());
									} else if (color_type_ ==
										integration::TSDFVolumeColorType::Gray32) {
										pointcloud->colors_.push_back(
											((c0 * r1 + c1 * r0) /
											(r0 + r1))
											.cast<double>());
									}


									std::vector<uint8_t> alllabels;
									for (auto element : l0) {
										alllabels.push_back(element);
									}
									for (auto element : l1) {
										alllabels.push_back(element);
									}
									std::vector<float> histogramcount;
									std::vector<uint8_t> histogramvalues;
									int maxvalue = 0;
									int maxclass = -1;


									for (int k = 0; k < alllabels.size(); k++) {
										auto element = alllabels[k];
										bool found = false;
										for (int a = 0; a < histogramvalues.size(); a++) {
											if (histogramvalues[a] == element) {
												histogramcount[a] ++;
												if (maxvalue < histogramcount[a]) {
													maxvalue = histogramcount[a];
													maxclass = histogramvalues[a];
												}
												found = true;
												break;
											}
										}
										if (!found) {
											histogramvalues.push_back(element);
											histogramcount.push_back(1);
										}
									}


									float confidence = 0;
									if (!(maxclass == -1)) {
										confidence = 1.0 * maxvalue / alllabels.size();
										if (alllabels.size() <= 4) {
											confidence = confidence * ((alllabels.size() - 1) * 0.1 + 0.6);
										}

									}

									pointcloud->confidences_.push_back(confidence);
									pointcloud->labels_.push_back(maxclass);
									// has_normal
									pointcloud->normals_.push_back(
										GetNormalAt(p));
								}
							}
						}
					}
				}
			}
		}
	}
	return pointcloud;
}



std::shared_ptr<LabeledTriangleMesh> LabeledScalTSDFVolume::ExtractLabeledTriangleMesh() {
	auto mesh = std::make_shared<LabeledTriangleMesh>();
	double half_voxel_length = voxel_length_ * 0.5;
	std::unordered_map<
		Eigen::Vector4i, int, utility::hash_eigen::hash<Eigen::Vector4i>,
		std::equal_to<Eigen::Vector4i>,
		Eigen::aligned_allocator<std::pair<const Eigen::Vector4i, int>>>
		edgeindex_to_vertexindex;
	int edge_to_index[12];
	for (const auto& unit : volume_units_) {
		if (unit.second.volume_) {
			const auto& volume0 = *unit.second.volume_;
			const auto& index0 = unit.second.index_;
			for (int x = 0; x < volume0.resolution_; x++) {
				for (int y = 0; y < volume0.resolution_; y++) {
					for (int z = 0; z < volume0.resolution_; z++) {
						Eigen::Vector3i idx0(x, y, z);
						int cube_index = 0;
						float w[8];
						float f[8];
						Eigen::Vector3d c[8];
						std::vector<uint8_t> l;
						for (int i = 0; i < 8; i++) {
							Eigen::Vector3i index1 = index0;
							Eigen::Vector3i idx1 = idx0 + shift[i];
							if (idx1(0) < volume_unit_resolution_ &&
								idx1(1) < volume_unit_resolution_ &&
								idx1(2) < volume_unit_resolution_) {
								w[i] = volume0
									.voxels_[volume0.IndexOf(idx1)]
									.weight_;
								f[i] = volume0
									.voxels_[volume0.IndexOf(idx1)]
									.tsdf_;
								auto labels = volume0
									.voxels_[volume0.IndexOf(idx1)]
									.labels_;
								l.reserve(l.size() + labels.size());
								for (auto label : labels) {
									l.push_back(label);
								}
								if (color_type_ == integration::TSDFVolumeColorType::RGB8)
									c[i] = volume0
									.voxels_[volume0.IndexOf(
										idx1)]
									.color_.cast<double>() /
									255.0;
								else if (color_type_ ==
									integration::TSDFVolumeColorType::Gray32)
									c[i] = volume0
									.voxels_[volume0.IndexOf(
										idx1)]
									.color_.cast<double>();
							} else {
								for (int j = 0; j < 3; j++) {
									if (idx1(j) >= volume_unit_resolution_) {
										idx1(j) -= volume_unit_resolution_;
										index1(j) += 1;
									}
								}
								auto unit_itr1 = volume_units_.find(index1);
								if (unit_itr1 == volume_units_.end()) {
									w[i] = 0.0f;
									f[i] = 0.0f;
								} else {
									const auto& volume1 =
										*unit_itr1->second.volume_;
									w[i] = volume1
										.voxels_[volume1.IndexOf(
											idx1)]
										.weight_;
									f[i] = volume1
										.voxels_[volume1.IndexOf(
											idx1)]
										.tsdf_;
									auto labels = volume0
										.voxels_[volume1.IndexOf(
											idx1)]
										.labels_;
									l.reserve(l.size() + labels.size());
									for (auto label : labels) {
										l.push_back(label);
									}
									if (color_type_ ==
										integration::TSDFVolumeColorType::RGB8)
										c[i] = volume1
										.voxels_[volume1.IndexOf(
											idx1)]
										.color_.cast<double>() /
										255.0;
									else if (color_type_ ==
										integration::TSDFVolumeColorType::Gray32)
										c[i] = volume1
										.voxels_[volume1.IndexOf(
											idx1)]
										.color_.cast<double>();
								}
							}
							if (w[i] == 0.0f) {
								cube_index = 0;
								break;
							} else {
								if (f[i] < 0.0f) {
									cube_index |= (1 << i);
								}
							}
						}
						if (cube_index == 0 || cube_index == 255) {
							continue;
						}
						for (int i = 0; i < 12; i++) {
							if (edge_table[cube_index] & (1 << i)) {
								Eigen::Vector4i edge_index =
									Eigen::Vector4i(index0(0), index0(1),
										index0(2), 0) *
									volume_unit_resolution_ +
									Eigen::Vector4i(x, y, z, 0) +
									edge_shift[i];
								if (edgeindex_to_vertexindex.find(edge_index) ==
									edgeindex_to_vertexindex.end()) {
									edge_to_index[i] =
										(int)mesh->vertices_.size();
									edgeindex_to_vertexindex[edge_index] =
										(int)mesh->vertices_.size();
									Eigen::Vector3d pt(
										half_voxel_length +
										voxel_length_ *
										edge_index(0),
										half_voxel_length +
										voxel_length_ *
										edge_index(1),
										half_voxel_length +
										voxel_length_ *
										edge_index(2));
									double f0 = std::abs(
										(double)f[edge_to_vert[i][0]]);
									double f1 = std::abs(
										(double)f[edge_to_vert[i][1]]);
									pt(edge_index(3)) +=
										f0 * voxel_length_ / (f0 + f1);
									mesh->vertices_.push_back(pt);

									std::vector<int> histogramcount;
									std::vector<uint8_t> histogramvalues;
									int maxvalue = 0;
									int maxclass = -1;

									for (uint8_t& element : l) {
										bool found = false;
										for (int a = 0; a < histogramvalues.size(); a++) {
											if (histogramvalues[a] == element) {
												histogramcount[a]++;
												if (maxvalue < histogramcount[a]) {
													maxvalue = histogramcount[a];
													maxclass = histogramvalues[a];
												}
												found = true;
												break;
											}
										}
										if (!found) {
											histogramvalues.push_back(element);
											histogramcount.push_back(1);
										}
									}
									float confidence = 0;
									if (!(maxclass == -1)) {
										confidence = 1.0 * maxvalue / l.size();
										if (l.size() <= 4) {
											confidence = confidence * ((l.size() - 1) * 0.1 + 0.6);
										}
									}

									mesh->vertex_labels.push_back(maxclass);
									mesh->vertex_labels_confidence.push_back(confidence);

									if (color_type_ !=
										integration::TSDFVolumeColorType::None) {
										const auto& c0 = c[edge_to_vert[i][0]];
										const auto& c1 = c[edge_to_vert[i][1]];
										mesh->vertex_colors_.push_back(
											(f1 * c0 + f0 * c1) /
											(f0 + f1));
									}
								} else {
									edge_to_index[i] = edgeindex_to_vertexindex
										[edge_index];
								}
							}
						}
						for (int i = 0; tri_table[cube_index][i] != -1;
							i += 3) {
							mesh->triangles_.push_back(Eigen::Vector3i(
								edge_to_index[tri_table[cube_index][i]],
								edge_to_index[tri_table[cube_index][i + 2]],
								edge_to_index[tri_table[cube_index]
								[i + 1]]));
						}
					}
				}
			}
		}
	}
	return mesh;
}

void LabeledTriangleMesh::ColorLabels(std::pair<std::vector<uint8_t>, std::vector<Eigen::Vector3d>> colormap)
{
	auto histogramvalues = colormap.first;
	auto colorvalues = colormap.second;
	vertex_colors_org = vertex_colors_;
#pragma omp parallel for
	for (int i = 0; i < vertex_colors_.size(); i++) {
		for (int a = 0; a < histogramvalues.size(); a++) {
			if (vertex_labels[i] == histogramvalues[a]) {
				vertex_colors_[i] = colorvalues[a];
				break;
			}
		}
	}
}

void LabeledScalTSDFVolume::IntegratewithLabels(
	const geometry::Image& colorimg,
	const geometry::Image& depthimg,
	const geometry::Image& segmap,
	const camera::PinholeCameraIntrinsic& intrinsic,
	const Eigen::Matrix4d& extrinsic) {
	if ((depthimg.num_of_channels_ != 1) ||
		(depthimg.bytes_per_channel_ != 4) ||
		(depthimg.width_ != intrinsic.width_) ||
		(depthimg.height_ != intrinsic.height_) ||
		(color_type_ == TSDFVolumeColorType::RGB8 &&
			colorimg.num_of_channels_ != 3) ||
			(color_type_ == TSDFVolumeColorType::RGB8 &&
				colorimg.bytes_per_channel_ != 1) ||
				(color_type_ == TSDFVolumeColorType::Gray32 &&
					colorimg.num_of_channels_ != 1) ||
					(color_type_ == TSDFVolumeColorType::Gray32 &&
						colorimg.bytes_per_channel_ != 4) ||
						(color_type_ != TSDFVolumeColorType::None &&
							colorimg.width_ != intrinsic.width_) ||
							(color_type_ != TSDFVolumeColorType::None &&
								colorimg.height_ != intrinsic.height_)) {
		utility::LogWarning(
			"[ScalableTSDFVolume::Integrate] Unsupported image format "
			"xx.\n");
		return;
	}
	auto depth2cameradistance =
		geometry::Image::CreateDepthToCameraDistanceMultiplierFloatImage(
			intrinsic);
	auto pointcloud = geometry::PointCloud::CreateFromDepthImage(
		depthimg, intrinsic, extrinsic, 1000.0, 1000.0,
		depth_sampling_stride_);
	std::unordered_set<Eigen::Vector3i,
		utility::hash_eigen::hash<Eigen::Vector3i>>
		touched_volume_units_;
	for (const auto& point : pointcloud->points_) {
		auto min_bound = LocateVolumeUnit(
			point - Eigen::Vector3d(sdf_trunc_, sdf_trunc_, sdf_trunc_));
		auto max_bound = LocateVolumeUnit(
			point + Eigen::Vector3d(sdf_trunc_, sdf_trunc_, sdf_trunc_));
		for (auto x = min_bound(0); x <= max_bound(0); x++) {
			for (auto y = min_bound(1); y <= max_bound(1); y++) {
				for (auto z = min_bound(2); z <= max_bound(2); z++) {
					auto loc = Eigen::Vector3i(x, y, z);
					if (touched_volume_units_.find(loc) ==
						touched_volume_units_.end()) {
						touched_volume_units_.insert(loc);
						auto volume = OpenVolumeUnit(Eigen::Vector3i(x, y, z));
						volume->IntegrateWithDepthToCameraDistanceMultiplier(
							colorimg, depthimg, segmap, intrinsic,
							extrinsic, *depth2cameradistance);
					}
				}
			}
		}
	}
}

void LabeledScalTSDFVolume::IntegrateonlyLabels(
	const geometry::Image& depthimg,
	const geometry::Image& segmap,
	const camera::PinholeCameraIntrinsic& intrinsic,
	const Eigen::Matrix4d& extrinsic) {
	if ((depthimg.num_of_channels_ != 1) ||
		(depthimg.bytes_per_channel_ != 4) ||
		(depthimg.width_ != intrinsic.width_) ||
		(depthimg.height_ != intrinsic.height_)) {
		utility::LogWarning(
			"[ScalableTSDFVolume::Integrate] Unsupported image format "
			"xx.\n");
		return;
	}
	auto depth2cameradistance =
		geometry::Image::CreateDepthToCameraDistanceMultiplierFloatImage(
			intrinsic);
	auto pointcloud = geometry::PointCloud::CreateFromDepthImage(
		depthimg, intrinsic, extrinsic, 1000.0, 1000.0,
		depth_sampling_stride_);
	std::unordered_set<Eigen::Vector3i,
		utility::hash_eigen::hash<Eigen::Vector3i>>
		touched_volume_units_;
	for (const auto& point : pointcloud->points_) {
		auto min_bound = LocateVolumeUnit(
			point - Eigen::Vector3d(sdf_trunc_, sdf_trunc_, sdf_trunc_));
		auto max_bound = LocateVolumeUnit(
			point + Eigen::Vector3d(sdf_trunc_, sdf_trunc_, sdf_trunc_));
		for (auto x = min_bound(0); x <= max_bound(0); x++) {
			for (auto y = min_bound(1); y <= max_bound(1); y++) {
				for (auto z = min_bound(2); z <= max_bound(2); z++) {
					auto loc = Eigen::Vector3i(x, y, z);
					if (touched_volume_units_.find(loc) ==
						touched_volume_units_.end()) {
						touched_volume_units_.insert(loc);
						auto volume = OpenVolumeUnit(Eigen::Vector3i(x, y, z));
						volume->IntegrateOnlyLabel(depthimg, segmap, intrinsic,
							extrinsic, *depth2cameradistance);
					}
				}
			}
		}
	}
}

// custom
bool LabeledScalTSDFVolume::IntegratewhileCounting(
	const geometry::Image& colorimg,
	const geometry::Image& depthimg,
	const camera::PinholeCameraIntrinsic& intrinsic,
	const Eigen::Matrix4d& extrinsic,
	double ratio) {
	if ((depthimg.num_of_channels_ != 1) ||
		(depthimg.bytes_per_channel_ != 4) ||
		(depthimg.width_ != intrinsic.width_) ||
		(depthimg.height_ != intrinsic.height_) ||
		(color_type_ == TSDFVolumeColorType::RGB8 &&
			colorimg.num_of_channels_ != 3) ||
			(color_type_ == TSDFVolumeColorType::RGB8 &&
				colorimg.bytes_per_channel_ != 1) ||
				(color_type_ == TSDFVolumeColorType::Gray32 &&
					colorimg.num_of_channels_ != 1) ||
					(color_type_ == TSDFVolumeColorType::Gray32 &&
						colorimg.bytes_per_channel_ != 4) ||
						(color_type_ != TSDFVolumeColorType::None &&
							colorimg.width_ != intrinsic.width_) ||
							(color_type_ != TSDFVolumeColorType::None &&
								colorimg.height_ != intrinsic.height_)) {
		utility::LogWarning(
			"[ScalableTSDFVolume::Integrate] Unsupported image format "
			".\n"
		);
		std::cout << "depthimg.num_of_channels_ " << depthimg.num_of_channels_ << std::endl;
		std::cout << "depthimg.bytes_per_channel_ " << depthimg.bytes_per_channel_
			<< std::endl;
		std::cout << "colorimg.num_of_channels_ " << colorimg.num_of_channels_
			<< std::endl;
		std::cout << "colorimg.bytes_per_channel_ " << colorimg.bytes_per_channel_
			<< std::endl;
		return false;
	}
	auto depth2cameradistance =
		geometry::Image::CreateDepthToCameraDistanceMultiplierFloatImage(
			intrinsic);
	auto pointcloud = geometry::PointCloud::CreateFromDepthImage(
		depthimg, intrinsic, extrinsic, 1000.0, 1000.0,
		depth_sampling_stride_);
	std::unordered_set<Eigen::Vector3i,
		utility::hash_eigen::hash<Eigen::Vector3i>>
		touched_volume_units_;
	int num_labeled_voxels = 0;
	int num_voxels = 0;
	for (const auto& point : pointcloud->points_) {
		auto min_bound = LocateVolumeUnit(
			point - Eigen::Vector3d(sdf_trunc_, sdf_trunc_, sdf_trunc_));
		auto max_bound = LocateVolumeUnit(
			point + Eigen::Vector3d(sdf_trunc_, sdf_trunc_, sdf_trunc_));
		for (auto x = min_bound(0); x <= max_bound(0); x++) {
			for (auto y = min_bound(1); y <= max_bound(1); y++) {
				for (auto z = min_bound(2); z <= max_bound(2); z++) {
					auto loc = Eigen::Vector3i(x, y, z);
					if (touched_volume_units_.find(loc) ==
						touched_volume_units_.end()) {
						touched_volume_units_.insert(loc);
						auto volume = OpenVolumeUnit(Eigen::Vector3i(x, y, z));
						auto pair = volume->IntegrateWithLabelCount(
							colorimg, depthimg, intrinsic, extrinsic,
							*depth2cameradistance);
						num_voxels += pair.first;
						num_labeled_voxels += pair.second;
					}
				}
			}
		}
	}

	if (((double)num_labeled_voxels / (double)num_voxels) < ratio) {
		return true;
	} else {
		return false;
	}

}