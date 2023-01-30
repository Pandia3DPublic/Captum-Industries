#include "labelutil.h"
#include "pythonutil.h"
#include <iostream>
#include <fstream>


using namespace open3d;
using namespace geometry;
//using namespace std::chrono;

//############################################################Headerless helper functions##########################################




//#############################################Header functions ######################################################

double determineAccuracy(LabeledPointCloud& lpcd, LabeledTriangleMesh& lmesh) {
	KDTreeFlann kdtree;

	double globalaccuracy = 0;
	double globaln = 0;
	kdtree.SetGeometry(lpcd);


#pragma omp parallel 
	{
		double n = 0;
		double accuracy = 0;
#pragma omp for
		for (int i = 0; i < lmesh.vertices_.size(); i++) {
			if (lmesh.vertex_labels[i] != 255) {
				std::vector<int> tmp_indices;
				std::vector<double> dist;
				kdtree.SearchKNN(lmesh.vertices_[i], 1, tmp_indices, dist);
				if (dist.size() > 0) {
					if (dist[0] < 0.05) {
						auto nearest_label = lpcd.labels_[tmp_indices[0]];
						n++;
						if (nearest_label == lmesh.vertex_labels[i]) {
							accuracy = (accuracy * (n - 1) + 1) / n;
						} else {
							accuracy = (accuracy * (n - 1)) / n;
						}
					}
				}
			}
		}
#pragma omp critical
		{
			globaln += n;
			globalaccuracy = ((globalaccuracy * (globaln - n) + (accuracy * n)) / globaln);
		}
	}
	return globalaccuracy;
}


double determineAccuracy(LabeledTriangleMesh& lmeshGround, LabeledTriangleMesh& lmeshpred) {

	double accuracy = 0;
	double n = 0;
	for (int i = 0; i < lmeshGround.vertex_labels.size(); i++) {
		if (lmeshGround.vertex_labels[i] != 255) {
			n++;
			if (lmeshpred.vertex_labels[i] == lmeshGround.vertex_labels[i]) {
				accuracy++;
			}

		}
	}
	return accuracy / n;
}

double determineunkown(LabeledTriangleMesh& lmesh) {

	int sum = 0;

	//#pragma omp parallel for reduction(+:sum)
	for (auto vertexlabel : lmesh.vertex_labels) {
		if (vertexlabel == 255) {
			sum++;
		}
	}

	return ((double)sum) / (double)lmesh.vertex_labels.size();
}


std::pair< std::vector<uint8_t>, std::vector<Eigen::Vector3d>> createColorMap(int numofValues, bool range) {
	std::vector<uint8_t> histogramvalues;
	for (int i = 0; i <= numofValues; i++) {
		histogramvalues.push_back(i);
	}

	std::vector<Eigen::Vector3d> colorvalues;
	for (int i = 0; i < numofValues; i++) {
		/* double color1 = i * 256 / num_unique_elements;
		 double color2 = 255.0 - (i * 2 * 256 / num_unique_elements) % 256;
		 double color3 = 255.0 - (i * 3 * 256 / num_unique_elements) % 256;
		 Eigen::Vector3d colorvec(color1 / 255.0, color2 / 255.0, color3 / 255.0);*/
		float hue = 360.0 * i / numofValues;
		float saturation;
		float value;
		if (range) {
			saturation = 0.5 + (i % 2) / 2.0;
			value = 0.5 + (i % 2) / 2.0;
		}
		else {
			saturation = 1;
			value = 1;
		}

		float r, g, b;
		HSVtoRGB(r, g, b, hue, saturation, value);
		Eigen::Vector3d colorvec(r, g, b);
		// std::cout << "rgb: " << colorvec << '\n';
		colorvalues.push_back(colorvec);
	}



	return std::pair< std::vector<uint8_t>, std::vector<Eigen::Vector3d>>(histogramvalues, colorvalues);
}


std::shared_ptr<geometry::Image> majorityVote(
	std::shared_ptr<geometry::Image> image_ptr, int kernel_size) {
	geometry::Image image = *image_ptr;
	geometry::Image outputimg;
	outputimg.Prepare(image.width_, image.height_, 1, 4);
	int offsetsize = kernel_size / 2;
	std::vector<int> histogramcount;
	std::vector<float> histogramvalues;

	// check for unique values

	for (int x = 0; x < image.width_; x++) {
		for (int y = 0; y < image.height_; y++) {
			float value = *image.PointerAt<float>(x, y);
			bool found = false;
			for (int a = 0; a < histogramvalues.size(); a++) {
				if (histogramvalues[a] == value) {
					//  histogramcount[a]++;
					found = true;
					break;
				}
			}
			if (!found) {
				histogramvalues.push_back(value);
				histogramcount.push_back(0);
			}
		}
	}

#pragma omp parallel for firstprivate(histogramvalues, histogramcount)
	for (int x = offsetsize; x < image.width_ - offsetsize; x++) {
		// Histogramm erstellen
		for (int i = 0; i < histogramcount.size(); i++) {
			histogramcount[i] = 0;
		}
		for (int i = -offsetsize; i <= offsetsize; i++) {
			for (int j = -offsetsize; j <= offsetsize; j++) {
				float value = *image.PointerAt<float>(x + i, offsetsize + j);
				for (int a = 0; a < histogramcount.size(); a++) {
					if (histogramvalues[a] == value) {
						histogramcount[a]++;
						break;
					}
				}
			}
		}
		for (int y = offsetsize; y < image.height_ - offsetsize; y++) {
			if (y != offsetsize) {
				// Fenster verschieben
				for (int j = -offsetsize; j <= offsetsize; j++) {
					float value =
						*image.PointerAt<float>(x + j, y + offsetsize);
					for (int i = 0; i < histogramcount.size(); i++) {
						if (histogramvalues[i] == value) {
							histogramcount[i]++;
							break;
						}
					}
				}
				for (int j = -offsetsize; j <= offsetsize; j++) {
					float value = *image.PointerAt<float>(x + j,
						y - (offsetsize + 1));
					for (int i = 0; i < histogramcount.size(); i++) {
						if (histogramvalues[i] == value) {
							histogramcount[i]--;
							break;
						}
					}
				}
			}
			// histogram auszählen
			float maxvalue = 0;
			int maxcount = 0;
			for (int i = 0; i < histogramcount.size(); i++) {
				if (maxcount < histogramcount[i]) {
					maxvalue = histogramvalues[i];
					maxcount = histogramcount[i];
				}
			}
			*outputimg.PointerAt<float>(x, y) = maxvalue;
		}
	}
	return std::make_shared<geometry::Image>(outputimg);
}


std::shared_ptr<geometry::Image> colorImageLabels(std::shared_ptr<geometry::Image> image_ptr, std::pair<std::vector<uint8_t>, std::vector<Eigen::Vector3d>> colormap)
{
	geometry::Image image = *image_ptr;
	geometry::Image outputimg;
	auto histogramvalues = colormap.first;
	auto colorvalues = colormap.second;
	outputimg.Prepare(image.width_, image.height_, 3, 1);
	for (int x = 0; x < image.width_; x++) {
		for (int y = 0; y < image.height_; y++) {
			float value = *image.PointerAt<uint8_t>(x, y);
			for (int a = 0; a < histogramvalues.size(); a++) {
				if (value == histogramvalues[a]) {
					for (int ch = 0; ch < 3; ch++) {
						*outputimg.PointerAt<uint8_t>(x, y, ch) = (uint8_t)(255.0 * colorvalues[a](ch));
					}
					break;
				}
			}
		}
	}
	return std::make_shared<geometry::Image>(outputimg);
}
std::shared_ptr<geometry::Image> getSegmentedImagePython(geometry::Image& img_ptr, py::module segment, py::object model) {

	py::array result = segment.attr("segment_img")(img_ptr, model);//stuck here
	auto img = py_object_to_o3dimg(result);
	return img;
}


int findmaxfreqelem(std::vector<uint8_t>& vec) {
	std::vector<int> histogramcount;
	std::vector<uint8_t> histogramvalues;
	int maxvalue = 0;
	int maxclass = -1;


	for (uint8_t& element : vec) {
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

	// if (maxclass == -1) std::cout << "no class found" << std::endl;
	return maxclass;
}

void determineIOU(LabeledTriangleMesh& lmeshGround, LabeledTriangleMesh& lmeshpred, std::vector<IOU>& IOUs) {

	for (auto curlabel : lmeshGround.vertex_labels) {
		bool found = false;
		for (int i = 0; i < IOUs.size(); i++) {
			if (curlabel == IOUs[i].label) {
				found = true;
				break;
			}
		}
		if (found == false) {
			IOU iou;
			iou.label = curlabel;
			IOUs.push_back(iou);
		}
	}

	for (auto curlabel : lmeshpred.vertex_labels) {
		bool found = false;
		for (int i = 0; i < IOUs.size(); i++) {
			if (curlabel == IOUs[i].label) {
				found = true;
				break;
			}
		}
		if (found == false) {
			IOU iou;
			iou.label = curlabel;
			IOUs.push_back(iou);
		}
	}

#pragma omp parallel for
	for (int l = 0; l < IOUs.size(); l++) {
		auto curlabel = IOUs[l].label;

		for (int i = 0; i < lmeshGround.vertex_labels.size(); i++) {
			if (lmeshGround.vertex_labels[i] == curlabel) {
				if (lmeshpred.vertex_labels[i] == curlabel) {
					IOUs[l].true_positive++;
				} else {
					IOUs[l].false_negative++;
				}
			} else {
				if (lmeshpred.vertex_labels[i] == curlabel) {
					IOUs[l].false_positive++;
				}
			}
		}
		IOUs[l].value = (double)IOUs[l].true_positive / (double)(IOUs[l].true_positive + IOUs[l].false_positive + IOUs[l].false_negative);
	}
	for (int l = 0; l < IOUs.size(); l++) {
		std::cout << "label " << (int)IOUs[l].label << " IOU: " << IOUs[l].value << std::endl;
	}
}
std::shared_ptr<LabeledTriangleMesh> getLabeledMesh(LabeledPointCloud& lpcd, LabeledTriangleMesh& lmesh, uint8_t defaultlabel) {
	KDTreeFlann kdtree;
	std::shared_ptr<LabeledTriangleMesh> out = std::make_shared<LabeledTriangleMesh>(lmesh);
	out->vertex_labels.clear();
	//std::cout << "lmesh.vertex_labels.size: " << lmesh.vertex_labels.size() << std::endl;
	kdtree.SetGeometry(lpcd);
	for (int i = 0; i < out->vertices_.size(); i++) {
		uint8_t curlabel = defaultlabel;
		std::vector<int> tmp_indices;
		std::vector<double> dist;
		kdtree.SearchKNN(out->vertices_[i], 1, tmp_indices, dist);
		if (dist.size() > 0) {
			if (dist[0] < 0.05) {
				curlabel = lpcd.labels_[tmp_indices[0]];
			}
		}
		out->vertex_labels.push_back(curlabel);
	}
	return out;
}

void saveColorMap(const std::pair< std::vector<uint8_t>, std::vector<Eigen::Vector3d>>& colormap, const std::vector<std::string>& names, const std::string& filename) {
	std::vector<uint8_t> histogramvalues = colormap.first;
	std::vector<Eigen::Vector3d> colorvalues = colormap.second;
	std::ofstream outFile(filename);
	if (outFile.is_open())
	{
		for (int i = 0; i < names.size(); i++) {
			if (names[i] != "") {
				for (int a = 0; a < histogramvalues.size(); a++) {
					if (histogramvalues[a] == i) {
						outFile << (int)histogramvalues[a] << "," << names[i] << "," << colorvalues[a](0) << "," << colorvalues[a](1) << "," << colorvalues[a](2) << std::endl;
						break;
					}
				}
			}
		}
		outFile.close();
	} else {
		std::cout << "unable to open " << filename << std::endl;
	}
}



void RGBtoHSV(float& fR, float& fG, float fB, float& fH, float& fS, float& fV) {
	float fCMax = std::max(std::max(fR, fG), fB);
	float fCMin = std::min(std::min(fR, fG), fB);
	float fDelta = fCMax - fCMin;

	if (fDelta > 0) {
		if (fCMax == fR) {
			fH = 60 * (fmod(((fG - fB) / fDelta), 6));
		} else if (fCMax == fG) {
			fH = 60 * (((fB - fR) / fDelta) + 2);
		} else if (fCMax == fB) {
			fH = 60 * (((fR - fG) / fDelta) + 4);
		}

		if (fCMax > 0) {
			fS = fDelta / fCMax;
		} else {
			fS = 0;
		}

		fV = fCMax;
	} else {
		fH = 0;
		fS = 0;
		fV = fCMax;
	}

	if (fH < 0) {
		fH = 360 + fH;
	}
}



void HSVtoRGB(float& fR, float& fG, float& fB, float& fH, float& fS, float& fV) {
	float fC = fV * fS; // Chroma
	float fHPrime = fmod(fH / 60.0, 6);
	float fX = fC * (1 - fabs(fmod(fHPrime, 2) - 1));
	float fM = fV - fC;

	if (0 <= fHPrime && fHPrime < 1) {
		fR = fC;
		fG = fX;
		fB = 0;
	} else if (1 <= fHPrime && fHPrime < 2) {
		fR = fX;
		fG = fC;
		fB = 0;
	} else if (2 <= fHPrime && fHPrime < 3) {
		fR = 0;
		fG = fC;
		fB = fX;
	} else if (3 <= fHPrime && fHPrime < 4) {
		fR = 0;
		fG = fX;
		fB = fC;
	} else if (4 <= fHPrime && fHPrime < 5) {
		fR = fX;
		fG = 0;
		fB = fC;
	} else if (5 <= fHPrime && fHPrime < 6) {
		fR = fC;
		fG = 0;
		fB = fX;
	} else {
		fR = 0;
		fG = 0;
		fB = 0;
	}

	fR += fM;
	fG += fM;
	fB += fM;
}
