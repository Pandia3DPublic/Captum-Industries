#pragma once
#include "Open3D/Open3D.h"
#include <opencv2/opencv.hpp>

using namespace open3d;

std::shared_ptr<geometry::Image> resizeImage(std::shared_ptr<geometry::Image> image_ptr, int width, int height, std::string mode); 
void topencv(std::shared_ptr<geometry::Image>& img, cv::Mat& cvimg);
