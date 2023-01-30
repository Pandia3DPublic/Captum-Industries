#include "imageutil.h"


std::shared_ptr<geometry::Image> resizeImage(std::shared_ptr<geometry::Image> image_ptr, int width, int height, std::string mode) {



	geometry::Image out;
	out.Prepare(width, height, image_ptr->num_of_channels_, image_ptr->bytes_per_channel_);
	//auto image = *image_ptr;
	double resize_factorH = (double)image_ptr->height_ / (double)height;
	double resize_factorW = (double)image_ptr->width_ / (double)width;

	//		// std::cout << "resize height: " << resize_height << std::endl;
	//		if (mode == "bilinear") { //todo komische striche noch wegmachen
	//#pragma omp parallel for
	//			for (int x = 0; x < returnimg.width_; x++) {
	//				for (int y = 0; y < returnimg.height_; y++) {
	//					double xres = resize_width * x;
	//					double yres = resize_height * y;
	//					uint8_t bottom_right = *image.PointerAt<uint8_t>(ceil(xres), ceil(yres));
	//					uint8_t bottom_left = *image.PointerAt<uint8_t>(floor(xres), ceil(yres));
	//					uint8_t top_right = *image.PointerAt<uint8_t>(ceil(xres), floor(yres));
	//					uint8_t top_left = *image.PointerAt<uint8_t>(floor(xres), floor(yres));
	//					*returnimg.PointerAt<uint8_t>(x, y) = (ceil(xres) - xres) * (ceil(yres) - yres) * top_left +
	//						(floor(xres) - xres) * (floor(yres) - yres) * bottom_right +
	//						(xres - floor(xres)) * (ceil(yres) - yres) * top_right +
	//						(ceil(xres) - xres) * (yres - floor(yres)) * bottom_left;
	//				}
	//			}
	//		}

	if (image_ptr->bytes_per_channel_ == 1) {
		if (mode == "nointerpol") {
#pragma omp parallel for
			for (int x = 0; x < out.width_; x++) {
				for (int y = 0; y < out.height_; y++) {
					for (int ch = 0; ch < image_ptr->num_of_channels_; ch++) {
						*out.PointerAt<uint8_t>(x, y, ch) = *(image_ptr->PointerAt<uint8_t>(floor(resize_factorW * x), floor(resize_factorH * y), ch));
					}
				}
			}
		}
	} else {
		if (image_ptr->bytes_per_channel_ == 2) {
			if (mode == "nointerpol") {
#pragma omp parallel for
				for (int x = 0; x < out.width_; x++) {
					for (int y = 0; y < out.height_; y++) {
						*out.PointerAt<uint16_t>(x, y) = *(image_ptr->PointerAt<uint16_t>(floor(resize_factorW * x), floor(resize_factorH * y)));
					}
				}
			}
		}
	}
	return std::make_shared<geometry::Image>(out);
}


void topencv(std::shared_ptr<geometry::Image>& img, cv::Mat& cvimg) {
	cvimg = cv::Mat(img->height_, img->width_, CV_8UC3);
	for (int i = 0; i < img->height_; i++) {
		uint8_t* pixel = cvimg.ptr<uint8_t>(i); // point to first color in row
		for (int j = 0; j < img->width_; j++) {
			*pixel++ = *img->PointerAt<uint8_t>(j, i, 2);
			*pixel++ = *img->PointerAt<uint8_t>(j, i, 1);
			*pixel++ = *img->PointerAt<uint8_t>(j, i, 0);

		}
	}


}
