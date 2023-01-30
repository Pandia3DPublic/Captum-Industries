#include "pythonutil.h"
using namespace open3d;
std::shared_ptr<geometry::Image> py_object_to_o3dimg(py::buffer b) {
	py::buffer_info info = b.request();
	int width, height, num_of_channels = 0, bytes_per_channel;
	if (info.format == py::format_descriptor<uint8_t>::format() ||
		info.format == py::format_descriptor<int8_t>::format()) {
		bytes_per_channel = 1;
	} else if (info.format ==
		py::format_descriptor<uint16_t>::format() ||
		info.format ==
		py::format_descriptor<int16_t>::format()) {
		bytes_per_channel = 2;
	} else if (info.format == py::format_descriptor<float>::format()) {
		bytes_per_channel = 4;
	} else {
		throw std::runtime_error(
			"Image can only be initialized from buffer of uint8, "
			"uint16, or float!");
	}
	if (info.strides[info.ndim - 1] != bytes_per_channel) {
		throw std::runtime_error(
			"Image can only be initialized from c-style buffer.");
	}
	if (info.ndim == 2) {
		num_of_channels = 1;
	} else if (info.ndim == 3) {
		num_of_channels = (int)info.shape[2];
	}
	height = (int)info.shape[0];
	width = (int)info.shape[1];
	auto img = std::make_shared<geometry::Image>();
	img->Prepare(width, height, num_of_channels, bytes_per_channel);
	memcpy(img->data_.data(), info.ptr, img->data_.size());
	return img;
}