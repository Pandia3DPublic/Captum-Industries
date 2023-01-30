#pragma once
#include "pybind11/embed.h"
#include "Python/open3d_pybind.h"
#include <Open3D/Open3D.h>

//converts a python numpy array to an open3d image.
std::shared_ptr<open3d::geometry::Image> py_object_to_o3dimg(py::buffer b);