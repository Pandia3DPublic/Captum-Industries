#pragma once
#include <vector>
#include <Eigen/Dense>
#include <Open3D/Open3D.h>
#include "Shader.h"
#include <Cuda/Open3DCuda.h>

using namespace std;
class GLGeometry {
public:
	//points, colors, indices , vertex location, color location
	GLGeometry(std::shared_ptr<open3d::geometry::TriangleMesh> mesh,GLuint vertex_location, GLuint color_location, GLuint normal_location = -1);
	GLGeometry(open3d::cuda::TriangleMeshCuda& mesh_ex, GLuint vertex_location_ex, GLuint color_location_ex, GLuint normal_location = -1);
	GLGeometry(open3d::cuda::ImageCuda<float, 3>& image, GLuint vertex_location, GLuint color_location = -1, GLuint normal_loction = -1);

	~GLGeometry();

	void changeGeometry(open3d::cuda::TriangleMeshCuda& mesh);
	void changeGeometry(open3d::cuda::ImageCuda<float,3>& image);

	void setGeometry(open3d::cuda::TriangleMeshCuda& mesh);
	void setGeometry(open3d::cuda::ImageCuda<float, 3>& image);
	
	void deleteGeometry();

	void getPointsandColorfromMesh(vector<Eigen::Vector3f>& points, vector<Eigen::Vector3f>& colors, shared_ptr<open3d::geometry::TriangleMesh> mesh, vector<Eigen::Vector3f>* normals = 0);
	
	void bind();
	void draw();


	std::vector<Eigen::Vector3f> points;
	shared_ptr<open3d::geometry::TriangleMesh> mesh_;
	cudaGraphicsResource_t vertex_position_cuda_resource_;
	cudaGraphicsResource_t vertex_color_cuda_resource_;
	cudaGraphicsResource_t triangle_cuda_resource_;
	cudaGraphicsResource_t normals_cuda_resource_;
	bool usingCudaResources = false;
	GLuint vertex_location_;
	GLuint color_location_;
	GLuint normal_location_;
	unsigned int VAO, vertex_position_buffer, vertex_color_buffer, triange_indice_buffer, normal_buffer;
	int ndrawElements = 0;
	template<typename T>
	bool CopyDataToCudaGraphicsResource(GLuint &opengl_buffer,cudaGraphicsResource_t &cuda_graphics_resource,T *cuda_vector, size_t cuda_vector_size);
	enum InputTyp {geo_triangleMesh, geo_triangleMeshCuda, geo_imageCuda} geometry_typ;

	//template<typename T>
	//bool RegisterResource(cudaGraphicsResource_t &cuda_graphics_resource,	GLenum opengl_buffer_type,	GLuint &opengl_buffer,	T *cuda_vector,	size_t cuda_vector_size);
	//bool GLGeometry::UnregisterResource(cudaGraphicsResource_t &cuda_graphics_resource,	GLuint &opengl_buffer);

	

};