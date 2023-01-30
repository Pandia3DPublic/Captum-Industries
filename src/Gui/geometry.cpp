#include "core/threadvars.h"
#include "geometry.h"
#include <cuda_gl_interop.h>
#include <Cuda/Common/UtilsCuda.h>
#include <cassert>
#include "ptimer.h"


using namespace std;
using namespace open3d;
GLGeometry::GLGeometry(open3d::cuda::TriangleMeshCuda& mesh, GLuint vertex_location, GLuint color_location, GLuint normal_location){
	vertex_location_ = vertex_location;
	color_location_ = color_location;
	normal_location_ = normal_location;
	geometry_typ = InputTyp::geo_triangleMeshCuda;
	setGeometry(mesh);

}

void GLGeometry::changeGeometry(open3d::cuda::TriangleMeshCuda& mesh) {
	if (ndrawElements != 0)
		deleteGeometry();
	setGeometry(mesh);
}

void GLGeometry::changeGeometry(open3d::cuda::ImageCuda<float, 3>& image) {
	deleteGeometry();
	setGeometry(image);
}


void GLGeometry::deleteGeometry() {

	if (wglGetCurrentContext() != NULL && usingCudaResources) {

		CheckCuda(cudaGraphicsUnregisterResource(vertex_position_cuda_resource_));

		if (geometry_typ != geo_imageCuda) { //prevents cuda crash if raycasted is build
			CheckCuda(cudaGraphicsUnregisterResource(vertex_color_cuda_resource_));
			CheckCuda(cudaGraphicsUnregisterResource(normals_cuda_resource_));
			CheckCuda(cudaGraphicsUnregisterResource(triangle_cuda_resource_));
		}
	}
	glBindVertexArray(0);
	glDeleteBuffers(1, &vertex_position_buffer);
	glDeleteBuffers(1, &vertex_color_buffer);
	glDeleteBuffers(1, &triange_indice_buffer);
	glDeleteBuffers(1, &normal_buffer);
	glDeleteVertexArrays(1, &VAO);
	points.clear();
}

//assumes that vertex position, colors and normals are called 
//in vec3 vertex_position;
//in vec3 vertex_normal;
//in vec3 vertex_color;
//in shader in that order
void GLGeometry::setGeometry(open3d::cuda::TriangleMeshCuda& mesh) {

	//save current vao to reset later
	int oldVAO;
	glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &oldVAO);

	meshlock.lock();
	if (mesh.triangles_.size() != 0){

		usingCudaResources = true; //needs to be moved in if-case so that program won't crash if user selects folders without images

		//fill points here for bounding box evaluation todo make unnecessary
		//vertcpu = mesh.vertices_.Download();
		// Prepare data to be passed to GPU
		cuda::Vector3f *vertices, *colors, *normals;
		cuda::Vector3i *triangles;
		vertices = mesh.vertices_.device_->data();
		colors = mesh.vertex_colors_.device_->data();
		triangles = mesh.triangles_.device_->data();
		normals = mesh.vertex_normals_.device_->data();
		int vertex_size = mesh.vertices_.size();
		int triangle_size = mesh.triangles_.size();

		if (mesh.vertex_colors_.size() != mesh.vertices_.size()  || mesh.vertex_colors_.size() != mesh.vertex_normals_.size()) {
			utility::LogError("not all vertices have color or normal in Glgeometry! \n");
		}

		glGenVertexArrays(1, &VAO);  
		glGenBuffers(1, &vertex_position_buffer);
		glGenBuffers(1, &vertex_color_buffer);
		glGenBuffers(1, &triange_indice_buffer);
		glGenBuffers(1, &normal_buffer);

		glBindVertexArray(VAO);

		//position
		glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer);  
		glBufferData(GL_ARRAY_BUFFER, sizeof(cuda::Vector3f) * vertex_size, nullptr, GL_STATIC_DRAW); //make empty buffer here!
		CheckCuda(cudaGraphicsGLRegisterBuffer(&vertex_position_cuda_resource_,	vertex_position_buffer,	cudaGraphicsMapFlagsReadOnly)); // register graphics buffer as cuda resource
		CopyDataToCudaGraphicsResource(vertex_position_buffer, vertex_position_cuda_resource_, vertices, vertex_size);
		glVertexAttribPointer(vertex_location_, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(vertex_location_);

		//color
		glBindBuffer(GL_ARRAY_BUFFER, vertex_color_buffer);  
		glBufferData(GL_ARRAY_BUFFER, sizeof(cuda::Vector3f) * vertex_size,nullptr, GL_STATIC_DRAW); 
		CheckCuda(cudaGraphicsGLRegisterBuffer(&vertex_color_cuda_resource_,vertex_color_buffer,cudaGraphicsMapFlagsReadOnly)); 
		CopyDataToCudaGraphicsResource(vertex_color_buffer, vertex_color_cuda_resource_, colors, vertex_size);
		glVertexAttribPointer(color_location_, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(color_location_);

		//triangles
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triange_indice_buffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER,  sizeof(cuda::Vector3i) * triangle_size, nullptr, GL_STATIC_DRAW);
		CheckCuda(cudaGraphicsGLRegisterBuffer(&triangle_cuda_resource_,triange_indice_buffer,cudaGraphicsMapFlagsReadOnly));
		CopyDataToCudaGraphicsResource(triange_indice_buffer, triangle_cuda_resource_, triangles, triangle_size);

		if (normal_location_ != -1){
			//normals
			glBindBuffer(GL_ARRAY_BUFFER, normal_buffer);  
			glBufferData(GL_ARRAY_BUFFER, sizeof(cuda::Vector3f) * vertex_size,nullptr, GL_STATIC_DRAW); 
			CheckCuda(cudaGraphicsGLRegisterBuffer(&normals_cuda_resource_,normal_buffer,cudaGraphicsMapFlagsReadOnly)); 
			CopyDataToCudaGraphicsResource(normal_buffer, normals_cuda_resource_, normals, vertex_size);
			glVertexAttribPointer(normal_location_, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(normal_location_);
		}

		glBindVertexArray(oldVAO);


		ndrawElements = mesh.triangles_.size() *3;
	} else {
		ndrawElements =0;
	}
	meshlock.unlock();

}



GLGeometry::GLGeometry(shared_ptr<geometry::TriangleMesh> mesh, GLuint vertex_location, GLuint color_location, GLuint normal_location) {
	vector<Eigen::Vector3f> colors;
	vector<Eigen::Vector3f> normals;
	mesh_ = mesh;
	vertex_location_ = vertex_location;
	color_location_ = color_location;
	normal_location_ = normal_location;
	geometry_typ = InputTyp::geo_triangleMesh;
	//save current vao to reset later
	int oldVAO;
	glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &oldVAO);

	getPointsandColorfromMesh(points,colors,mesh, &normals); //normals pointer just for compatibility with non normals vector meshes

	glGenVertexArrays(1, &VAO);  
	glGenBuffers(1, &vertex_position_buffer);
	glGenBuffers(1, &vertex_color_buffer);
	glGenBuffers(1, &normal_buffer);
	glGenBuffers(1, &triange_indice_buffer);

	glBindVertexArray(VAO);

	//position
	glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer);  
	glBufferData(GL_ARRAY_BUFFER, sizeof(Eigen::Vector3f) * points.size(), points.data(), GL_STATIC_DRAW); 
	glVertexAttribPointer(vertex_location, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(vertex_location);


	//color
	glBindBuffer(GL_ARRAY_BUFFER, vertex_color_buffer);  
	glBufferData(GL_ARRAY_BUFFER, sizeof(Eigen::Vector3f) * colors.size(), colors.data(), GL_STATIC_DRAW); 
	glVertexAttribPointer(color_location, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(color_location);

	//normals
	if (normal_location_ != -1){
		glBindBuffer(GL_ARRAY_BUFFER, normal_buffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Eigen::Vector3f) * normals.size(), normals.data(), GL_STATIC_DRAW);
		glVertexAttribPointer(normal_location, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(normal_location);
	}

	//triangles
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triange_indice_buffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER,  sizeof(Eigen::Vector3i) * mesh->triangles_.size(), mesh->triangles_.data(), GL_STATIC_DRAW);
	glBindVertexArray(oldVAO);
	ndrawElements = mesh->triangles_.size() *3;

}


GLGeometry::GLGeometry(open3d::cuda::ImageCuda<float, 3>& image, GLuint vertex_location, GLuint color_location, GLuint normal_loction){

	vertex_location_ = vertex_location;
	color_location_ = color_location;
	normal_location_ = normal_loction;
	geometry_typ = InputTyp::geo_imageCuda;
	setGeometry(image);	
}



void GLGeometry::bind() {
	glBindVertexArray(VAO);
}

void GLGeometry::draw() {
	bind();
	if(geometry_typ == geo_imageCuda)
		glDrawArrays(GL_POINTS, 0, ndrawElements); //needs to be glDrawArrays. glDrawElements won't work
	else
		glDrawElements(GL_TRIANGLES, ndrawElements, GL_UNSIGNED_INT, 0); 
}


void GLGeometry::setGeometry(open3d::cuda::ImageCuda<float, 3>& image) {

	usingCudaResources = true;
	int oldVAO;
	glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &oldVAO);
	int image_size = image.width_ * image.height_;
	cuda::Vector3f* vertices, *colors, *normals;
	vertices = image.device_->data();
	

	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &vertex_position_buffer);
	glGenBuffers(1, &vertex_color_buffer);
	glGenBuffers(1, &normal_buffer);
	//glGenBuffers(1, &triange_indice_buffer);

	glBindVertexArray(VAO);

	//position
	glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cuda::Vector3f) * image_size, nullptr, GL_STATIC_DRAW); //make empty buffer here!
	CheckCuda(cudaGraphicsGLRegisterBuffer(&vertex_position_cuda_resource_, vertex_position_buffer, cudaGraphicsMapFlagsReadOnly)); // register graphics buffer as cuda resource
	CopyDataToCudaGraphicsResource(vertex_position_buffer, vertex_position_cuda_resource_, vertices, image_size);
	glVertexAttribPointer(vertex_location_, 3, GL_FLOAT, GL_FALSE, sizeof(cuda::Vector3f), (void*)0);
	glEnableVertexAttribArray(vertex_location_);


	//color
	if (color_location_ != -1) {
		/*glBindBuffer(GL_ARRAY_BUFFER, vertex_color_buffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(cuda::Vector3f) * image_size, nullptr, GL_STATIC_DRAW);
		CheckCuda(cudaGraphicsGLRegisterBuffer(&vertex_color_cuda_resource_, vertex_color_buffer, cudaGraphicsMapFlagsReadOnly));
		CopyDataToCudaGraphicsResource(vertex_color_buffer, vertex_color_cuda_resource_, colors, image_size);
		glVertexAttribPointer(color_location_, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(color_location_);*/
	}

	

	//normals
	if (normal_location_ != -1) {
	/*	glBindBuffer(GL_ARRAY_BUFFER, normal_buffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(cuda::Vector3f) * image_size, nullptr, GL_STATIC_DRAW);
		CheckCuda(cudaGraphicsGLRegisterBuffer(&normals_cuda_resource_, normal_buffer, cudaGraphicsMapFlagsReadOnly));
		CopyDataToCudaGraphicsResource(normal_buffer, normals_cuda_resource_, normals, image_size);
		glVertexAttribPointer(normal_location_, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(normal_location_);*/
	}

	glBindVertexArray(oldVAO);

	ndrawElements = image_size;
}



void GLGeometry::getPointsandColorfromMesh(vector<Eigen::Vector3f>& points, vector<Eigen::Vector3f>& colors, shared_ptr<geometry::TriangleMesh> mesh, vector<Eigen::Vector3f>* normals){

	points.resize(mesh->vertices_.size());
	colors.resize(mesh->vertex_colors_.size());
	if (normals != 0) {
		normals->resize(mesh->vertex_normals_.size());
		for (int i=0; i< mesh->vertex_normals_.size(); i++) {
		(*normals)[i] = mesh->vertex_normals_[i].cast<float>();
		}

		if (mesh->vertices_.size()  != mesh->vertex_normals_.size()) {
			utility::LogWarning("Normal vector not the same length as vertex vector in mesh. \n");
		}
	}

	if (mesh->vertices_.size() != mesh->vertex_colors_.size() ) {
		utility::LogWarning("Color vector not the same length as vertex vector in mesh . \n");
	}

	for (int i=0; i< mesh->vertices_.size(); i++) {
		points[i] = mesh->vertices_[i].cast<float>();
		colors[i] = mesh->vertex_colors_[i].cast<float>();
	}


}

GLGeometry::~GLGeometry() {
	if (ndrawElements> 0) //if ndrawElements is 0 geometry is already deleted in SetGeometry. Prevents second time deletion. It's not a leak
		deleteGeometry();
}

//copy data from cuda vector to cuda graphics resource. graphics resource is same as opengl buffer
template<typename T>
bool GLGeometry::CopyDataToCudaGraphicsResource(GLuint &opengl_buffer,cudaGraphicsResource_t &cuda_graphics_resource,T *cuda_vector, size_t cuda_vector_size) {
	//as long as resource is mapped opengl must not use the buffer
	CheckCuda(cudaGraphicsMapResources(1, &cuda_graphics_resource));

	/* Copy memory on-chip */
	void *mapped_ptr;
	size_t mapped_size;
	CheckCuda(cudaGraphicsResourceGetMappedPointer(&mapped_ptr, &mapped_size,cuda_graphics_resource));

	const size_t byte_count = cuda_vector_size * sizeof(T);
	assert(byte_count <= mapped_size);
	CheckCuda(cudaMemcpy(mapped_ptr, cuda_vector,byte_count,cudaMemcpyDeviceToDevice));

	CheckCuda(cudaGraphicsUnmapResources(1, &cuda_graphics_resource, nullptr));

	return true;
}

//template<typename T>
//bool GLGeometry::RegisterResource(cudaGraphicsResource_t &cuda_graphics_resource,GLenum opengl_buffer_type,	GLuint &opengl_buffer,	T *cuda_vector,	size_t cuda_vector_size) {
//	glGenBuffers(1, &opengl_buffer);
//	glBindBuffer(opengl_buffer_type, opengl_buffer);
//	glBufferData(opengl_buffer_type, sizeof(T) * cuda_vector_size,	nullptr, GL_STATIC_DRAW);
//	CheckCuda(cudaGraphicsGLRegisterBuffer(&cuda_graphics_resource,	opengl_buffer,	cudaGraphicsMapFlagsReadOnly));
//
//	CopyDataToCudaGraphicsResource(opengl_buffer, cuda_graphics_resource,cuda_vector, cuda_vector_size);
//	return true;
//}

//bool GLGeometry::UnregisterResource(cudaGraphicsResource_t &cuda_graphics_resource,
//	GLuint &opengl_buffer) {
//
//	glDeleteBuffers(1, &opengl_buffer);
//	CheckCuda(cudaGraphicsUnregisterResource(cuda_graphics_resource));
//
//	return true;
//}