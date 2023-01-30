#include "visutil.h"
#include "configvars.h"
#include <opencv2/opencv.hpp>
#include "matrixutil.h"
#include "integrate.h"
#include "core/threadvars.h"


//interal function
shared_ptr<geometry::TriangleMesh> createPathMesh(Model& m) {
	vector<Eigen::Matrix4d> transVec;

	for (int i = 0; i < m.chunks.size(); i++) {
		transVec.push_back(m.chunks[i]->frames[0]->getFrametoWorldTrans());
	}
	auto wholePath = make_shared<geometry::TriangleMesh>();
	for (int i = 0; i < transVec.size(); i++) {
		auto& T = transVec[i];
		auto pathConnector = make_shared<geometry::TriangleMesh>(); //line
		Eigen::Matrix4d lastMatrix = Eigen::Matrix4d::Zero();
		//checks if there are at least 2 camPositions
		if (transVec.size() > 1 && i>0) {
			lastMatrix = transVec[i - 1]; //gets second-to-last Matrix
		}

		//if no 2 camPositions exist there wont be a path added
		if (!lastMatrix.isZero()) {

			Eigen::Vector3d head = T.block<3,1>(0,3);
			Eigen::Vector3d tail = lastMatrix.block<3, 1>(0, 3);

			pathConnector->vertices_.push_back(head); //0
			pathConnector->vertices_.push_back(tail); //1

			pathConnector->triangles_.push_back(Eigen::Vector3i(0, 0, 1));
		}

		//constructs whole camera Path
		*wholePath += *pathConnector;
	}

	//colors Vertices
	for (int j = 0; j < wholePath->vertices_.size(); j++) {
		wholePath->vertex_colors_.push_back(Eigen::Vector3d(1.0, 0.1, 0.1));
	}

	wholePath->ComputeVertexNormals();

	return wholePath;
}

shared_ptr<geometry::TriangleMesh> getSingleCamMesh(const Eigen::Matrix4d& T) {
	auto cameraMesh = make_shared<geometry::TriangleMesh>();
	Eigen::Vector3d origin = T.block<3,1>(0,3);
	Eigen::Vector3d a(-0.05, 0.03, 0.1);
	Eigen::Vector3d b(-0.05, -0.03, 0.1);
	Eigen::Vector3d c(0.05, -0.03, 0.1);
	Eigen::Vector3d d(0.05, 0.03, 0.1);

	//sets vertices to cameraposition
	a = (T * Eigen::Vector4d(a(0), a(1), a(2), 1)).block<3, 1>(0, 0);
	b = (T * Eigen::Vector4d(b(0), b(1), b(2), 1)).block<3, 1>(0, 0);
	c = (T * Eigen::Vector4d(c(0), c(1), c(2), 1)).block<3, 1>(0, 0);
	d = (T * Eigen::Vector4d(d(0), d(1), d(2), 1)).block<3, 1>(0, 0);

	cameraMesh->vertices_.push_back(origin); //0
	cameraMesh->vertices_.push_back(a); //1
	cameraMesh->vertices_.push_back(b); //2
	cameraMesh->vertices_.push_back(c); //3
	cameraMesh->vertices_.push_back(d); //4

										//constructs camera body
	cameraMesh->triangles_.push_back(Eigen::Vector3i(2, 1, 0));
	cameraMesh->triangles_.push_back(Eigen::Vector3i(3, 2, 0));
	cameraMesh->triangles_.push_back(Eigen::Vector3i(4, 3, 0));
	cameraMesh->triangles_.push_back(Eigen::Vector3i(1, 4, 0));

	//generation "lens" of camera / closing camera-Mesh
	cameraMesh->triangles_.push_back(Eigen::Vector3i(1, 2, 4));
	cameraMesh->triangles_.push_back(Eigen::Vector3i(2, 3, 4));
	cameraMesh->triangles_.push_back(Eigen::Vector3i(1, 3, 4)); //just for generation cross on camera for line-drawing
	cameraMesh->triangles_.push_back(Eigen::Vector3i(1, 2, 3)); //just for generation cross on camera for line-drawing

	return cameraMesh;
}
//also colors them
shared_ptr<geometry::TriangleMesh> createCameraMeshes(vector<Eigen::Matrix4d>& transVec) {

	auto wholeMesh = make_shared<geometry::TriangleMesh>();
	auto col_orange = Eigen::Vector3d(1.0, 0.7, 0.35);
	auto col_green = Eigen::Vector3d(0.0, 0.7, 0.35);
	auto col_blue = Eigen::Vector3d(1.0, 1.0, 1.0);

	for (int i = 0; i < transVec.size(); i++) {
		*wholeMesh += *getSingleCamMesh(transVec[i]);
	}

	//colors Vertices
	auto redStep = 0.8 / transVec.size();
	auto greenStep = 0.6 / transVec.size();

	//generates gradient color from first to last camPosition
	for (int i = 0; i < wholeMesh->vertices_.size(); i+=5) { //camera mesh has 5 vertices. One cam has one color
		for (int j = i; j < (i + 5); j++) {
			wholeMesh->vertex_colors_.push_back(col_blue);
		}
		col_blue(0) -= redStep;
		col_blue(1) -= greenStep;
	}

	return wholeMesh;
}

std::shared_ptr<geometry::TriangleMesh> getCameraPathMesh(Model& m) {

	shared_ptr<geometry::TriangleMesh> mesh = make_shared<geometry::TriangleMesh>();
	vector<Eigen::Matrix4d> transformations;

	for (int i = 0; i < m.chunks.size(); i++) {
		transformations.push_back(m.chunks[i]->frames[0]->getFrametoWorldTrans());
	}


	*mesh += *createCameraMeshes(transformations);
	//*mesh += *createPathMesh(transformations);
	mesh->ComputeVertexNormals();

	return mesh;
}



void visThreadFunction(Model* m, std::atomic<bool>& stopVisualizing) {
	cuda::ScalableMeshVolumeCuda mesher(cuda::VertexWithNormalAndColor, 8, 120000);
	//build window
	visualization::VisualizerWithCudaModule meshvis;
	if (!meshvis.CreateVisualizerWindow("Live Mesh", 1280, 720, 100, 100)) {
		utility::LogWarning("Failed creating OpenGL window.\n");
	}
	meshvis.BuildUtilities();
	meshvis.UpdateWindowTitle();

	Eigen::Matrix4d ident = Eigen::Matrix4d::Identity();
	auto cam = getCamera(ident);

	//build and add mesh
	std::shared_ptr<cuda::TriangleMeshCuda> mesh = std::make_shared<cuda::TriangleMeshCuda>();
	meshvis.AddGeometry(mesh);
	meshvis.AddGeometry(getOrigin());
	meshvis.AddGeometry(cam);


	camera::PinholeCameraParameters paras;
	Eigen::Matrix4d tmp;
	tmp<< 1, 0,0,0 
		,0,1,0, 0
		,0,0,1,-2
		,0,0,0,1;
	paras.extrinsic_ = tmp.inverse();
	paras.intrinsic_ =g_intrinsic;
	tmp = tmp.inverse();
	meshvis.GetViewControl().setCameraPos(tmp);
	meshvis.GetViewControl().SetConstantZFar(15.0);


	while (m->tsdf_cuda.active_subvolume_entry_array_.size() == 0) {
		std::this_thread::sleep_for(20ms); //sleep since no task is necessary
	}
	*mesh = mesher.mesh();

	while (!stopVisualizing) {
		if (m->tsdf_cuda.unmeshed_data){
			//update mesh and take user inputs
			pandia_integration::tsdfLock.lock();
			mesher.MarchingCubes(m->tsdf_cuda,true);
			currentposlock.lock();
			*cam = *getCamera(m->currentPos); 
			currentposlock.unlock();
			pandia_integration::tsdfLock.unlock();
			//meshvis.GetViewControl().setCameraPos(m->chunks.back()->frames[0]->frametoworldtrans.inverse());
			meshvis.PollEvents(); //this takes long (5-20ms) probably blocks a lot!
			meshvis.UpdateGeometry();
			//meshvis.GetViewControl().SetViewMatrices(f->frametoworldtrans.inverse());
		} else {// todo dont know if this or thread sleep
			meshvis.PollEvents(); //this takes long (5-20ms) probably blocks a lot!
			meshvis.UpdateGeometry();
		}
	}

	while (meshvis.WaitEvents()) {
		meshvis.PollEvents(); //this takes long (5-20ms) probably blocks a lot!
		meshvis.UpdateGeometry();
	}
	meshvis.DestroyVisualizerWindow();
	utility::LogInfo("Stopping Mesh Visualization \n");

}


std::shared_ptr<geometry::TriangleMesh> getSpherePointer(Eigen::Vector3d v) {
	double radius = 0.1;
	int resolution = 20;
	auto mesh_ptr = std::make_shared<geometry::TriangleMesh>();
	if (radius <= 0.0 || resolution <= 0) {
		return mesh_ptr;
	}
	mesh_ptr->vertices_.resize(2 * resolution * (resolution - 1) + 2);
	mesh_ptr->vertices_[0] = Eigen::Vector3d(v(0), v(1), v(2) + radius);
	mesh_ptr->vertices_[1] = Eigen::Vector3d(v(0), v(1), v(2) - radius);
	double step = 3.14159265359 / (double)resolution;
	for (int i = 1; i < resolution; i++) {
		double alpha = step * i;
		int base = 2 + 2 * resolution * (i - 1);
		for (int j = 0; j < 2 * resolution; j++) {
			double theta = step * j;
			mesh_ptr->vertices_[base + j] =
				Eigen::Vector3d(sin(alpha) * cos(theta) * radius + v(0), sin(alpha) * sin(theta) * radius + v(1), cos(alpha) * radius + v(2));
		}
	}
	for (int j = 0; j < 2 * resolution; j++) {
		int j1 = (j + 1) % (2 * resolution);
		int base = 2;
		mesh_ptr->triangles_.push_back(Eigen::Vector3i(0, base + j, base + j1));
		base = 2 + 2 * resolution * (resolution - 2);
		mesh_ptr->triangles_.push_back(Eigen::Vector3i(1, base + j1, base + j));
	}
	for (int i = 1; i < resolution - 1; i++) {
		int base1 = 2 + 2 * resolution * (i - 1);
		int base2 = base1 + 2 * resolution;
		for (int j = 0; j < 2 * resolution; j++) {
			int j1 = (j + 1) % (2 * resolution);
			mesh_ptr->triangles_.push_back(
				Eigen::Vector3i(base2 + j, base1 + j1, base1 + j));
			mesh_ptr->triangles_.push_back(
				Eigen::Vector3i(base2 + j, base2 + j1, base1 + j1));
		}
	}
	return mesh_ptr;
}



void visualizeChunk(Chunk& c, bool wordlcoords)
{
	auto f = c.frames;
	std::vector<std::shared_ptr<const geometry::Geometry>> tmp;
	// add frames
	for (int i = 0; i < f.size(); i++) {
		auto pcd = geometry::PointCloud::CreateFromRGBDImage(*(f[i]->rgbd), g_intrinsic);
		if (wordlcoords) {
			pcd->Transform(f[i]->getFrametoWorldTrans());
		}
		tmp.push_back(pcd);

	}

	tmp.push_back(getOrigin());
	visualization::DrawGeometries(tmp);
}

void visualizetsdfModel(Model& m) {
	std::vector<std::shared_ptr<const geometry::Geometry>> tmp;
	tmp.push_back(m.tsdf->ExtractTriangleMesh());
	visualization::DrawGeometries(tmp);
}


void drawEigen(std::vector<Eigen::Matrix3Xd> vvec)
{
	std::vector<std::shared_ptr<const geometry::Geometry>> tmp;
	auto pcd = std::make_shared<geometry::PointCloud>();
	int count = 0;
	for (auto v : vvec) {
		for (int i = 0; i < v.cols(); i++) {
			Eigen::Vector3d newpoint(v(0, i), v(1, i), v(2, i));
			pcd->points_.push_back(newpoint);
			if (count == 0) {
				pcd->colors_.push_back(Eigen::Vector3d(1.0, 0.0, 0.0)); //red
			} else {
				pcd->colors_.push_back(Eigen::Vector3d(0.0, 0.0, 1.0)); //blue
			}
		}
		tmp.push_back(pcd);
		count++;
	}
	tmp.push_back(getOrigin());
	visualization::DrawGeometries(tmp);

}


void drawPointVector(std::vector<Eigen::Vector3d> v) {
	std::vector<std::shared_ptr<const geometry::Geometry>> tmp;
	for (int i = 0; i < v.size(); i++) {
		auto mesh_ptr = geometry::TriangleMesh::CreateSphere(0.05, 10);
		mesh_ptr->Transform(gettrans(v[i]));
		mesh_ptr->PaintUniformColor(Eigen::Vector3d(255, 0, 0));
		tmp.push_back(mesh_ptr);
	}
	tmp.push_back(getOrigin());
	visualization::DrawGeometries(tmp);

}

void visualizeHomogeniousVector(vector<Eigen::Vector4d>& v) {
	std::vector<std::shared_ptr<const geometry::Geometry>> vis;
	geometry::PointCloud pcd;
	for (int i = 0; i < v.size(); i++) {
		pcd.points_.push_back(v[i].block<3, 1>(0, 0));
	}
	vis.push_back(std::make_shared<geometry::PointCloud>(pcd));
	visualization::DrawGeometries(vis);
}


void visualizecustomskeypoints(std::vector<c_keypoint>& kps)
{
	std::vector<std::shared_ptr<const geometry::Geometry>> tmp;
	geometry::PointCloud pcd;
	for (int i = 0; i < kps.size(); i++) {
		pcd.points_.push_back(kps[i].p.block<3, 1>(0, 0));
	}
	tmp.push_back(std::make_shared<geometry::PointCloud>(pcd));
	visualization::DrawGeometries(tmp);

}


//draws all pcds with keypoints and lines between them
void visualizecurrentMatches(shared_ptr<Chunk>c)
{
	double r = 2 * 3.14159265359 / (double)(c->frames.size() + 1);
	std::vector<std::shared_ptr<const geometry::Geometry>> vis;
	for (int i = 0; i < c->frames.size(); i++) {
		auto tmp = *geometry::PointCloud::CreateFromRGBDImage(*(c->frames[i]->rgbd), g_intrinsic);
		Eigen::Matrix4d tmp2 = getRy(r * i) * gettrans(Eigen::Vector3d(0, 0, 1.5));
		tmp.Transform(tmp2);
		vis.push_back(make_shared<geometry::PointCloud>(tmp));
	}
	std::shared_ptr<geometry::LineSet> ls = std::make_shared<geometry::LineSet>();
	for (int i = 0; i < c->frames.size() - 1; i++) {
		for (int j = i + 1; j < c->frames.size(); j++) {
			auto& matches = c->filteredmatches(i, j);
			for (int k = 0; k < matches.size(); k++) {
				auto mesh_ptr = geometry::TriangleMesh::CreateSphere(0.02, 10);
				Eigen::Matrix4d t = getRy(r * i) * gettrans(Eigen::Vector3d(0, 0, 1.5)) * gettrans(matches[k].p1);
				mesh_ptr->Transform(t);
				mesh_ptr->PaintUniformColor(Eigen::Vector3d(1, 0, 0));
				vis.push_back(mesh_ptr);
				auto mesh_ptr2 = geometry::TriangleMesh::CreateSphere(0.02, 10);
				Eigen::Matrix4d t2 = getRy(r * j) * gettrans(Eigen::Vector3d(0, 0, 1.5)) * gettrans(matches[k].p2);
				mesh_ptr2->Transform(t2);
				mesh_ptr2->PaintUniformColor(Eigen::Vector3d(1, 0, 0));
				vis.push_back(mesh_ptr2);
				ls->points_.push_back((t2 * Eigen::Vector4d(0, 0, 0, 1)).block<3, 1>(0, 0));
				ls->points_.push_back((t * Eigen::Vector4d(0, 0, 0, 1)).block<3, 1>(0, 0));
				ls->lines_.push_back(Eigen::Vector2i(ls->points_.size() - 1, ls->points_.size() - 2));
			}
		}
	}
	vis.push_back(ls);
	vis.push_back(getOrigin());
	visualization::DrawGeometries(vis);

}

//draws all pcds with keypoints and lines between them
//if raw is set to true the unfiltered matches are plotted
void visualizecurrentMatches(Model & m, bool raw /* =false */)
{
	//pcds
	std::vector<std::shared_ptr<const geometry::Geometry>> vis;
	for (int j = 0; j < m.chunks.size(); j++) {
		if (m.chunks[j]->frames.size() == 11) {
			auto& c = m.chunks[j];
			geometry::PointCloud tmppcd;
			for (int i = 0; i < c->frames.size(); i++) {
				auto tmp = *geometry::PointCloud::CreateFromRGBDImage(*(c->frames[i]->rgbd), g_intrinsic);
				tmp.Transform(c->frames[i]->chunktransform);
				tmppcd = tmppcd + tmp;
			}
			tmppcd.VoxelDownSample(0.005);
			Eigen::Matrix4d tmp2 = getRy(0.5 * j) * gettrans(Eigen::Vector3d(0, 0, 1.5));
			tmppcd.Transform(tmp2);
			vis.push_back(make_shared<geometry::PointCloud>(tmppcd));
		}
	}
	//invalids
	cout << " blubber \n ";
	for (int j = 0; j < m.invalidChunks.size(); j++) {
		if (m.chunks[j]->frames.size() == 11) {
			auto& c = m.invalidChunks[j];
			geometry::PointCloud tmppcd;
			for (int i = 0; i < c->frames.size(); i++) {
				auto tmp = *geometry::PointCloud::CreateFromRGBDImage(*(c->frames[i]->rgbd), g_intrinsic);
				tmp.Transform(c->frames[i]->chunktransform);
				tmppcd = tmppcd + tmp;
			}
			tmppcd.VoxelDownSample(0.005);
			Eigen::Matrix4d tmp2 = getRy(0.5 * (j + m.chunks.size())) * gettrans(Eigen::Vector3d(0, 0, 1.5));
			tmppcd.Transform(tmp2);
			vis.push_back(make_shared<geometry::PointCloud>(tmppcd));
		}
	}

	std::shared_ptr<geometry::LineSet> ls = std::make_shared<geometry::LineSet>();
	int n = m.chunks.size();
	if (raw) {
		n += m.invalidChunks.size();
	}
	for (int i = 0; i < n - 1; i++) {
		for (int j = i + 1; j < n; j++) {
			auto& matches = m.filteredmatches(i, j);
			if (raw) {
				matches = m.rawmatches(i, j);
			}
			for (int k = 0; k < matches.size(); k++) {
				auto mesh_ptr = geometry::TriangleMesh::CreateSphere(0.02, 10);
				Eigen::Matrix4d t = getRy(0.5 * i) * gettrans(Eigen::Vector3d(0, 0, 1.5)) * gettrans(matches[k].p1);
				mesh_ptr->Transform(t);
				mesh_ptr->PaintUniformColor(Eigen::Vector3d(1, 0, 0));
				vis.push_back(mesh_ptr);
				auto mesh_ptr2 = geometry::TriangleMesh::CreateSphere(0.02, 10);
				Eigen::Matrix4d t2 = getRy(0.5 * j) * gettrans(Eigen::Vector3d(0, 0, 1.5)) * gettrans(matches[k].p2);
				mesh_ptr2->Transform(t2);
				mesh_ptr2->PaintUniformColor(Eigen::Vector3d(1, 0, 0));
				vis.push_back(mesh_ptr2);
				ls->points_.push_back((t2 * Eigen::Vector4d(0, 0, 0, 1)).block<3, 1>(0, 0));
				ls->points_.push_back((t * Eigen::Vector4d(0, 0, 0, 1)).block<3, 1>(0, 0));
				ls->lines_.push_back(Eigen::Vector2i(ls->points_.size() - 1, ls->points_.size() - 2));
			}
		}
	}
	vis.push_back(ls);
	vis.push_back(getOrigin());
	visualization::DrawGeometries(vis);

}



std::shared_ptr<geometry::LineSet> getCamera(Eigen::Matrix4d& t,Eigen::Vector3d color) {
	auto ls = std::make_shared<geometry::LineSet>();
	Eigen::Vector3d zero(0.0, 0.0, 0.0);
	Eigen::Vector3d a(-0.05, 0.03, 0.1);
	Eigen::Vector3d b(-0.05, -0.03, 0.1);
	Eigen::Vector3d c(0.05, -0.03, 0.1);
	Eigen::Vector3d d(0.05, 0.03, 0.1);
	glEnable(GL_LINE_WIDTH);
	glLineWidth(100.f);
	zero = (t * Eigen::Vector4d(zero(0), zero(1), zero(2), 1)).block<3, 1>(0, 0);
	a = (t * Eigen::Vector4d(a(0), a(1), a(2), 1)).block<3, 1>(0, 0);
	b = (t * Eigen::Vector4d(b(0), b(1), b(2), 1)).block<3, 1>(0, 0);
	c = (t * Eigen::Vector4d(c(0), c(1), c(2), 1)).block<3, 1>(0, 0);
	d = (t * Eigen::Vector4d(d(0), d(1), d(2), 1)).block<3, 1>(0, 0);

	ls->points_.push_back(zero);
	ls->points_.push_back(a);
	ls->points_.push_back(b);
	ls->points_.push_back(c);
	ls->points_.push_back(d);

	ls->lines_.push_back(Eigen::Vector2i(0, 1));
	ls->lines_.push_back(Eigen::Vector2i(0, 2));
	ls->lines_.push_back(Eigen::Vector2i(0, 3));
	ls->lines_.push_back(Eigen::Vector2i(0, 4));

	ls->lines_.push_back(Eigen::Vector2i(1, 2));
	ls->lines_.push_back(Eigen::Vector2i(1, 4));
	ls->lines_.push_back(Eigen::Vector2i(3, 2));
	ls->lines_.push_back(Eigen::Vector2i(3, 4));

	for (int i = 0; i < 8; i++) {
		ls->colors_.push_back(color);
	}


	return ls;
}
std::shared_ptr<geometry::LineSet> getCameraPath(std::vector<Eigen::Matrix4d>& v, Eigen::Vector3d color) {


	auto ls = std::make_shared<geometry::LineSet>();
	for (int i = 0; i < v.size(); i++) {
		*ls += *getCamera(v[i],color);
	}
	Eigen::Vector3d red(1.0,0.0,0.0);
	for (int i = 0; i < v.size() - 1; i++) {
		ls->lines_.push_back(Eigen::Vector2i(5 * i, 5 * (i + 1)));
		ls->colors_.push_back(red);
	}
	return ls;
}



std::shared_ptr<geometry::LineSet> getOrigin() {
	auto ls = std::make_shared<geometry::LineSet>();
	Eigen::Vector3d zero(0.0, 0.0, 0.0);
	Eigen::Vector3d x(1.0, 0.0, 0.0);
	Eigen::Vector3d y(0.0, 1.0, 0.0);
	Eigen::Vector3d z(0.0, 0.0, 1.0);

	Eigen::Vector3d xcolor(1.0, 0.0, 0.0);
	Eigen::Vector3d ycolor(0.0, 1.0, 0.0);
	Eigen::Vector3d zcolor(0.0, 0.0, 1.0);

	ls->points_.push_back(zero);
	ls->points_.push_back(x);
	ls->points_.push_back(y);
	ls->points_.push_back(z);
	ls->lines_.push_back(Eigen::Vector2i(0, 1));
	ls->lines_.push_back(Eigen::Vector2i(0, 2));
	ls->lines_.push_back(Eigen::Vector2i(0, 3));
	ls->colors_.push_back(xcolor);
	ls->colors_.push_back(ycolor);
	ls->colors_.push_back(zcolor);

	return ls;
}


std::shared_ptr<geometry::LineSet> getFrustumLineSet(Frustum& fr,const Eigen::Vector3d& color) {
	auto ls = std::make_shared<geometry::LineSet>();

	for (auto& p : fr.corners) {
		ls->points_.push_back(p);
	}

	//cc again starting with back half
	ls->lines_.push_back(Eigen::Vector2i(0, 1));
	ls->lines_.push_back(Eigen::Vector2i(1, 2));
	ls->lines_.push_back(Eigen::Vector2i(2, 3));
	ls->lines_.push_back(Eigen::Vector2i(3, 0));

	//sides
	ls->lines_.push_back(Eigen::Vector2i(0, 4));
	ls->lines_.push_back(Eigen::Vector2i(1, 5));
	ls->lines_.push_back(Eigen::Vector2i(2, 6));
	ls->lines_.push_back(Eigen::Vector2i(3, 7));

	//front 
	ls->lines_.push_back(Eigen::Vector2i(4, 5));
	ls->lines_.push_back(Eigen::Vector2i(5, 6));
	ls->lines_.push_back(Eigen::Vector2i(6, 7));
	ls->lines_.push_back(Eigen::Vector2i(7, 4));


	for (int i = 0; i < 12; i++) {
		ls->colors_.push_back(color);
	}


	return ls;
}


std::shared_ptr<geometry::LineSet> getvisFrusti(Model& m, Frustum& corefrustum) {
	auto ls = std::make_shared<geometry::LineSet>();
	vector<Frustum> frusti;
	for (auto& c : m.chunks) {
		frusti.push_back(*c->frustum);
	}
	*ls+= *getFrustumLineSet(corefrustum,Eigen::Vector3d(0,0,1));//blue

	for (int i = 0; i < frusti.size(); i++) {
		//if (corefrustum.intersect(frusti[i])) {
		if (corefrustum.inLocalGroup(frusti[i])) {
			*ls+= *getFrustumLineSet(frusti[i],Eigen::Vector3d(0,1,0)); //green
		} else {
			*ls+= *getFrustumLineSet(frusti[i],Eigen::Vector3d(1,0,0)); //red
			cout << "chunk " << i << " not in local group vis\n";

		}

	}

	return ls;
}


//interal function
shared_ptr<geometry::TriangleMesh> createCameraMesh(vector<Eigen::Matrix4d>& transVec) {

	auto wholeMesh = make_shared<geometry::TriangleMesh>();
	auto col_orange = Eigen::Vector3d(1.0, 0.7, 0.35);
	auto col_green = Eigen::Vector3d(0.0, 0.7, 0.35);
	auto col_blue = Eigen::Vector3d(1.0, 1.0, 1.0);

	for (int i = 0; i < transVec.size(); i++) {
		
		auto t = transVec[i];
		auto cameraMesh = make_shared<geometry::TriangleMesh>();
		Eigen::Vector3d zero(0.0, 0.0, 0.0);
		Eigen::Vector3d a(-0.05, 0.03, 0.1);
		Eigen::Vector3d b(-0.05, -0.03, 0.1);
		Eigen::Vector3d c(0.05, -0.03, 0.1);
		Eigen::Vector3d d(0.05, 0.03, 0.1);

		//sets vertices to cameraposition
		zero = (t * Eigen::Vector4d(zero(0), zero(1), zero(2), 1)).block<3, 1>(0, 0);
		a = (t * Eigen::Vector4d(a(0), a(1), a(2), 1)).block<3, 1>(0, 0);
		b = (t * Eigen::Vector4d(b(0), b(1), b(2), 1)).block<3, 1>(0, 0);
		c = (t * Eigen::Vector4d(c(0), c(1), c(2), 1)).block<3, 1>(0, 0);
		d = (t * Eigen::Vector4d(d(0), d(1), d(2), 1)).block<3, 1>(0, 0);

		cameraMesh->vertices_.push_back(zero); //0
		cameraMesh->vertices_.push_back(a); //1
		cameraMesh->vertices_.push_back(b); //2
		cameraMesh->vertices_.push_back(c); //3
		cameraMesh->vertices_.push_back(d); //4

		//constructs camera body
		cameraMesh->triangles_.push_back(Eigen::Vector3i(2, 1, 0));
		cameraMesh->triangles_.push_back(Eigen::Vector3i(3, 2, 0));
		cameraMesh->triangles_.push_back(Eigen::Vector3i(4, 3, 0));
		cameraMesh->triangles_.push_back(Eigen::Vector3i(1, 4, 0));

		//generation "lens" of camera / closing camera-Mesh
		cameraMesh->triangles_.push_back(Eigen::Vector3i(1, 2, 4));
		cameraMesh->triangles_.push_back(Eigen::Vector3i(2, 3, 4));
		cameraMesh->triangles_.push_back(Eigen::Vector3i(1, 3, 4)); //just for generation cross on camera for line-drawing
		cameraMesh->triangles_.push_back(Eigen::Vector3i(1, 2, 3)); //just for generation cross on camera for line-drawing

		*wholeMesh += *cameraMesh;
	}

	
	//colors Vertices
	auto redStep = 0.8 / transVec.size();
	auto greenStep = 0.6 / transVec.size();

	//generates gradient color from first to last camPosition
	for (int i = 0; i < wholeMesh->vertices_.size(); i+=5) {

		for (int j = i; j < (i + 5); j++) {
			wholeMesh->vertex_colors_.push_back(col_blue);
		}
		
		col_blue(0) -= redStep;
		col_blue(1) -= greenStep;
	}

	//first camera Position
	for (int i = 0; i < 5; i++) {
		//wholeMesh->vertex_colors_[i] = col_green;
		wholeMesh->vertex_colors_[i] = Eigen::Vector3d(0.0, 0.0, 0.0);
	}

	//last camera Position
	auto lastCamVertices = wholeMesh->vertex_colors_.size() - 5;
	for (int i = lastCamVertices; i < wholeMesh->vertex_colors_.size(); i++) {
		//wholeMesh->vertex_colors_[i] = col_orange;
		wholeMesh->vertex_colors_[i] = Eigen::Vector3d(0.0, 0.0, 0.0);

	}


	wholeMesh->ComputeTriangleNormals();
	
	return wholeMesh;
}


//interal function
shared_ptr<geometry::TriangleMesh> createPathMesh(vector<Eigen::Matrix4d>& transVec) {

	auto wholePath = make_shared<geometry::TriangleMesh>();

	for (int i = 0; i < transVec.size(); i++) {
		auto t = transVec[i];
		
		auto pathConnector = make_shared<geometry::TriangleMesh>();
		Eigen::Matrix4d lastMatrix = Eigen::Matrix4d::Zero();

		//checks if there are at least 2 camPositions
		if (transVec.size() > 1 && i>0) {
			lastMatrix = transVec[i - 1]; //gets second-to-last Matrix
		}
		
		//if no 2 camPositions exist there wont be a path added
		if (!lastMatrix.isZero()) {

			Eigen::Vector3d head(t(0, 3), t(1, 3), t(2, 3));
			Eigen::Vector3d tail(lastMatrix.block<3, 1>(0, 3));

			pathConnector->vertices_.push_back(head); //0
			pathConnector->vertices_.push_back(tail); //1

			pathConnector->triangles_.push_back(Eigen::Vector3i(0, 0, 1));
		}

		//constructs whole camera Path
		*wholePath += *pathConnector;
	}
	
	//colors Vertices
	for (int j = 0; j < wholePath->vertices_.size(); j++) {
		wholePath->vertex_colors_.push_back(Eigen::Vector3d(1.0, 0.1, 0.1));
	}

	return wholePath;
}


std::shared_ptr<geometry::TriangleMesh> getCameraPathMesh(Model& m, int& divider) {

	shared_ptr<geometry::TriangleMesh> mesh = make_shared<geometry::TriangleMesh>();


	vector<Eigen::Matrix4d> transformations;

	for (int i = 0; i < m.chunks.size(); i++) {
		for (int j = 0; j < 10; j++) {


			if (j == 0) { //first frame of every chunk will be drawn as cameraPathMesh

				auto& matrix = m.chunks[i]->frames[j]->getFrametoWorldTrans();
				transformations.push_back(matrix);
			}
		}
	}

	
	*mesh += *createCameraMesh(transformations);
	divider = mesh->vertices_.size();
	*mesh += *createPathMesh(transformations);

	mesh->ComputeVertexNormals();


	return mesh;
}



shared_ptr<geometry::TriangleMesh> createCameraScaffolding(shared_ptr<GLGeometry>& cameraPath, int& divider) {

	auto& mesh = cameraPath->mesh_;

	for (int i = 0; i < divider; i++) {
		mesh->vertex_colors_[i] = Eigen::Vector3d(0.0, 0.0, 0.9);
	}

	for (int i = divider; i < mesh->vertices_.size(); i++) {
		mesh->vertex_colors_[i] = Eigen::Vector3d(0.9, 0.1, 0.1);
	}


	return mesh;

}