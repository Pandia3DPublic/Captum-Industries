#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING

#include "coreutil.h"
#include "configvars.h"
#include "imageutil.h"]
#include "matrixutil.h"
#include <experimental/filesystem>
#include <fstream>
#include <direct.h> //mkdir
#include <list>
#include "genKps.h"

//pen3d::utility::filesystem::remove
//std::experimental::filesystem::remove


//without haeder, internal functions
void loadScanNNCameraPath(const std::string& path, std::vector<Eigen::Matrix4d_u>& out);
Eigen::Matrix4d_u txttoPose2(const std::string& filename);
Eigen::Matrix3d loadScanNNIntrinsic(const std::string& filename);



//####################################################with header from here#####################################################################################################################

void prepareDatapath(std::string& s) {
	for (int i = 0; i < s.length(); i++) {
		if (s[i] == '/') {
			s.insert(i + 1, "/");
			i++;
		}
	}
}


camera::PinholeCameraIntrinsic getLowIntr(camera::PinholeCameraIntrinsic intrinsic) {
	auto& intr = intrinsic.intrinsic_matrix_;
	double xfactor = (double)g_lowx / intrinsic.width_; // xres
	double yfactor = (double)g_lowy / intrinsic.height_; //yres
	intr(0, 0) *= xfactor;
	intr(0, 2) *= xfactor;
	intr(1, 2) *= yfactor;
	intr(1, 1) *= yfactor;
	intrinsic.width_ = g_lowx;
	intrinsic.height_ = g_lowy;
	return intrinsic;

}

camera::PinholeCameraIntrinsic getScaledIntr(camera::PinholeCameraIntrinsic intrinsic, int width, int height) {
	auto& intr = intrinsic.intrinsic_matrix_;
	double xfactor = (double)width / intrinsic.width_; // xres
	double yfactor = (double)height / intrinsic.height_; //yres
	intr(0, 0) *= xfactor;
	intr(0, 2) *= xfactor;
	intr(1, 2) *= yfactor;
	intr(1, 1) *= yfactor;
	intrinsic.width_ = width;
	intrinsic.height_ = height;
	return intrinsic;

}

//for hardrive data
std::shared_ptr<Frame> getSingleFrame(std::string path, int nstart)
{

	std::string filepath_rgb(path + "\\color\\");
	std::string filepath_depth(path + "\\depth\\");

	std::shared_ptr<geometry::Image> rgb_image = make_shared<geometry::Image>();
	std::shared_ptr<geometry::Image> depth_image= make_shared<geometry::Image>();

	auto tmp = make_shared<Frame>();

	//checks if filepath leads to scenes
	if (g_readimagePath.find("scene00") != string::npos) {//filepath is scene
		//read scene
		string filename_scene_depth = filepath_depth + to_string(nstart) + ".png";
		string filename_scene_rgb = filepath_rgb + to_string(nstart) + ".jpg";

		io::ReadImage(filename_scene_rgb, *rgb_image);
		tmp->rgbPath = filename_scene_rgb;
		io::ReadImage(filename_scene_depth, *depth_image);
		tmp->depthPath = filename_scene_depth;
		rgb_image = resizeImage(rgb_image, depth_image->width_, depth_image->height_, "nointerpol");
	}
	else {//filepath is custom data
		// read images
		std::string filename_rgb_jpg = filepath_rgb + getPicNumberString(nstart) + ".jpg";
		std::string filename_rgb_png = filepath_rgb + getPicNumberString(nstart) + ".png";
		//checks if file is .jpg or .png
		auto tmpLevel = utility::GetVerbosityLevel();
		utility::SetVerbosityLevel(utility::VerbosityLevel::Error);
		if (io::ReadImage(filename_rgb_jpg, *rgb_image)) {
			tmp->rgbPath = filename_rgb_jpg;
		} else {
			io::ReadImage(filename_rgb_png, *rgb_image);
			tmp->rgbPath = filename_rgb_png;
		}
		utility::SetVerbosityLevel(tmpLevel);
		std::string filename_depth_c = filepath_depth + getPicNumberString(nstart) + ".png";
		io::ReadImage(filename_depth_c, *depth_image);
		tmp->depthPath = filename_depth_c;
	}


	generateFrame(rgb_image, depth_image,tmp);
	cv::Mat cvColor;
	topencv(rgb_image,cvColor);//todo replace with wrapper funtion of opencv, prevent copy
	generateOrbKeypoints(tmp, cvColor);
	return tmp;

}


void setDefaultIntrinsic() {
	g_intrinsic = camera::PinholeCameraIntrinsic(camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);
	g_intrinsic_cuda = open3d::cuda::PinholeCameraIntrinsicCuda(g_intrinsic);
	g_lowIntr = getLowIntr(g_intrinsic);
}


void checkRecursive(const shared_ptr<Chunk>& c, int indf, vector<int>& indices) { //indeces contain all traversed nodes
	for (int i = 0; i < c->frames.size(); i++) {
		if (indf != i) {
			auto t = std::find(indices.begin(), indices.end(), i);
			if (t == indices.end() && c->filteredmatches(i, indf).size() != 0) { // not found
				indices.push_back(i);
				checkRecursive(c, i, indices);
			}
		}
	}
}

//check if the chunk contains frames that are conected with each other, but not with all frames through any number of connections
//check if you can reach any node an the graph from any other node
//todo test this rigorously
bool checkValid(const shared_ptr<Chunk> c, vector<int>& removeIndices)
{
	vector<int> indeces;
	indeces.reserve(c->frames.size());
	int a = 0;
	indeces.push_back(a);
	checkRecursive(c, a, indeces);
	bool validchunk = true;
	if (indeces.size() < c->frames.size()) {
		validchunk = false;
		if (indeces.size() < c->frames.size() / 2) { //a large unit is missing
			removeIndices = indeces;
		} else { // a small unit is missing but everything else in removeindex
			for (int i = 0; i < c->frames.size(); i++) { //frame 0 was in the small unit
				auto t = std::find(indeces.begin(), indeces.end(), i);
				if (t == indeces.end()) {
					removeIndices.push_back(i);
				}
			}
		}
		//delete all filteredmatches

	}
	//cout << "new indices " << endl;
	//for (int i = 0; i < removeIndices.size(); i++) {
	//	cout << removeIndices[i];
	//}
	return validchunk;
}

//only check the newest chunk here
bool checkValid(Model& m)
{
	bool validChunkInModel = false;
	auto index = m.chunks.size() - 1;
	for (int k = 0; k < m.chunks.size(); k++) {
		if (m.filteredmatches(index, k).size() != 0) {
			validChunkInModel = true;
		}
	}
	if (!validChunkInModel) {
		utility::LogWarning("Chunk number {} cannot be fitted in model \n", index + 1);
	}
	return validChunkInModel;
}

bool getBit(const unsigned char& a, const int& n) {
	return a >> n & 0x1;
}
void setBitOne(unsigned char& a, const int& n) {
	a |= 1UL << n;
}
//todo this taeks forever (6-15ms)
//todo get rid of rgbd and pcd during core algorithm. Note hat rgbd generates a float depth images, which is necessary for 
//tsdf integration. Note also that gcf needs pcds.
//todo put normal calculation and downsizing on graphics card.
//This generates a frame form open3d images. Does not set intrinsics or opencv data. Assumes that images have same size
void generateFrame(shared_ptr<open3d::geometry::Image> color, shared_ptr<open3d::geometry::Image> depth, shared_ptr<Frame> out)
{

	auto rgbd = geometry::RGBDImage::CreateFromColorAndDepth(*color, *depth, 1000.0, g_cutoff, false); //done made unecessary
	//Timer t3("rest");
	out->rgbd = rgbd;
	out->chunktransform = getIdentity();

	//get smaller images for costly calculations 
	out->rgblow = resizeImage(color, g_lowx, g_lowy, "nointerpol"); //todo do bilinear interpolate
	out->depthlow = resizeImage(depth, g_lowx, g_lowy, "nointerpol");
	auto rgbdlow = geometry::RGBDImage::CreateFromColorAndDepth(*out->rgblow, *out->depthlow, 1000.0, g_cutoff, false);
	out->lowpcd = geometry::PointCloud::CreateFromRGBDImage(*rgbdlow, g_lowIntr);

	if (out->lowpcd->HasNormals() == false) {
		out->lowpcd->EstimateNormals();
	}
	out->lowpcd->NormalizeNormals();
	out->lowpcd->OrientNormalsTowardsCameraLocation();
	//visualization::DrawGeometries({out->rgbd});

}

//assigns intrinsic-data from .txt-File
void assignIntrinsic(string value, int lineCounter)
{
	int row = lineCounter -2;
	if (lineCounter == 0) g_intrinsic.width_ = stoi(value);
	if (lineCounter == 1) g_intrinsic.height_ = stoi(value);
	if (lineCounter > 1) {
		istringstream iss(value);
		for (int column = 0; column < 3; column++) {
			double matrixValue;
			iss >> matrixValue;
			g_intrinsic.intrinsic_matrix_(row,column) = matrixValue;
		}
	}
}
//checks if file exists
bool fileExists(string& filename)
{
	ifstream file;
	file.open(filename);
	if (file){
		return true;
	} else {
		return false;
	}
}
//reads intrinsic-data from .txt-File
void setFromIntrinsicFile(string& filepath)
{
	string stringline;
	ifstream cameraIntrinsic(filepath);
	int linecounter = 0;
	int row = 0;
	if (cameraIntrinsic.is_open()) {
		while (getline(cameraIntrinsic, stringline))
		{
			assignIntrinsic(stringline, linecounter);
			linecounter++;
		}
		g_intrinsic_cuda = open3d::cuda::PinholeCameraIntrinsicCuda(g_intrinsic);
		g_lowIntr = getLowIntr(g_intrinsic);
	}
	cameraIntrinsic.close();
}



int CountValidDepthPixelsLocal(shared_ptr<geometry::Image> depth, int stride) {
	int num_valid_pixels = 0;
	for (int i = 0; i < depth->height_; i += stride) {
		for (int j = 0; j < depth->width_; j += stride) {
			const uint16_t* p = depth->PointerAt<uint16_t>(j, i);
			if (*p > 0) num_valid_pixels += 1;
		}
	}
	return num_valid_pixels;
}

shared_ptr < geometry::PointCloud> genpcd(shared_ptr<open3d::geometry::Image> color, shared_ptr<open3d::geometry::Image> depth) {

	auto pointcloud = std::make_shared<geometry::PointCloud>();
	auto focal_length = g_intrinsic.GetFocalLength();
	auto principal_point = g_intrinsic.GetPrincipalPoint();
	int num_valid_pixels = CountValidDepthPixelsLocal(depth, 1);
	pointcloud->points_.resize(num_valid_pixels);
	pointcloud->colors_.resize(num_valid_pixels);
	pointcloud->indeces.resize(depth->width_, depth->height_);
	//initialize with -1 in case no corresponding points exist
	for (int i = 0; i < depth->width_; i++) {
		for (int j = 0; j < depth->height_; j++) {
			pointcloud->indeces(i, j) = -1;
		}
	}
	int cnt = 0;
	for (int i = 0; i < depth->height_; i++) {
		uint16_t* p = (uint16_t*)(depth->data_.data() +
			i * depth->BytesPerLine());
		uint8_t * pc = (uint8_t*)(color->data_.data() +
			i * color->BytesPerLine());
		for (int j = 0; j < depth->width_; j++, p++, pc += 3) {
			if (*p > 0) {
				double z = (double)(*p);
				double x = (j - principal_point.first) * z / focal_length.first;
				double y = (i - principal_point.second) * z / focal_length.second;
				pointcloud->points_[cnt] = Eigen::Vector3d(x, y, z); //todo probably need to divide by 1000 here and remove the same in genkps
				pointcloud->colors_[cnt++] = Eigen::Vector3d(pc[0], pc[1], pc[2]);
				pointcloud->indeces(j, i) = cnt - 1;
			}
		}
	}

	return pointcloud;

	//auto out = make_shared<geometry::PointCloud>();
	////reserve indeces memory
	//out->indeces.resize(depth->width_, depth->height_);
	////get number of non zero depth pixels
	//int n = 0;
	//for (int y = 0; y < depth->height_; y++) {
	//	for (int x = 0; x < depth->width_; x++) {
	//		if (depth->PointerAt<uint16_t>(x, y) != 0) {
	//			n++;
	//		}
	//		else {
	//			out->indeces(x, y) = -1; //note one can get rid of this for big pcds if genkps is slightly changed
	//		}

	//	}

	//}
	////reserve memory
	//out->points_.reserve(n);
	//out->colors_.reserve(n);
	//int n2 = 0;
	//for (int y = 0; y < depth->height_; y++) {
	//	for (int x = 0; x < depth->width_; x++) {
	//		double d = *depth->PointerAt<uint16_t>(x, y);
	//		if (d != 0) {
	//			out->points_.emplace_back(Eigen::Vector3d((x - g_intrinsic.intrinsic_matrix_(0, 2)) * d / g_intrinsic.intrinsic_matrix_(0, 0), 
	//				(y - g_intrinsic.intrinsic_matrix_(1, 2)) * d / g_intrinsic.intrinsic_matrix_(1, 1), 
	//				d));
	//			out->colors_.emplace_back(Eigen::Vector3d(*color->PointerAt<uint8_t>(x, y, 0), *color->PointerAt<uint8_t>(x, y, 1), *color->PointerAt<uint8_t>(x, y, 2)));
	//			out->indeces(x, y) = n2;
	//			n2++;
	//		}
	//	}
	//}

	//return out;
}

//string operator+ (const int& i_wert, const string& s_wert)
//{
//	string s = to_string(i_wert);
//	s.append(s_wert);
//	return s;
//}


//puts the thread to sleep until a frame is available
std::shared_ptr<Frame> getSingleFrame(list <shared_ptr<Frame>>& framebuffer, list <shared_ptr<Frame>>& recordbuffer)
{
	
	while (framebuffer.empty() && !g_pause) {
		std::this_thread::sleep_for(20ms);
	}

	if (g_pause) {
		return nullptr;
	}

	g_bufferlock.lock();
	auto tmp = framebuffer.front();
	framebuffer.pop_front();
#ifdef RECORDBUFFER
	recordbuffer.push_back(tmp);
#endif
	g_bufferlock.unlock();

	return tmp;
}


//###############################################################################headeless internal functions#######################################################################################



std::string getPicNumberString(int a) {
	std::string out = std::to_string(a);
	while (out.length() < 6) {
		out = "0" + out;
	}
	return out;
}


void loadScanNNCameraPath(const std::string& path, std::vector<Eigen::Matrix4d_u>& out) {
	int count = 0;
	for (const auto& entry : std::experimental::filesystem::directory_iterator(path))
		count++;
	for (int i = 0; i < count; i++) {
		out.push_back(txttoPose2(path + "\\" + std::to_string(i) + ".txt"));
	}
}


Eigen::Matrix4d_u txttoPose2(const std::string & filename) {
	std::ifstream myfile(filename);
	Eigen::Matrix4d_u output;
	std::string line, word, temp;
	int x = 0;
	if (myfile.is_open())
	{
		while (getline(myfile, line))
		{
			std::vector<std::string> row;
			std::stringstream s(line);
			while (s.good())
			{
				std::string substr;
				std::getline(s, substr, ' ');
				row.push_back(substr);
			}
			for (int y = 0; y < 4; y++) {
				output(x, y) = std::stod(row[y]);
			}
			x++;
		}
		myfile.close();
	} else {
		std::cout << "unable to open " << filename << std::endl;
	}

	return output;
}

Eigen::Matrix3d loadScanNNIntrinsic(const std::string & filename) {
	std::ifstream myfile(filename);
	std::string line, word, temp;
	Eigen::Matrix3d output;
	int x = 0;
	if (myfile.is_open())
	{
		while (getline(myfile, line) && x < 3)
		{
			std::vector<std::string> row;
			std::stringstream s(line);
			while (s.good())
			{
				std::string substr;
				std::getline(s, substr, ' ');
				row.push_back(substr);
			}
			output(x, 0) = std::stod(row[0]);
			output(x, 1) = std::stod(row[1]);
			output(x, 2) = std::stod(row[2]);
			x++;
		}
		myfile.close();
	} else {
		std::cout << "unable to open " << filename << std::endl;
	}
	return output;
}

//tested and working for our getT (x,y,z rot than trans)
Eigen::Vector6d MattoDof(const Eigen::Matrix4d& R) {
	// Decompose Rotation Matrix
	double x = atan2(R(2, 1), R(2, 2));
	double y = -asin(R(2, 0)); // problem if excactly 90 deg?
	double z = atan2(R(1, 0), R(0, 0));
	
	Eigen::Vector6d v; // check this if initialized ok
	v(0) = x;
	v(1) = y;
	v(2) = z;
	v(3) = R(0,3);
	v(4) = R(1,3);
	v(5) = R(2,3);

	//double* a = new double[6];
	//for (int i = 0; i < 6; i++) {
	//	a[i] = v(i);
	//}
	//cout << "original kabsch matrix: \n";
	//cout << R << endl;
	//cout << "reconstructed matrix from dof: \n";
	//cout << getT(a) << endl;

	//if ((R - getT(a)).norm() > 1e-5) {
	//	cout << "bad \n ";
	//}

	//delete[]a;
	return v;
}


//todo get inits over chains of kabsch if it doesnt work with the first frame
//get the initial 6 dofs for optimization for a intrachunk
vector<Eigen::Vector6d> getDoffromKabschChunk(const shared_ptr<Chunk> c)
{
	vector<Eigen::Vector6d> out;
	out.reserve(c->frames.size() - 1);
	for (int i = 1; i < c->frames.size(); i++) {
		if (!c->pairTransforms(i, 0).kabschtrans.isZero()) {
			if (c->frames[i]->unique_id < c->frames[0]->unique_id) { //this will never happen atm
				out.push_back(MattoDof(c->pairTransforms(i, 0).kabschtrans));
			}
			else {
				Eigen::Matrix4d tmp = c->pairTransforms(i, 0).kabschtrans.inverse();
				out.push_back(MattoDof(tmp));
			}
		}
		else {
			Eigen::Vector6d v;
			v << 0, 0, 0, 0, 0, 0;
			out.push_back(v);
		}
	}


	return out;
}

//todo get inits over chains of kabsch if it doesnt work with the first frame
//is only called if there are at least two chunks!
//uses the chunktransform of last chunk and the kabsch results of the current chunk
vector<Eigen::Vector6d> getInitialDofs(Model& m)
{
	vector<Eigen::Vector6d> out;
	out.reserve(m.chunks.size() - 2);
	for (int i = 1; i < m.chunks.size()-1; i++) {
		out.push_back(MattoDof(m.chunks[i]->chunktoworldtrans));
	}
	// last chunk gets transform of former chunk as starting point + kabschtrans
	Eigen::Matrix4d kabscht = m.chunks.back()->pairTransforms(0, m.chunks.back()->frames.size() - 1).kabschtrans;
	if (!kabscht.isZero()) {
		kabscht = kabscht.inverse();
	}
	else {
		kabscht = Eigen::Matrix4d::Identity();
	}

	out.push_back(MattoDof(kabscht * m.chunks[m.chunks.size()-2]->chunktoworldtrans));

	


	return out;
}

int getdircount(string path) {
	int count = 0;
	for (const auto& entry : std::experimental::filesystem::directory_iterator(path)) {
		count++;
	}
	return count;
}


#ifdef RECORDBUFFER
//once cpp 17 replace with std lib functions
void saveImagestoDisc(string& path, list <shared_ptr<Frame>>& recordbuffer) {
	for (int i = 0; i < path.length(); i++) {
		if (path[i] == '/') {
			path.replace(i,1,"\\");
			i++;
		}
	}
	cout <<"Saving images to " << path << endl;
	cout <<"Do not do anything till this is finished !\n";
	//it is guaruanteed that camera paras are set at this point
	ofstream intrinsicFile(path + "\\intrinsic.txt");
	intrinsicFile << g_intrinsic.width_ << endl << g_intrinsic.height_ << endl << g_intrinsic.intrinsic_matrix_;
	intrinsicFile.close();

	string removeOrder = "rmdir /s /q " + path;				
	string tmpPath;
	// remove color-directory
	tmpPath = "rmdir /s /q " + path + "\\color"; //quiet and ...
	system(tmpPath.c_str());
	// make color-directory
	tmpPath = path + "\\color";
	mkdir(tmpPath.c_str());
	// remove depth directory
	tmpPath = removeOrder + "\\depth";
	system(tmpPath.c_str());
	// make depth-directory
	tmpPath = path + "\\depth";
	mkdir(tmpPath.c_str());



	int pictureNumber = 0;
	while (!recordbuffer.empty()) {
		auto tmp = recordbuffer.front();
		recordbuffer.pop_front();
		auto jpgPath = path + "\\color/" + getPicNumberString(pictureNumber) + ".png";
		auto pngPath = path + "\\depth/" + getPicNumberString(pictureNumber) + ".png";
		io::WriteImage(jpgPath, tmp->rgbd->color_);
		auto tmpimg = tmp->rgbd->depth_.CreateImageFromFloatImage<uint16_t>(1000.0);
		io::WriteImage(pngPath, *tmpimg);
		pictureNumber++;
	}

	cout <<"Finished saving images!\n";

}
#endif


#ifdef SAVETRAJECTORY
void saveTrajectorytoDisk(const string& path, Model& m, string name) {
	cout << "Saving trajectory to " << path << "\n";
	ofstream trajectory(path + name);
	for (int i = 0; i < m.chunks.size(); i++) {
		for (int j = 0; j < m.chunks[i]->frames.size(); j++) {
			auto&f = m.chunks[i]->frames[j];
			trajectory << f->getFrametoWorldTrans() << endl;
		}
	}

	trajectory.close();

	cout << "Done saving trajectory \n";

}
#endif

//reads mat4d as written by saveTrajectorytoDisk
void readTrajectory(const string& path, vector<Eigen::Matrix4d>& poses) {
	cout << "reading " << path  << endl;
	ifstream trajectory(path);
	string line;
	std::string delimiter = " ";
	if (trajectory.is_open()) {
		int lineNumber = 0;
		while (getline(trajectory, line)) {
			//cout << " line " << line << endl;
			int row = lineNumber %4;
			if (row == 0) {
				poses.emplace_back(Eigen::Matrix4d());
			}
			std::string token;
			size_t pos;
			int col = 0;
			while ((pos = line.find(delimiter)) != std::string::npos) {
				token = line.substr(0, pos);
				if (token.compare("")){ //this stupid shit return wrong if strings are equal
				//	cout << token;// << delimiter;
					poses.back()(row,col) = stod(token);
					line.erase(0, pos + delimiter.length());
					col++;
				} else {
					//cout << "token is empty" << token << "after \n";
					line.erase(0,1);
				}
			}
			//cout << line <<  endl;

			poses.back()(row,col) = stod(line); //last entry
			lineNumber++;
		}
	} else {
		cout << "Error: Could not read trajectory file \n";
	}

}

