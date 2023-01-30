#include "core/reconrun.h"
#include <iostream>
#include <stdio.h>
#include <Open3D/Open3D.h>
#include "configvars.h"
#include "utils/coreutil.h"
#include "ptimer.h"
#include <Cuda/Open3DCuda.h>
#include "utils/visutil.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "Gui/Shader.h"
#include "Gui/imgui-docking/imgui.h"
#include "Gui/imgui-docking/imgui_impl_glfw.h"
#include "Gui/imgui-docking/imgui_impl_opengl3.h"
#include "Gui/imgui-docking/FileBrowser/dirent.h"
#include "Gui/imgui-docking/FileBrowser/ImGuiFileBrowser.h"
#include "cmakedefines.h"
#include "Gui/geometry.h"
#include "Gui/PandiaView.h"
#include "Gui/guiutil.h"
#include "Gui/guiwindow.h"
#include "postprocessing/postprocessing.h"
#include "Gui/meshing.h"


//using namespace std;
using namespace open3d;


void glfw_error_callback(int error, const char* description)
{
	fprintf(stderr, "Error: %s\n", description);
}



void InitWindow(GLFWwindow*& window, int& w_width, int& w_height) {

	if (!glfwInit()) {
		std::cout << "fatal error. glfw init not successfull \n";
	}
	
	glfwSetErrorCallback(glfw_error_callback);


	int visible = 1;
	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_VISIBLE, visible);


	string window_name_ = "From Scratch";

	window = glfwCreateWindow(w_width, w_height, window_name_.c_str(), NULL, NULL);
	if (!window) {
		utility::LogError("Failed to create window\n");
	}
	int left = 100; int top = 100;
	glfwSetWindowPos(window, left, top);
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);


	//glfwSetFramebufferSizeCallback(window,framebuffer_size_callback);
}

static bool spacedown = false;

void processInput(GLFWwindow* window, Model& m)
{
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, true);
		std::cout << "close callback received \n";
	}
	bool set =false;
	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_RELEASE  && spacedown) {
		if (g_programState == gui_RUNNING && !g_reconThreadFinished) {
			std::cout << "stopping \n";
			g_pause = true;
			g_programState = gui_PAUSE;
			g_wholeMesh = true;
			m.tsdf_cuda.unmeshed_data = true; //mesh the whole tsdf once
			set = true;
		}
		if (g_programState == gui_PAUSE && !g_current_slam_finished && !set) {
			std::cout << "resuming \n";
			g_pause = false;
			g_programState = gui_RUNNING; //programm runnign again
			g_wholeMesh = false;
		}
	}

	spacedown = (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS);

}

void initImGui(GLFWwindow* window) {
	const char* glsl_version = "#version 130";
	IMGUI_CHECKVERSION();
	// Setup Platform/Renderer bindings
	ImGui::CreateContext();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);
	ImGui::StyleColorsDark();
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
	//Style
	ImGuiStyle& style = ImGui::GetStyle();
	style.WindowBorderSize = 0.0f;
	style.FrameRounding = 5.0f;
	style.FrameBorderSize = 0.f; //Prevents GuiBlock-Element from being smaller resized than actual window
}



int main() {
	readconfig("config.txt");
	utility::SetVerbosityLevel((open3d::utility::VerbosityLevel)g_verbosity);
	//start threads
	thread cameraConnectionThread;
	cameraConnectionThread = thread(cameraConnectionThreadFunction); //check every second if camera is connected  and sets g_parameterset

	thread postProcessingThread;

	//init window
	GLFWwindow* window;
	InitWindow(window, g_initial_width, g_initial_height);


	//init opengl with glew here
	if (glewInit() != GLEW_OK) {
		std::cout << "something is very wrong! glew init failed \n";
		return false;
	}
	//init ImGUI  //order is critical here otherwise imgui is missing events
	PandiaView view(window);
	initImGui(window);
	PandiaGui::initPandiaGui(); //set variables, load textures, etc.

	//########################## shader stuff #######################
	string shaderpath = TOSTRING(SHADER_PATH);
	//get Shader from text files
	string vs_name = "o3d_phong_vertex_shader.txt";
	string fs_name = "o3d_phong_fragment_shader.txt";
	string vsp = shaderpath + vs_name;
	string fsp = shaderpath + fs_name;
	Shader o3dphongShader(vsp.c_str(), fsp.c_str());
	//o3dphongShader.use();

	Eigen::Matrix4f lightpos = Eigen::Matrix4f::Identity();
	o3dphongShader.setUniform4Matrix("light_position_world_4", lightpos); // set to zero position
	o3dphongShader.setUniform4Matrix("M", Eigen::Matrix4f::Identity());
	float unitlength = 0.70710678118 * 0.32;
	Eigen::Matrix4f neutralmatrix;
	neutralmatrix << unitlength, unitlength, unitlength, unitlength,
		unitlength, unitlength, unitlength, unitlength,
		unitlength, unitlength, unitlength, unitlength,
		1, 1, 1, 1;
	Eigen::Vector4f unitvec(unitlength, unitlength, unitlength, 1);
	o3dphongShader.setUniform4Matrix("light_color_4", neutralmatrix);
	o3dphongShader.setUniform4("light_diffuse_power_4", unitvec);
	o3dphongShader.setUniform4("light_specular_power_4", unitvec);
	o3dphongShader.setUniform4("light_specular_shininess_4", unitvec);
	o3dphongShader.setUniform4("light_ambient", unitvec);


	vs_name = "simple_vertex_shader.txt";
	fs_name = "simple_fragment_shader.txt";
	vsp = shaderpath + vs_name;
	fsp = shaderpath + fs_name;
	Shader simpleShader(vsp.c_str(), fsp.c_str());

	vs_name = "wireframe_vertex_shader.txt";
	fs_name = "wireframe_fragment_shader.txt";
	vsp = shaderpath + vs_name;
	fsp = shaderpath + fs_name;
	Shader wireFrameShader(vsp.c_str(), fsp.c_str());
	  
	vs_name = "phong_vertex_shader.txt";
	fs_name = "phong_fragment_shader.txt";
	vsp = shaderpath + vs_name;
	fsp = shaderpath + fs_name;
	Shader phongShader(vsp.c_str(), fsp.c_str());
	phongShader.setVec3("lightColor", Eigen::Vector3f::Ones());
	Eigen::Vector3f ldir(0.0, -1.0, 0.3); ldir.normalize();
	phongShader.setVec3("lightDir", ldir);

	vs_name = "empty_vertex_shader.txt";
	fs_name = "empty_fragment_shader.txt";
	vsp = shaderpath + vs_name;
	fsp = shaderpath + fs_name;
	Shader emptyShader(vsp.c_str(), fsp.c_str());
	Eigen::Vector4f background_color(0.45f, 0.55f, 0.60f, 1.00f);
	emptyShader.setUniform4("backgroundColor", background_color);

	vs_name = "raycast_vertex_shader.txt";
	fs_name = "raycast_fragment_shader.txt";
	vsp = shaderpath + vs_name;
	fsp = shaderpath + fs_name;
	Shader raycastShader(vsp.c_str(), fsp.c_str());



	//get opengl locations for shader inputs. 
	//In all shaders the following location and naming convention must be held true:
	//layout (location = 0) in vec3 vertex_position;
	//layout (location = 1) in vec3 vertex_color;
	//layout (location = 2) in vec3 vertex_normal;
	GLuint vertex_location = 0;
	GLuint color_location = 1;
	GLuint normal_location = 2;

	//########################## end shader stuff #######################

	//build and init view object that handles camera position, opengl viewport, mvp matrix generation and framebuffer
	glViewport(0, 0, g_initial_width, g_initial_height);
	view.ChangeWindowSize(g_initial_width, g_initial_height);
	view.Reset();
	view.SetConstantZFar(2.5 * g_cutoff);
	view.SetConstantZNear(g_mincutoff);
	view.lookat_ = Eigen::Vector3d(0, 0, 0);
	view.front_ = Eigen::Vector3d(0, 0, -1);
	view.SetProjectionParameters();
	view.SetViewMatrices();
	view.genFramebuffer(); //generates a framebuffer for us to draw upon

	//set background color
	glClearColor(background_color(0), background_color(1), background_color(2), background_color(3));
	
	glEnable(GL_DEPTH_TEST);
	

	//####################################define imGUI util variables####################
	ImVec2 oldWindowSize(0, 0);
	bool resetAll = false;
	bool open = true;
	//####################################end define imGUI util variables####################



	//################################### MAIN LOOP START #######################################
	//################################### MAIN LOOP START #######################################
	//################################### MAIN LOOP START #######################################
	//################################### MAIN LOOP START #######################################
	//################################### MAIN LOOP START #######################################
	Model m;
	thread reconstructionThread;
	thread meshThread;
	PandiaGui::PandiaClock clock(false);
	GLGeometry cudaMesh(m.mesher.mesh(), vertex_location, color_location, normal_location);
	shared_ptr<GLGeometry> cpuMesh;
	shared_ptr<GLGeometry> cameraMesh; //just cameras for each chunk position
	shared_ptr<GLGeometry> cameraConnectorMesh; // connector lines for drawing them red
	GLGeometry* currentGeometryPointer = &cudaMesh; //either cudamesh, cpumesh, or raycast stuff
	Shader* currentShaderPointer = &simpleShader;
	currentShaderPointer->use();

	cuda::ImageCuda<float, 3> raycasted(640, 480);
	GLGeometry cudaRaycastImage(raycasted, vertex_location, color_location, normal_location);
	
	cuda::TransformCuda cudapos = cuda::TransformCuda::Identity();

	GLuint o3dMVP = o3dphongShader.getUniformLocation("MVP");
	GLuint o3dV = o3dphongShader.getUniformLocation("V");
	GLuint simpleMVP = simpleShader.getUniformLocation("MVP");
	GLuint wireFrameMVP = wireFrameShader.getUniformLocation("MVP");
	GLuint phongMVP = phongShader.getUniformLocation("MVP");
	GLuint phongViewpos = phongShader.getUniformLocation("viewPos");
	GLuint rayCastMVP = raycastShader.getUniformLocation("MVP");


	while (!glfwWindowShouldClose(window))
	{
		if (resetAll) {
			//placement new for a clean slate
			m.~Model();
			new (&m) Model;
			cudaMesh.~GLGeometry();
			new(&cudaMesh) GLGeometry(m.mesher.mesh(), vertex_location, color_location, normal_location);
			resetAll = false;

			raycasted.~ImageCuda();
			new (&raycasted) cuda::ImageCuda<float, 3>(640, 480);
		}

		//check if mesh has changed and update opengl buffers and camera pos
		if (m.meshchanged) {
			cudaMesh.changeGeometry(m.mesher.mesh());
			m.meshchanged = false;
		}

		if (g_programState == gui_RUNNING){ // do not set cam pos in post processing
			currentposlock.lock(); //todo should be different lock (minor)
			view.setRealCPos(m.currentPos);
			if (PandiaGui::shadertyp ==  3) 
				cudapos.FromEigen(m.currentPos);
			currentposlock.unlock();
			if (PandiaGui::shadertyp == 3) {
				pandia_integration::tsdfLock.lock();
				m.tsdf_cuda.RayCasting(raycasted, g_intrinsic_cuda, cudapos);
				pandia_integration::tsdfLock.unlock();
			}
		}

		//Only processes Input if no ImGui-FileDialog is open. Prevents closing, resuming, etc.
		//Prevents from actions while typing file names
		if(!PandiaGui::fileDialogOpenNow)
			processInput(window,m); //only close and toggle resume

		view.SetViewMatrices(); //calculates mvp matrix

		
		glBindFramebuffer(GL_FRAMEBUFFER, view.framebufferID); //bind our framebuffer to render in it

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); //clears color- and depth buffer of OpenGL
	

		//Cpu mesh can't be rendered by raycasting
		if (currentGeometryPointer == cpuMesh.get() && PandiaGui::shadertyp == 3) {
			PandiaGui::shadertyp = 0;
		}

		//integer makes sense here for imgui compabtibility
		switch (PandiaGui::shadertyp)
		{
		case -1:
			currentShaderPointer = &emptyShader;
			break;
		case 0:
			currentShaderPointer = &simpleShader;
			simpleShader.setUniform4Matrix(simpleMVP, view.GetMVPMatrix());

			break;
		case 1:
			currentShaderPointer = &phongShader;
			phongShader.setUniform4Matrix(phongMVP, view.GetMVPMatrix());
			phongShader.setUniform4Matrix(phongViewpos, m.currentPos.cast<float>());
			break;
		case 2:
			currentShaderPointer = &o3dphongShader;
			o3dphongShader.setUniform4Matrix(o3dMVP, view.GetMVPMatrix());//give mvp matrix to opengl
			o3dphongShader.setUniform4Matrix(o3dV, view.GetViewMatrix()); //o3d
			break;
		case 3:
			currentShaderPointer = &raycastShader;
			currentGeometryPointer = &cudaRaycastImage;
			// currentPos MUST be taken into account of MVPMatrix, else coordination system origin is wrong
			raycastShader.setUniform4Matrix(currentShaderPointer->getUniformLocation("MVP"), view.GetMVPMatrix() * m.currentPos.cast<float>());
			currentGeometryPointer->changeGeometry(raycasted);
		default:
			break;
		}



		//show camera path (only in valid states)
		if (PandiaGui::menu_showCameraPath && !m.chunks.empty() && g_programState == gui_PAUSE) {

			//build camera path once
			if (!PandiaGui::cameraPathBuild) {
				cameraMesh = make_shared<GLGeometry>(getCameraPathMesh(m), vertex_location, color_location);
				cameraConnectorMesh = make_shared<GLGeometry>(createPathMesh(m), vertex_location, color_location);
				PandiaGui::cameraPathBuild = true;
				std::cout << "building camera \n";
			}

			wireFrameShader.setUniform4Matrix(wireFrameShader.getUniformLocation("MVP"), view.GetMVPMatrix());
			simpleShader.setUniform4Matrix(simpleMVP, view.GetMVPMatrix());

			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			wireFrameShader.use();
			cameraMesh->draw();
			simpleShader.use();
			cameraConnectorMesh->draw();

			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			cameraMesh->draw();

			currentShaderPointer->use();
		}



		currentShaderPointer->use();
		currentGeometryPointer->draw();


		//handle wireframe rendering
		if (view.wireframe) {

			wireFrameShader.setUniform4Matrix(wireFrameMVP, view.GetMVPMatrix());

			wireFrameShader.use();
			if(PandiaGui::shadertyp == 3)//raycasting
				wireFrameShader.setUniform4Matrix(wireFrameShader.getUniformLocation("MVP"), view.GetMVPMatrix() * m.currentPos.cast<float>());

			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			currentGeometryPointer->draw();
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			currentShaderPointer->use();
		}
		glBindFramebuffer(GL_FRAMEBUFFER, 0);  //unbind framebuffer

	

//######################################## Start ImGUI stuff ###############################
		// Start the Dear ImGui frame, no ide what this does internally
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		//get viewport to make next element fill whole screen
		ImGuiViewport* imgui_viewport = ImGui::GetMainViewport();
		ImGui::SetNextWindowPos(imgui_viewport->GetWorkPos());
		ImGui::SetNextWindowSize(imgui_viewport->GetWorkSize());
		ImGui::SetNextWindowViewport(imgui_viewport->ID);


		//This element fills the whole screen and is a docking space with a menu bar.
		PandiaGui::window_Menubar(view);

#ifdef DEVELOPERTOOLS
		if (PandiaGui::menu_DebugWindow) {
			PandiaGui::showDebugWindow(m, view);
		}
		if (PandiaGui::menu_DemoWindow) {
			ImGui::ShowDemoWindow();
		}
#endif //DEVELOPERTOOLS


		
		//################################## left menu bar static components ############################
		ImGui::Begin("Control Window", NULL, PandiaGui::subWindow_flags | ImGuiWindowFlags_AlwaysVerticalScrollbar);
		PandiaGui::absolutWindowPos = ImGui::GetWindowPos();
		bool oldflag = PandiaGui::loadDataFlag;
		PandiaGui::GuiBlockItem(PandiaGui::disableInput); //GuiBlockItem needs to be set after ImGui-Item which to block and before next ImGui-Item
		ImGui::Checkbox(PandiaGui::lan_loadData.c_str(), &PandiaGui::loadDataFlag);
		g_take_dataCam = PandiaGui::loadDataFlag;
		if (oldflag != PandiaGui::loadDataFlag) {
			g_cameraParameterSet = false;
			g_cameraConnected = false;
			if (!PandiaGui::loadDataFlag) currentGeometryPointer = &cudaMesh;
		}
		

		PandiaGui::InfoMarker(PandiaGui::lan_loadDataInfo.c_str(), PandiaGui::menu_InfoMode, PandiaGui::infoSymbol_texture);
		ImGui::Spacing(); ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing(); ImGui::Spacing();
		
		if (PandiaGui::loadDataFlag) { //reconstruction from data

			PandiaGui::GuiBlockItem(PandiaGui::disableInput);
			ImGui::RadioButton(PandiaGui::lan_loadMesh.c_str(), &PandiaGui::loadOptionSwitch, 0);
			PandiaGui::InfoMarker(PandiaGui::lan_loadMeshInfo.c_str(), PandiaGui::menu_InfoMode, PandiaGui::infoSymbol_texture);

			ImGui::Spacing(); ImGui::Spacing();

			PandiaGui::GuiBlockItem(PandiaGui::disableInput);
			ImGui::RadioButton(PandiaGui::lan_loadImages.c_str(), &PandiaGui::loadOptionSwitch, 1);
			PandiaGui::InfoMarker(PandiaGui::lan_loadImagesInfo.c_str(), PandiaGui::menu_InfoMode, PandiaGui::infoSymbol_texture);

			ImGui::Text("");
			ImGui::Separator();
			ImGui::Text("");

			if (PandiaGui::loadOptionSwitch == 0) {//load mesh case
				if (ImGui::Button(PandiaGui::lan_loadMeshDirectory.c_str())) {
					ImGui::OpenPopup("Load Mesh");
					PandiaGui::file_dialogLoadMesh.current_path = PandiaGui::getTopDirectoryPath(PandiaGui::selectedMeshPath);
				}
				PandiaGui::InfoMarker(PandiaGui::lan_loadMeshDirectoryInfo.c_str(), PandiaGui::menu_InfoMode, PandiaGui::infoSymbol_texture);
				ImGui::Spacing(); ImGui::Spacing();
				ImGui::Text(PandiaGui::lan_loadMeshTextHeader.c_str());
				PandiaGui::InfoMarker(PandiaGui::lan_loadMeshTextHeaderInfo.c_str(), PandiaGui::menu_InfoMode, PandiaGui::infoSymbol_texture);
				ImGui::PushStyleColor(ImGuiCol_Text, PandiaGui::col_yellow);
				ImGui::TextWrapped(PandiaGui::selectedMeshPath.c_str());
				ImGui::PopStyleColor();

				ImGui::Spacing();
				ImGui::Spacing();
				ImGui::Separator();
				ImGui::Spacing();
				ImGui::Spacing();

				ImGui::Text("");
				ImGui::SetCursorPosX((ImGui::GetWindowSize().x - PandiaGui::buttonsize.x) * 0.5f);
				ImGui::PushStyleColor(ImGuiCol_Button, PandiaGui::col_green_1);
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, PandiaGui::col_green_2);
				ImGui::PushStyleColor(ImGuiCol_ButtonActive, PandiaGui::col_green_3);
				if (ImGui::Button(PandiaGui::lan_ButtonLoadMesh.c_str(), PandiaGui::buttonsize)) {
					io::ReadTriangleMesh(PandiaGui::selectedMeshPath, *PandiaGui::cpumesh, false);

					cpuMesh = make_shared<GLGeometry>(PandiaGui::cpumesh, vertex_location, color_location, normal_location);
					currentGeometryPointer = cpuMesh.get();
					if (PandiaGui::shadertyp == 3) { //raycasting not possible here
						PandiaGui::shadertyp = 0;
					}
				}
				ImGui::PopStyleColor(3);
			}


			if (PandiaGui::loadOptionSwitch == 1) { //load image case
					
				currentGeometryPointer = &cudaMesh;
				PandiaGui::GuiBlockItem(PandiaGui::disableInput);
				if (ImGui::Button(PandiaGui::lan_loadImagesDirectory.c_str())) {
					ImGui::OpenPopup("Load Images From");
					//navigates to one directory above current one and sets FileDialog to custom path
					//to enable this current_path must be changed from private to public
					PandiaGui::file_dialogLoadImage.current_path = PandiaGui::getTopDirectoryPath(g_readimagePath);
				}
				PandiaGui::InfoMarker(PandiaGui::lan_loadImagesDirectoryInfo.c_str(), PandiaGui::menu_InfoMode, PandiaGui::infoSymbol_texture);
				ImGui::Spacing(); ImGui::Spacing();
				ImGui::Text(PandiaGui::lan_loadImagesTextHeader.c_str());
				PandiaGui::InfoMarker(PandiaGui::lan_loadImagesTextHeaderInfo.c_str(), PandiaGui::menu_InfoMode, PandiaGui::infoSymbol_texture);
				ImGui::PushStyleColor(ImGuiCol_Text, PandiaGui::col_yellow);
				ImGui::TextWrapped(g_readimagePath.c_str());
				ImGui::PopStyleColor();

				ImGui::Spacing();
				ImGui::Spacing();
				ImGui::Separator();

			}
		}


		//this sets currentMeshPointer to the cpu mesh
		PandiaGui::handle_FileDialogs(m, cpuMesh, currentGeometryPointer, vertex_location, color_location, normal_location);


		//################################### now start with programm state dependend gui stuff #########################
		PandiaGui::buttonsize = ImVec2(ImGui::GetWindowSize().x * 0.8f, 60.f);

		switch (g_programState) {

			//nothing going on yet
		case gui_READY:
			//set mouse state dependend colors
			ImGui::Spacing(); ImGui::Spacing();
			ImGui::SetCursorPosX((ImGui::GetWindowSize().x - PandiaGui::buttonsize.x) * 0.5f);
			if (PandiaGui::loadOptionSwitch == 1 || PandiaGui::loadDataFlag == 0) { //dont show button if we are loading mesh
					
				if (g_cameraParameterSet) { //enable button

					switch (g_camType) {
						case camtyp::typ_kinect:
							ImGui::SetCursorPosX((ImGui::GetWindowSize().x - ImGui::CalcTextSize(PandiaGui::lan_InfoAzure.c_str()).x) * 0.5f);
							ImGui::Text(PandiaGui::lan_InfoAzure.c_str());
							break;
						case camtyp::typ_realsense:
							ImGui::SetCursorPosX((ImGui::GetWindowSize().x - ImGui::CalcTextSize(PandiaGui::lan_InfoRealsense.c_str()).x) * 0.5f);
							ImGui::Text(PandiaGui::lan_InfoRealsense.c_str());
							break;
						case camtyp::typ_data:
							ImGui::SetCursorPosX((ImGui::GetWindowSize().x - ImGui::CalcTextSize(PandiaGui::lan_InfoDataCam.c_str()).x) * 0.5f);
							ImGui::Text(PandiaGui::lan_InfoDataCam.c_str());
							break;
						default:
							break;
					}

					ImGui::SetCursorPosX((ImGui::GetWindowSize().x - PandiaGui::buttonsize.x) * 0.5f);
					ImGui::PushStyleColor(ImGuiCol_Button, PandiaGui::col_green_1);
					ImGui::PushStyleColor(ImGuiCol_ButtonHovered, PandiaGui::col_green_2);
					ImGui::PushStyleColor(ImGuiCol_ButtonActive, PandiaGui::col_green_3);
					if (ImGui::Button(PandiaGui::lan_ButtonStart.c_str(), PandiaGui::buttonsize)) {
						view.setFOVfromCamera(g_intrinsic);
						view.setVirtualIntrinsic();
						reconstructionThread = thread(reconrun, std::ref(m), false, true);//model, test, livevis, integration
						meshThread = thread(meshThreadFunction, &m);
						clock.start();
						PandiaGui::disableInput = true;
						g_programState = gui_RUNNING;
					};
					ImGui::PopStyleColor(3);
				}
				else { //block button
					sprintf(PandiaGui::animationBuffer, PandiaGui::lan_AnimationText.c_str(), "|/-\\"[(int)(ImGui::GetTime() / 0.25f) & 3]);
					ImGui::SetCursorPosX((ImGui::GetWindowSize().x - ImGui::CalcTextSize(PandiaGui::lan_AnimationText.c_str()).x) * 0.5f);
					ImGui::TextColored(PandiaGui::col_yellow, PandiaGui::animationBuffer);
					ImGui::SetCursorPosX((ImGui::GetWindowSize().x - PandiaGui::buttonsize.x) * 0.5f);
					ImGui::PushStyleColor(ImGuiCol_Button, PandiaGui::col_gray);
					ImGui::PushStyleColor(ImGuiCol_ButtonHovered, PandiaGui::col_gray);
					ImGui::PushStyleColor(ImGuiCol_ButtonActive, PandiaGui::col_gray);
					ImGui::Button(PandiaGui::lan_ButtonStartDummy.c_str(), PandiaGui::buttonsize);
					ImGui::PopStyleColor(3);
				}
			}
			break;
			//reconrun is executing
		case gui_RUNNING:
			//set mouse state dependend colors
			ImGui::Text("");
			ImGui::PushStyleColor(ImGuiCol_Button, PandiaGui::col_red_1);
			ImGui::PushStyleColor(ImGuiCol_ButtonHovered, PandiaGui::col_red_2);
			ImGui::PushStyleColor(ImGuiCol_ButtonActive, PandiaGui::col_red_3);
			ImGui::Spacing(); ImGui::Spacing();
			ImGui::SetCursorPosX((ImGui::GetWindowSize().x - PandiaGui::buttonsize.x) * 0.5f);
			if (ImGui::Button(PandiaGui::lan_ButtonPause.c_str(), PandiaGui::buttonsize)) {
				//note if something changes here, same must be done in reconrun nread pause segment
				g_pause = true;
				g_programState = gui_PAUSE;
				g_wholeMesh = true;
				m.tsdf_cuda.unmeshed_data = true; //mesh the whole tsdf once
				clock.stop();
			}
			//GuiBlockItem(g_cameraWarmup, buttonsize);
			ImGui::PopStyleColor(3);

			break;
			//programm paused or reconstruction finished
		case gui_PAUSE:
			if (!g_current_slam_finished) {
				clock.stop();
				ImGui::Text("");
				//color stuff
				ImGui::PushStyleColor(ImGuiCol_Button, PandiaGui::col_green_1);
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, PandiaGui::col_green_2);
				ImGui::PushStyleColor(ImGuiCol_ButtonActive, PandiaGui::col_green_3);
				ImGui::Spacing(); ImGui::Spacing();
				ImGui::SetCursorPosX((ImGui::GetWindowSize().x - PandiaGui::buttonsize.x) * 0.5f);
				if (ImGui::Button(PandiaGui::lan_ButtonResume.c_str(), PandiaGui::buttonsize)) {
					g_pause = false;
					g_programState = gui_RUNNING; //programm running again
					g_wholeMesh = false;
					PandiaGui::cameraPathBuild = false;
					clock.start();
				}
				PandiaGui::GuiBlockItem(PandiaGui::g_postProcessing, PandiaGui::buttonsize);
				ImGui::PopStyleColor(3);
			}

			//save 
			ImGui::PushStyleColor(ImGuiCol_Button, PandiaGui::col_green_1);
			ImGui::PushStyleColor(ImGuiCol_ButtonHovered, PandiaGui::col_green_2);
			ImGui::PushStyleColor(ImGuiCol_ButtonActive, PandiaGui::col_green_3);
			ImGui::Spacing(); ImGui::Spacing();
			ImGui::SetCursorPosX((ImGui::GetWindowSize().x - PandiaGui::buttonsize.x) * 0.5f);
			if (ImGui::Button(PandiaGui::lan_ButtonSaveMesh.c_str(), PandiaGui::buttonsize)) {
				PandiaGui::file_dialogSaveMesh.current_path = PandiaGui::getTopDirectoryPath(PandiaGui::saveMeshPath);
				ImGui::OpenPopup("Save mesh to"); //this start the Open File Dialog method 
				g_programState = gui_PAUSE;
			}
			//is true if any buffer has any data
			PandiaGui::nonemptyBuffer = !(pandia_integration::reintegrationBuffer.size() == 0 && pandia_integration::integrationBuffer.size() == 0 && pandia_integration::deintegrationBuffer.size() == 0);
			PandiaGui::GuiBlockItem(PandiaGui::g_postProcessing, PandiaGui::buttonsize);
			PandiaGui::GuiBlockItem(PandiaGui::nonemptyBuffer, PandiaGui::buttonsize);
			ImGui::PopStyleColor(3);
			//clear mesh
			ImGui::Spacing(); ImGui::Spacing();
			ImGui::PushStyleColor(ImGuiCol_Button, PandiaGui::col_red_1);
			ImGui::PushStyleColor(ImGuiCol_ButtonHovered, PandiaGui::col_red_2);
			ImGui::PushStyleColor(ImGuiCol_ButtonActive, PandiaGui::col_red_3);
			ImGui::SetCursorPosX((ImGui::GetWindowSize().x - PandiaGui::buttonsize.x) * 0.5f);
			if (ImGui::Button(PandiaGui::lan_ButtonClearMesh.c_str(), PandiaGui::buttonsize)) {
				g_wholeMesh = false; //reset
				g_clear_button = true; //needs to be set before threads join, thread communication
				g_pause = false; //needs to be set before threads join
				PandiaGui::cameraPathBuild = false;
				if (reconstructionThread.joinable()) {
					reconstructionThread.join();
				}
				if (meshThread.joinable()) {
					stopMeshing = true;
					meshThread.join();
				}
				resetAll = true; //mark for beginning of loop
				g_clear_button = false; //needs to be set after threads have joined
				stopMeshing = false; //needs to be set after threads have joined
				pandia_integration::stopintegrating = false;
				PandiaGui::disableInput = false;
				clock.reset();
				//reset postprocessing flags
				PandiaGui::pp_denseAlign = false;
				PandiaGui::pp_meshReduction = false;
				PandiaGui::pp_voxelLength = false;
				//reset frame, chunk ID
				frame_id_counter.id = 0;
				chunk_id_counter.id = 0;
				g_programState = gui_READY;
			}
			PandiaGui::GuiBlockItem(PandiaGui::g_postProcessing, PandiaGui::buttonsize);
			ImGui::PopStyleColor(3);
			ImGui::Spacing();
			
			//################################### save images stuff ###############
#ifdef DEVELOPERTOOLS
			PandiaGui::window_SaveImageTrajectory(m);
#endif; //DEVELOPERTOOLS

			//post processing
			ImGui::Separator();
			ImGui::Spacing();
			ImGui::Text(PandiaGui::lan_TextPostProcessing.c_str());
			PandiaGui::InfoMarker(PandiaGui::lan_textPostProcessingInfo.c_str(), PandiaGui::menu_InfoMode, PandiaGui::infoSymbol_texture);
			ImGui::Spacing();
			PandiaGui::GuiBlockItem(PandiaGui::g_postProcessing);
			//ImGui::Checkbox("Dense alignment", &PandiaGui::pp_denseAlign);
			ImGui::Spacing();
			PandiaGui::GuiBlockItem(PandiaGui::g_postProcessing);
			ImGui::Checkbox(PandiaGui::lan_voxelLength.c_str(), &PandiaGui::pp_voxelLength);
			PandiaGui::InfoMarker(PandiaGui::lan_voxelLenghtInfo.c_str(), PandiaGui::menu_InfoMode, PandiaGui::infoSymbol_texture);
			if (PandiaGui::pp_voxelLength) {
				PandiaGui::GuiBlockItem(PandiaGui::g_postProcessing);
				ImGui::SetCursorPosX((ImGui::GetWindowSize().x - PandiaGui::buttonsize.x) * 0.5f);
				ImGui::SliderFloat("##voxel length", &PandiaGui::voxelSlider, 0.5f, 3.0f, "%.1f cm");
				ImGui::SameLine();
				ImGui::PushButtonRepeat(true);
				if (ImGui::ArrowButton("decreaseVoxelLength", ImGuiDir_Left) && PandiaGui::voxelSlider > 0.5f) {
					PandiaGui::voxelSlider -= 0.1f;
					if (PandiaGui::voxelSlider < 0.5f) PandiaGui::voxelSlider = 0.5f;
				};
				ImGui::SameLine();
				if (ImGui::ArrowButton("increaseVoxelLength", ImGuiDir_Right) && PandiaGui::voxelSlider < 3.f) {
					PandiaGui::voxelSlider += 0.1f;
					if (PandiaGui::voxelSlider > 3.f) PandiaGui::voxelSlider = 3.f;
				};
				ImGui::PopButtonRepeat();
			}
			ImGui::Spacing();
			ImGui::Spacing();
			PandiaGui::GuiBlockItem(PandiaGui::g_postProcessing);
			ImGui::Checkbox(PandiaGui::lan_meshReduction.c_str(), &PandiaGui::pp_meshReduction);
			PandiaGui::InfoMarker(PandiaGui::lan_meshReductionInfo.c_str(), PandiaGui::menu_InfoMode, PandiaGui::infoSymbol_texture);
			if (PandiaGui::pp_meshReduction) {
				PandiaGui::GuiBlockItem(PandiaGui::g_postProcessing);
				ImGui::SetCursorPosX((ImGui::GetWindowSize().x - PandiaGui::buttonsize.x) * 0.5f);
				ImGui::SliderFloat("##reduction", &PandiaGui::meshSlider, 1.f, 100.f, "%.0f %%");
				ImGui::SameLine();
				ImGui::PushButtonRepeat(true);
				if (ImGui::ArrowButton("decreaseMeshReduction", ImGuiDir_Left) && PandiaGui::meshSlider > 1.f) {
					PandiaGui::meshSlider -= 1.f;
					if (PandiaGui::meshSlider < 1.f) PandiaGui::meshSlider = 1.f;
				};
				ImGui::SameLine();
				if (ImGui::ArrowButton("increaseMeshReduction", ImGuiDir_Right) && PandiaGui::meshSlider < 100.f) {
					PandiaGui::meshSlider += 1.f;
					if (PandiaGui::meshSlider > 100.f) PandiaGui::meshSlider = 100.f;
				};
				ImGui::PopButtonRepeat();
			}
			ImGui::Text("");

			ImGui::Spacing(); ImGui::Spacing();

			ImGui::SetCursorPosX((ImGui::GetWindowSize().x - PandiaGui::buttonsize.x) * 0.5f);
			if (ImGui::Button(PandiaGui::lan_ButtonStartPostProcessing.c_str(), PandiaGui::buttonsize)) {
				PandiaGui::overallPostProgress = 0.f;
				PandiaGui::currentProgress = 0.f;
				PandiaGui::openOverallProgress = 0.f;
				PandiaGui::openCurrentProgress = 0.f;

				PandiaGui::g_postProcessing = true;
				g_pause = false; // let programm resume so that it finishes
				if (reconstructionThread.joinable())
					reconstructionThread.join();
				g_programState = gui_PAUSE;

				if (PandiaGui::pp_meshReduction) {
					PandiaGui::meshSliderValue = (double)PandiaGui::meshSlider / 100;
				}

				if (PandiaGui::pp_voxelLength) {
					PandiaGui::voxelSliderValue = (double)PandiaGui::voxelSlider / 100;
				}

				//can only start if at least one post process checkbox is ticked
				postProcessingThread = thread(PandiaGui::PostProcessingThread, std::ref(m));
			}
			PandiaGui::GuiBlockItem(PandiaGui::g_postProcessing, PandiaGui::buttonsize);

			//if either denseAling or colorOptimize are checked GuiBlockItem won't block Button
			PandiaGui::postpro_disable = !(PandiaGui::pp_denseAlign || PandiaGui::pp_meshReduction || PandiaGui::pp_voxelLength);
			PandiaGui::GuiBlockItem(PandiaGui::postpro_disable, PandiaGui::buttonsize);
			break;

		default:
			break;
		}
		// ##################### state dependence over #########################

		//Check if any fileDialog is open. Needs to be inside ControlWindow
		PandiaGui::fileDialogOpenNow = (ImGui::IsPopupOpen("Load Mesh") || ImGui::IsPopupOpen("Load Images From") ||
			ImGui::IsPopupOpen("Save mesh to") || ImGui::IsPopupOpen("Save Image to:") || ImGui::IsPopupOpen("Save Trajectory to:"));

		ImGui::End();
		//####################################### end element ##########################



		//############################### starting state dependend gui drawings ########################

		//contains OpenGL Framebuffer and tracking lost indicator
		//MainViewPort for user
		PandiaGui::window_OpenGL(view);


		//####################################### start new element ##########################
		//text output
		if (PandiaGui::menu_InfoMode) {

			ImGui::Begin("info window", NULL, PandiaGui::subWindow_flags);
			
			
		

			ImGui::BeginTabBar("InfoTabBar", ImGuiTabBarFlags_None);
			ImVec2 pos = ImGui::GetCursorPos();

			if (ImGui::BeginTabItem("Session")) {

				int integrationcounter = pandia_integration::integratedframes.size();
				int framecounter = frame_id_counter.id;
				float effectivity;

				if (framecounter == 0)
					effectivity = 0;
				else
					effectivity = integrationcounter / (float)framecounter;
				ImGui::Spacing();
				ImGui::Text("Session timer:            %s", clock.getClock());
				ImGui::Spacing();
				ImGui::Text("Frame counter:            %d", framecounter);
				ImGui::Spacing();
				ImGui::Text("Effective frames:         %d", integrationcounter);
				ImGui::Spacing();
				ImGui::Text("Effectivity of recording:");		
				ImGui::SameLine();

				if (effectivity >= 0.9) {
					ImGui::PushStyleColor(ImGuiCol_Text, PandiaGui::col_brightGreen);
				}
				else if (0.8 < effectivity && effectivity < 0.9) {
					ImGui::PushStyleColor(ImGuiCol_Text, PandiaGui::col_yellow);
				}
				else {
					ImGui::PushStyleColor(ImGuiCol_Text, PandiaGui::col_red_2);
				}
				ImGui::Text("%.1f %%", 100 * effectivity);
				ImGui::PopStyleColor();
				
				ImGui::EndTabItem();
			}

			if (ImGui::BeginTabItem("History")) {
				ImGui::EndTabItem();
			}


			// Pandia-Logo
			float logo_width = 200.f;
			float logo_height = logo_width / PandiaGui::logo_aspect;

			int offset = 15;
			float windowheight = ImGui::GetWindowHeight() - pos.y - offset;
			float windowwidth = ImGui::GetWindowWidth();


			if ((logo_height) > windowheight) {
				logo_height = windowheight;
				logo_width = PandiaGui::logo_aspect * logo_height;
				//ImGui::SetCursorPosX(windowwidth - widthtmp - 15.f);
			}

			ImGui::SetCursorPos(ImVec2(ImGui::GetWindowWidth() - logo_width - offset, pos.y + 5));
			//ImGui::Image((void*)(intptr_t)PandiaGui::logo_texture, ImVec2((int)logo_width, (int)logo_height));

			ImGui::EndTabBar();
			
			ImGui::End();
		}
		//############################# end element #####################################
			
		
		//shows indication window when camera is ready to capture picture
		PandiaGui::window_WarmUp();

		//shows progress of postprocessing
		PandiaGui::window_PostProcessing();

		//shows if no camera is connected
		PandiaGui::window_CameraConnection();

		//legend for camera path display
		PandiaGui::window_CameraLegend(m);

		// shows generell information about the firm
		PandiaGui::window_AboutUs();


		//if no longer postprocessing and thread started
		if (!PandiaGui::g_postProcessing && postProcessingThread.joinable()) {
			postProcessingThread.join();
			PandiaGui::stopPostProcessing = false;
		}

		// Rendering
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		view.setWindowPos(PandiaGui::topleft.x, PandiaGui::topleft.y);
		if (oldWindowSize.x != PandiaGui::renderWindowSize.x || oldWindowSize.y != PandiaGui::renderWindowSize.y)
			view.ChangeWindowSize(PandiaGui::renderWindowSize.x, PandiaGui::renderWindowSize.y);
		oldWindowSize = PandiaGui::renderWindowSize;

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	PandiaGui::createPandiaGuiIni();

	//joins Threads if user wants to close program without proper clearing
	g_closeProgram = true;
	if (glfwWindowShouldClose(window)) {
		g_clear_button = true; //needs to be set before threads join
		g_pause = false; //needs to be set before threads join
		if (reconstructionThread.joinable()) {
			reconstructionThread.join();
		}
		if (meshThread.joinable()) {
			stopMeshing = true;
			meshThread.join();
		}
	}

	

	cameraConnectionThread.join();
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(window);
	glfwTerminate();
	std::cout << "after everything! \n";

	return 0;
}
//visualization::VisualizerWithCudaModule meshvis;
//if (!meshvis.CreateVisualizerWindow("Live Mesh", 1280, 720, 100, 100)) {
//	utility::LogWarning("Failed creating OpenGL window.\n");
//}
//meshvis.BuildUtilities();
//meshvis.UpdateWindowTitle();
//Timer t;
//meshvis.AddGeometry(m.cudamesh); //takes way long
//t.~Timer();
//meshvis.AddGeometry(getOrigin());
//while (true) {
//	meshvis.PollEvents(); //this takes long (5-20ms) probably blocks a lot!
//	meshvis.UpdateGeometry();
//}
//meshvis.DestroyVisualizerWindow();