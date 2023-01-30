#include <iostream>
#include <stdio.h>
#include <GL/glew.h>     
#include <GLFW/glfw3.h>

#include "Gui/imgui-docking/imgui.h"
#include "Gui/imgui-docking/imgui_impl_glfw.h"
#include "Gui/imgui-docking/imgui_impl_opengl3.h"
#include "Gui/imgui-docking/FileBrowser/dirent.h"
#include "Gui/imgui-docking/FileBrowser/ImGuiFileBrowser.h"
#include "recorder/recorderConnection.h"
#include <k4a/k4a.h>
#include <k4a/k4a.hpp>
#include <opencv2/opencv.hpp>
#include <Open3D/Geometry/Geometry.h>
#include <Open3D/Geometry/Image.h>
#include <cstdint>
#include <direct.h> //mkdir
#include "Gui/guiutil.h"


using namespace std;

//Variables



static void glfw_error_callback(int error, const char* description)
{
	fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}


void prepareSaveLocation(string& s) {

	for (int i = 0; i < s.length(); i++) {
		if (s[i] == '/') {
			s.replace(i, 1, "\\");
			i++;
		}
	}

	string removeOrder = "rmdir /s /q " + s;
	string tmpPath;
	// remove color-directory
	tmpPath = "rmdir /s /q " + s + "\\color"; //quiet and ...
	system(tmpPath.c_str());
	// make color-directory
	tmpPath = s + "\\color";
	mkdir(tmpPath.c_str());
	// remove depth directory
	tmpPath = removeOrder + "\\depth";
	system(tmpPath.c_str());
	// make depth-directory
	tmpPath = s + "\\depth";
	mkdir(tmpPath.c_str());
}


void saveIntrinsic(string& s) {
	for (int i = 0; i < s.length(); i++) {
		if (s[i] == '/') {
			s.replace(i, 1, "\\");
			i++;
		}
	}

	//cout << "Saving intrinsic to " << s << endl;

	ofstream intrinsicFile(s + "\\intrinsic.txt");
	intrinsicFile << recorder::recorderIntrinsic.width_ << endl
		<< recorder::recorderIntrinsic.height_ << endl << recorder::recorderIntrinsic.intrinsic_matrix_;
	intrinsicFile.close();

}



int main(){

	const char* glsl_version = "#version 130";

	// Create window with graphics context
	int initial_width = 1280;
	int initial_height = 720;
	bool startCapture = false;
	int color_width = -1;
	int color_height = -1;
	int depth_width = -1;
	int depth_height = -1;
	bool warmUp = true;
	int currentImageCount = 0;
	int desiredImageCount = 0;
	int internalImageCount = 0;

	GLFWwindow* window;
	thread recorderThread;
	recorderThread = thread(recorder::recorderConnectionThreadFunction);
	thread colorImageThread;
	thread depthImageThread;

	ImVec2 miniWindow;
	float warmUpCounter = 0.f;
	float waitFrames = 60.f;

	// Setup window
	glfwSetErrorCallback(glfw_error_callback);
	if (!glfwInit()) {
		cout << "Error. Glfw init unsuccessfull" << endl;
	}
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

	string window_name_ = "From Scratch";

	window = glfwCreateWindow(initial_width, initial_height, window_name_.c_str(), NULL, NULL);
	if (!window) {
		cout << "Failed to create Window" << endl;
	}

	glfwMakeContextCurrent(window);
	glfwSwapInterval(1); // Enable vsyncInit

	bool err = glewInit() != GLEW_OK;

	if (err)
	{
		fprintf(stderr, "Failed to initialize OpenGL loader!\n");
		return 1;
	}


	const ImGuiWindowFlags window_flags = /*ImGuiWindowFlags_MenuBar |*/ ImGuiWindowFlags_NoDocking |
		ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoResize |
		ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar;


	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;


	// Setup Platform/Renderer bindings
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);

	// Our state
	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

	bool capturesuccess = false;
	bool open = true;

	std::list<shared_ptr<k4a::image>> colorBuffer;
	std::list<shared_ptr<k4a::image>> depthBuffer;
	k4a::image liveFeed;
	k4a::transformation imageTransform(recorder::recorderCalibration);
	GLuint colorTexture;
	GLuint depthTexture;
	k4a::image colorImage;
	k4a::image depthImage;
	k4a::capture recordCapture;
	//int camera_x = 2048;
	//int camera_y = 1536;

	//Texture Init
	glGenTextures(1, &colorTexture);
	glBindTexture(GL_TEXTURE_2D, colorTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//glTexStorage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, camera_x, camera_y);


	//######################################################## Main loop ##################################################################
	while (!glfwWindowShouldClose(window))
	{
		//############################### ImGui Frame/Window Stuff ########################

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ImGuiViewport* imgui_viewport = ImGui::GetMainViewport();
		ImGui::SetNextWindowPos(imgui_viewport->GetWorkPos());
		ImGui::SetNextWindowSize(imgui_viewport->GetWorkSize());
		ImGui::SetNextWindowViewport(imgui_viewport->ID);

		ImGui::Begin("WholeWindow", &open, window_flags); //start menu bar
		ImGuiID dockspace_id = ImGui::GetID("DockSpace_Main");
		ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);
		ImGui::End();
		//############################### ImGui Frame/Window Stuff End ########################


		
		//############################### Camera capture and LiveFeed generation ########################
		try {
			//get capture and images from camera
			recorder::recorderDevice.get_capture(&recordCapture);
			colorImage = recordCapture.get_color_image();
			depthImage = recordCapture.get_depth_image();
			//depthImage = imageTransform.depth_image_to_color_camera(depthImage);

			liveFeed = colorImage; //image for window live_feed

			capturesuccess = true;
		}
		catch(...)		{
			cout << "connection lost" << endl;
			capturesuccess = false;
		}
		
		if (recorder::parameterSet && capturesuccess) {

			//generates ImageTexture for live feed


			color_width = liveFeed.get_width_pixels();
			color_height = liveFeed.get_height_pixels();
			unsigned char* image_data = (unsigned char*)liveFeed.get_buffer();
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, color_width, color_height, 0, GL_BGRA, GL_UNSIGNED_BYTE, image_data);
			//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, color_width, color_height, GL_BGRA, GL_UNSIGNED_BYTE, image_data);
		
		}
		
		//pushing images to buffers only if start is pressed / ends on stop
		if (recorder::parameterSet && capturesuccess && startCapture) {
			
			recorder::colorBufferLock.lock();
			colorBuffer.push_back(make_shared<k4a::image>(colorImage));
			recorder::colorBufferLock.unlock();
			recorder::depthBufferLock.lock();
			depthBuffer.push_back(make_shared<k4a::image>(depthImage));
			recorder::depthBufferLock.unlock();
			currentImageCount++;
			internalImageCount++;
		}

		//programm stops pushing to buffers after camera recovery
		if (!recorder::parameterSet)
			startCapture = false;

		if (desiredImageCount != 0) {
			if (internalImageCount >= desiredImageCount) {
				startCapture = false;
				internalImageCount = 0;
				desiredImageCount = 0;
			}
		}
		//############################### Camera capture and LiveFeed generation End ########################




		//############################### LiveFeed ########################
		ImGui::Begin("color_feed", &open, ImGuiWindowFlags_NoScrollbar);

		//adjusts Image size to window
		float camera_aspect = color_width / (float)color_height;
		float width_tmp = ImGui::GetWindowWidth();
		float height_tmp = width_tmp / camera_aspect;
		float windowheight = ImGui::GetWindowHeight();
		float windowwidth = ImGui::GetWindowWidth();
		if (height_tmp > windowheight) {
			height_tmp = windowheight - 15.f;
			width_tmp = height_tmp * camera_aspect;
		}
		ImGui::SetCursorPosX(windowwidth /2 - width_tmp /2);
		ImGui::Image((void*)(intptr_t)colorTexture, ImVec2(width_tmp, height_tmp));


		//visual indicator if warming up or if no connection
		if (warmUp || !recorder::parameterSet) {

			float progress = warmUpCounter / waitFrames;

			ImGui::SetNextWindowPos(ImVec2(width_tmp / 2 + ImGui::GetWindowPos().x, height_tmp / 2 + ImGui::GetWindowPos().y));
			ImGui::Begin("feedOverlay", &open, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration);
			//ImGui::SetWindowPos(ImVec2(topleft.x + renderWindowSize.x / 2 - miniWindow.x / 2, topleft.y + renderWindowSize.y / 2 - miniWindow.y));

			if (!recorder::parameterSet) {
				miniWindow = ImVec2(ImGui::CalcTextSize("no camera connection").x + 2 * ImGui::GetStyle().WindowPadding.x, -1.f);
				ImGui::Text("no camera connection");
			}
			else if (warmUp) {
				miniWindow = ImVec2(ImGui::CalcTextSize("Warming up camera").x + 2 * ImGui::GetStyle().WindowPadding.x, -1.f);
				ImGui::Text("Warming up camera");
				ImGui::ProgressBar(progress, ImVec2(-1.f, 0.f)); //shows progressbar how much frames are skipped

				warmUpCounter++;
				if (warmUpCounter > waitFrames) {
					warmUp = false;
					warmUpCounter = 0.f;
				}
			}

			ImGui::SetWindowSize(miniWindow);
			ImGui::End();
		}
		ImGui::End();
		//############################### LiveFeed End ########################



		//############################### Control Panel ########################
		ImGui::Begin("controle_window", &open);
		ImVec2 buttonsize(ImGui::GetWindowWidth(), 50.f);
		ImGui::Text("");

		//Hides Button if currently Recording
		if (!startCapture) {
			if (ImGui::Button("Save images to", ImVec2(buttonsize.x / 2, buttonsize.y / 2)))
			{
				ImGui::OpenPopup("Save Images");
			}
		}
		else {
			ImGui::Text("");
		}
		

		if (recorder::imagePath == "") {
			ImGui::PushStyleColor(ImGuiCol_Text, PandiaGui::col_yellow);
			ImGui::TextWrapped("Current save to: NO SAVE LOCATION ADDED");
			ImGui::PopStyleColor();
		}
		else {
			ImGui::TextWrapped("Current save to: %s", recorder::imagePath.c_str());
		}


		if (recorder::saveImages.showFileDialog("Save Images", imgui_addons::ImGuiFileBrowser::DialogMode::SELECT, ImVec2(700.f, 300.f))) {
			recorder::imagePath = recorder::saveImages.selected_path;

			recorder::closeProgram = true; //needs to be set to end threads
			if (colorImageThread.joinable()) {
				colorImageThread.join();
			}

			if (depthImageThread.joinable()) {
				depthImageThread.join();
			}
			recorder::closeProgram = false; //needs to be set to be ready for next program iteration

			startCapture = false;
			currentImageCount = 0;
			//recorder::imagePath = "";

			warmUp = false;
			while (!colorBuffer.empty()) {
				colorBuffer.pop_front();
			}

			while (!depthBuffer.empty()) {
				depthBuffer.pop_front();
			}
		};

		ImGui::Text("");
		ImGui::Separator();
		ImGui::Text("");

		ImGui::TextWrapped("Save color images as:");
		if (!startCapture) {
			ImGui::RadioButton("PNG", &recorder::choose_PNG_JPG, 0);
			ImGui::RadioButton("JPG", &recorder::choose_PNG_JPG, 1);
		}
		else {
			if (recorder::choose_PNG_JPG)
				ImGui::Text("JPG");
			else
				ImGui::Text("PNG");
		}

		ImGui::Text("");
		ImGui::Separator();
		ImGui::Text("");
		
		//shows/controlls custom ImageCounter
		if (!startCapture)
			ImGui::DragInt("no. save Images", &desiredImageCount, 1.f, 0, INT_MAX);
		else if(desiredImageCount == 0)
			ImGui::Text("You are saving all Images", desiredImageCount);
		else
			ImGui::Text("You are saving up to %d Images", desiredImageCount);


		// sets button visibility/functionality for program controlling
		if (!warmUp && recorder::parameterSet && recorder::imagePath != "") {

			if (!startCapture) { //state before pushing to buffers
				ImGui::PushStyleColor(ImGuiCol_Button, PandiaGui::col_green_1);
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, PandiaGui::col_green_2);
				ImGui::PushStyleColor(ImGuiCol_ButtonActive, PandiaGui::col_green_3);
				if (ImGui::Button("Start", buttonsize)) {
					startCapture = true;
					saveIntrinsic(recorder::imagePath);

					if (!colorImageThread.joinable() && !depthImageThread.joinable())
						prepareSaveLocation(recorder::imagePath); //delets old directory and creates new one

					if (!colorImageThread.joinable()) {
						colorImageThread = thread(recorder::saveColorImageThreadFunction, std::ref(colorBuffer));
					}

					if (!depthImageThread.joinable()) {
						depthImageThread = thread(recorder::saveDepthImageThreadFunction, std::ref(depthBuffer));
					}
				}
				ImGui::PopStyleColor(3);
			}
			else { //state currently pushing in buffers
				ImGui::PushStyleColor(ImGuiCol_Button, PandiaGui::col_red_1);
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, PandiaGui::col_red_2);
				ImGui::PushStyleColor(ImGuiCol_ButtonActive, PandiaGui::col_red_3);
				if (ImGui::Button("Stop", buttonsize)) {
					startCapture = false;
				}
				ImGui::PopStyleColor(3);
			}

		}
		else {// if warmUp / no connection
			ImGui::PushStyleColor(ImGuiCol_Button, PandiaGui::col_gray);
			ImGui::PushStyleColor(ImGuiCol_ButtonHovered, PandiaGui::col_gray);
			ImGui::PushStyleColor(ImGuiCol_ButtonActive, PandiaGui::col_gray);
			ImGui::Button("Start###nNoEffect", buttonsize);
			ImGui::PopStyleColor(3);
		}

		ImGui::Text("");
		if(ImGui::Button("Reset")) {//hard reset on program
			
			startCapture = false;
			currentImageCount = 0;
			recorder::imagePath = "";
			warmUp = false;

			recorder::closeProgram = true;
			if (colorImageThread.joinable()) {
				colorImageThread.join();
			}

			if (depthImageThread.joinable()) {
				depthImageThread.join();
			}
			recorder::closeProgram = false;
			
			//warmUp = true;
			while (!colorBuffer.empty()) {
				colorBuffer.pop_front();
			}

			while (!depthBuffer.empty()) {
				depthBuffer.pop_front();
			}
		}
		
		ImGui::Separator();
		ImGui::Text("image count: %d", currentImageCount);
		ImGui::Text("colorBuffer size: %d", colorBuffer.size());
		ImGui::Text("depthBuffer size: %d", depthBuffer.size());
		ImGui::Separator();

		if (ImGui::TreeNode("Debug")) {
			ImGui::Text("device count: %d", recorder::recorderDeviceCount);
			ImGui::Text("parameter set: %d", (bool)recorder::parameterSet);
			ImGui::Text("captureSuccess: %d", capturesuccess);
			ImGui::Text("Camera: %d, %d", color_width, color_height);
			ImGui::Text("camera aspect: %f", camera_aspect);
			ImGui::Text("Image: %f, %f", width_tmp, height_tmp);
			ImGui::Text("Window: %f, %f", ImGui::GetWindowWidth(), ImGui::GetWindowHeight());
			ImGui::Spacing();
			ImGui::Text("desired Image count: %d", desiredImageCount);
			ImGui::Text("internal Image count: %d", internalImageCount);

			ImGui::TreePop();
		}
		
		
		ImGui::End();
		//############################### Control Panel End ########################


		// Rendering
		ImGui::Render();
		int display_w, display_h;
		glfwGetFramebufferSize(window, &display_w, &display_h);
		glViewport(0, 0, display_w, display_h);
		glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
		glClear(GL_COLOR_BUFFER_BIT);
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		glfwSwapBuffers(window);
		glfwPollEvents();
			  
	}

	recorder::closeProgram = true;

	if(recorderThread.joinable())
		recorderThread.join();
	
	if(colorImageThread.joinable())
		colorImageThread.join();
	
	if(depthImageThread.joinable())
		depthImageThread.join();


	// Cleanup
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glDeleteTextures(1, &colorTexture);

	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}