#pragma once
#include "Gui/imgui-docking/imgui.h"
#include "Gui/imgui-docking/FileBrowser/ImGuiFileBrowser.h"
#include <GL/glew.h>
#include "tinygltf/stb_image.h"
#include <string>
#include "core/Model.h"
#include "configvars.h"
#include "../postprocessing/postprocessing.h"
#include <atomic>
#include <assert.h>
#include <chrono>
#include "integrate.h"
#include "Gui/PandiaView.h"
#include "cameras/CameraThreadandInit.h"
#include "core/threadvars.h"


//contains util functions and variables!
//todo maybe make this singleton class
namespace PandiaGui {
	
	extern int shadertyp; //-1 is non, 0 is simple, 1 pandia, 2 is o3d
	extern bool menu_showCameraPath; //only indicates menu
	extern bool cameraPathBuild; //indicates if mesh needs to be build

	//menu, saving and loading flags
	extern bool menu_InfoMode; //user can hover over info icons
	extern bool menu_DemoWindow; //shows imgui demo window
	extern bool menu_DebugWindow;
	extern bool menu_showCameraPath;
	extern bool menu_ShaderDefault;
	extern bool menu_ShaderWireFrame;
	extern bool menu_ShaderNeutral;
	extern bool menu_AboutUs;
	
	extern bool loadDataFlag; //if true reads data from harddrive
	extern bool saveImageFlag;
	extern bool saveTrajectoryFlag;
	extern int loadOptionSwitch; //status int. Status 0 is load mesh, status 1 is load images
	extern bool disableInput;
	extern bool nonemptyBuffer;

	//window flags
	extern const ImGuiWindowFlags window_flags; //flags for whole application window
	extern const ImGuiWindowFlags subWindow_flags; //flags for control-/ and info window


	//colors 
	extern const ImVec4 col_darkgreen;
	extern const ImVec4 col_green;
	extern const ImVec4 col_lime;
	extern const ImVec4 col_brightGreen;
	extern const ImVec4 col_yellow;
	extern const ImVec4 col_orange;
	extern const ImVec4 col_gray;
	extern const ImVec4 col_green_1;
	extern const ImVec4 col_green_2;
	extern const ImVec4 col_green_3;
	extern const ImVec4 col_red_1;
	extern const ImVec4 col_red_2;
	extern const ImVec4 col_red_3;


	//filedialog variables (pathes for loading and saving mesh/images)
	extern string saveMeshPath;
	extern string selectedMeshPath;
	extern string saveImagePath;
	extern string saveTrajectoriesPath;
	extern imgui_addons::ImGuiFileBrowser file_dialogLoadMesh;
	extern imgui_addons::ImGuiFileBrowser file_dialogLoadImage;
	extern imgui_addons::ImGuiFileBrowser file_dialogSaveImage;
	extern imgui_addons::ImGuiFileBrowser file_dialogSaveMesh;
	extern bool fileDialogOpenNow;


	//image variables
	extern GLuint infoSymbol_texture;
	extern GLuint logo_texture;
	extern GLuint logoWhite_texture;
	extern GLuint background_image;
	extern int logo_widht;
	extern int logo_height;
	extern int logoWhite_width;
	extern int logoWhite_height;
	extern float logo_aspect;


	//Text language variables
	extern int languageChoice;
	//menu
	extern string lan_settings;
	extern string lan_language;
	extern string lan_english;
	extern string lan_german;
	extern string lan_noShader;
	extern string lan_simpleShader;
	extern string lan_metalPhongShader;
	extern string lan_wireframe;
	extern string lan_help;
	extern string lan_InfoModeText;
	extern string lan_cameraPath;
	extern string lan_raycasting;
	extern string lan_aboutUs;
	//control window
	extern string lan_loadData;
	extern string lan_loadDataInfo;
	extern string lan_loadMesh;
	extern string lan_ButtonLoadMesh;
	extern string lan_loadMeshInfo;
	extern string lan_loadImages;
	extern string lan_loadImagesInfo;
	extern string lan_loadMeshDirectory;
	extern string lan_loadMeshDirectoryInfo;
	extern string lan_loadMeshTextHeader;
	extern string lan_loadMeshTextHeaderInfo;
	extern string lan_loadImagesDirectory;
	extern string lan_loadImagesDirectoryInfo;
	extern string lan_loadImagesTextHeader;
	extern string lan_loadImagesTextHeaderInfo;
	//GUI-States
	extern string lan_InfoAzure;
	extern string lan_InfoRealsense;
	extern string lan_InfoDataCam;
	extern string lan_ButtonStart;
	extern string lan_AnimationText;
	extern string lan_ButtonStartDummy;
	extern string lan_ButtonPause;
	extern string lan_ButtonResume;
	extern string lan_ButtonSaveMesh;
	extern string lan_ButtonClearMesh;
	extern string lan_BoxSaveImages;
	extern string lan_InfoImageSave;
	extern string lan_DirectoryImageSave;
	extern string lan_DirectoryImageSaveInfo;
	extern string lan_ButtonSaveImages;
	extern string lan_TextPostProcessing;
	extern string lan_textPostProcessingInfo;
	extern string lan_voxelLength;
	extern string lan_voxelLenghtInfo;
	extern string lan_meshReduction;
	extern string lan_meshReductionInfo;
	extern string lan_ButtonStartPostProcessing;
	//GUI-Popups
	extern string lan_warmUp;
	extern string lan_voxelLength;
	extern string lan_cancel;
	extern string lan_meshReduction;
	extern string lan_noConnection;
	extern string lan_cameraLegend;


	//window variables
	extern ImVec2 topleft;
	extern ImVec2 renderWindowSize;
	extern ImVec2 buttonsize;


	//miscellaneous variables
	extern char animationBuffer[25]; //small animation for "searching camera"
	extern shared_ptr<geometry::TriangleMesh> cpumesh; //loaded mesh will be renderd in this variable
	extern ImVec2 absolutWindowPos; // variable for GuiBlockItem()
	extern bool showRaycast;


//#################################### functions ################################

	void initPandiaGui();
	

	void InfoMarker(const char* desc, bool& enabled, GLuint texture);
	
	//must be called before text-sized entity
	template<typename var>
	void GuiBlockItem(var& disable);

	//must be called before any ImGui-Widget
	//Spawns semi-transparent Window to prevent User-interaction
	//NOTE: keyboard shortcuts will circumvent this
	template<typename var>
	void GuiBlockItem(var& disable, ImVec2 size);

	std::string getTopDirectoryPath(std::string& path);

	bool LoadTextureFromFile(const char* filename, GLuint* out_texture, int* out_width, int* out_height);

	//postprocessing thread
	void PostProcessingThread(Model& m);

	void setGuiLanguage(int choice);
	void assignTranslation(string& name, string& translation);

	//INI-functions
	void createPandiaGuiIni();
	void readPandiaGuiIni();
	void assignIniValues(string& s, int i);

	//renders WindowsElements
	

	//
	void handle_FileDialogs(Model& m, shared_ptr<GLGeometry>& mesh, GLGeometry*& pointer, GLuint& vertex, GLuint& color, GLuint& normal);



#ifdef DEVELOPERTOOLS
	void showDebugWindow(Model& m, PandiaView& v);
#endif // DEVELOPERTOOLS

	enum PandiaTimeUnit {hours, minutes, seconds, milliseconds};

	class PandiaClock {
	public:
		PandiaClock(bool start = true);
		double getDuration(PandiaTimeUnit unit = PandiaTimeUnit::seconds);
		string getClock();
		void saveTimestamp();
		void reset();
		void start();
		void stop();
		vector < std::chrono::time_point<std::chrono::steady_clock> > timestamp;

		//saves timePasssed to add it later to start where timePassed left off
		double offset;
		
		// saves time interval in which user starts and stops reconrun
		double timePassed;

	private:
		std::chrono::time_point<std::chrono::steady_clock> begin, current;
		std::chrono::duration<double> duration;
		//PandiaTimeUnit unit_;
		bool isRunning;
		string clock;
	};



} //namespace PandiaGui