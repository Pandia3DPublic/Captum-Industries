cmake_minimum_required(VERSION 3.1)
set (CMAKE_CXX_STANDARD 14)

#open3d necessary: export a few folder to parent scope in open3d and link against the stupid 3rd party build
#also in the cmaketext in 3rd party change the directories fucntion for tinygltf. othwerise returns way to many folder and max lenght for command line is surpassed
#for now all tinygltf include dirs are commented out

#make list of own source code
project(PandiaScanner LANGUAGES CXX CUDA)
file(GLOB srcfiles 
${PROJECT_SOURCE_DIR}/src/*.h   
${PROJECT_SOURCE_DIR}/src/*.cpp
${PROJECT_SOURCE_DIR}/src/*.cu
${PROJECT_SOURCE_DIR}/src/*/*.h   
${PROJECT_SOURCE_DIR}/src/*/*.cpp
${PROJECT_SOURCE_DIR}/src/*/*.cc
${PROJECT_SOURCE_DIR}/src/*/*.cu
${PROJECT_SOURCE_DIR}/src/cameras/ClientCamera/*.h
${PROJECT_SOURCE_DIR}/src/cameras/ClientCamera/*.cpp
${PROJECT_SOURCE_DIR}/src/Gui/imgui-docking/*.h
${PROJECT_SOURCE_DIR}/src/Gui/imgui-docking/*.cpp
${PROJECT_SOURCE_DIR}/src/Gui/imgui-docking/FileBrowser/*.h
${PROJECT_SOURCE_DIR}/src/Gui/imgui-docking/FileBrowser/*.cpp
)

#configure headers 
set(SHADER_PATH shader/)
set (datapath ${PROJECT_SOURCE_DIR}/data)
set (REFERENCEPATH ${PROJECT_SOURCE_DIR}/Testdata/GitTrackedData/)
configure_file (
  "${PROJECT_SOURCE_DIR}/src/cmakedefines.h.in"
  "${PROJECT_SOURCE_DIR}/src/cmakedefines.h"
)
  

#open3d stuff as build from source
message("\n Starting open3d stuff. \n")
add_subdirectory(Open3D)

include_directories(
    ${PROJECT_SOURCE_DIR}/Open3D/src
    ${PROJECT_SOURCE_DIR}/Open3D/3rdparty
    ${PROJECT_SOURCE_DIR}/Open3D/3rdparty/zlib #is for some reason not in 3RDPARTY_INCLUDE_DIRS
    ${3RDPARTY_INCLUDE_DIRS}
	${PROJECT_SOURCE_DIR}/src
	$ENV{CUDA_PATH_V10_1}/include
)

#need to export this in the open3d cmake to parent scope. stupid new system since 0.8
link_directories(${CMAKE_PROJECT_NAME} ${O3D_3RDPARTY_INSTALL_PREFIX}/lib)
#message("prefix " ${O3D_3RDPARTY_INSTALL_PREFIX})
#for python, even though we don't build open3d for it
include_directories(${PROJECT_SOURCE_DIR}/3rdparty/python/include)
link_directories(${PROJECT_SOURCE_DIR}/3rdparty/python/libs)

message("\n Ending open3d stuff. \n")

#opencv stuff
message("Finding opencv package:")
find_package(OpenCV HINTS ${PROJECT_SOURCE_DIR}/3rdparty/opencv REQUIRED)

#ceres stuff
message("\n Finding ceres package:")
set(EIGEN_PREFER_EXPORTED_EIGEN_CMAKE_CONFIGURATION TRUE)
find_package(CERES HINTS ${PROJECT_SOURCE_DIR}/3rdparty/ceres/ceres REQUIRED)

#protobuf stuff, dont know why all the seperate includes
set(Protobuf_DIR 3rdparty/protobuf)
include(${Protobuf_DIR}/cmake/protobuf-config.cmake)
include(${Protobuf_DIR}/cmake/protobuf-module.cmake)
include(${Protobuf_DIR}/cmake/protobuf-options.cmake)
include(${Protobuf_DIR}/cmake/protobuf-targets.cmake)
find_package(Protobuf HINTS ${Protobuf_DIR} REQUIRED)

#Kinect stuff
set(KINECT_LIBRARY_DIR ${PROJECT_SOURCE_DIR}\\3rdparty\\Kinect\\lib)
set(KINECT_DLL_DIR ${PROJECT_SOURCE_DIR}\\3rdparty\\Kinect\\bin)
set(KINECT_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}\\3rdparty\\Kinect\\include)
find_library(KINECT_FOUND k4a HINTS ${KINECT_LIBRARY_DIR} REQUIRED)

#realsense stuff
set(REALSENSE_LIBRARY_DIR ${PROJECT_SOURCE_DIR}\\3rdparty\\Realsense\\lib\\x64)          # used to find lib and lib includes
set(REALSENSE_DLL_DIR ${PROJECT_SOURCE_DIR}\\3rdparty\\Realsense\\bin\\x64)              # used for postbuild event
set(REALSENSE_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}\\3rdparty\\Realsense\\include)   # used for includes
find_library(REALSENSE2_FOUND realsense2 HINTS ${REALSENSE_LIBRARY_DIR} REQUIRED)

#structure stuff
set(STRUCTURE_LIBRARY_DIR ${PROJECT_SOURCE_DIR}\\3rdparty\\Structure\\lib)          # used to find lib and lib includes
set(STRUCTURE_DLL_DIR ${PROJECT_SOURCE_DIR}\\3rdparty\\Structure\\bin)              # used for postbuild event
set(STRUCTURE_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}\\3rdparty\\Structure\\include)   # used for includes
find_library(STRUCTURE_FOUND Structure HINTS "${STRUCTURE_LIBRARY_DIR}" REQUIRED)

#ffmpeg stuff
set(FFMPEG_LIBRARY_DIR ${PROJECT_SOURCE_DIR}/3rdparty/ffmpeg/lib)
set(FFMPEG_DLL_DIR ${PROJECT_SOURCE_DIR}/3rdparty/ffmpeg/bin)
set(FFMPEG_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/3rdparty/ffmpeg/include)


#this does not more than set some compiler flags. There is no real package
find_package(OpenMP QUIET)
if (OPENMP_FOUND)
    message(STATUS "Using installed OpenMP ${OpenMP_VERSION}. Setting compiler flags")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
    message(STATUS "OpenMP NOT found")
endif ()


#do target stuff
set(target_names realDeal tryouts tests recorder o3drecon) #cpp files with same name must be located in exe

message("\n #################### Entering Target Loop \n")

FOREACH(target_name ${target_names})
	add_executable( ${target_name} exe/${target_name}.cpp ${srcfiles})
	#target_link_libraries( ${target_name} Open3D)	#open3dcuda contains everything
	target_link_libraries( ${target_name} Open3DCuda)	
	#target_precompile_headers(${target_name} PRIVATE src/pch.h)

	#added lib include dirs and links
	target_include_directories(${target_name} PUBLIC ${OpenCV_INCLUDE_DIRS})
	target_link_libraries( ${target_name} ${OpenCV_LIBS})
	target_include_directories(${target_name} PUBLIC ${CERES_INCLUDE_DIRS})
	target_link_libraries(${target_name} ${CERES_LIBRARIES})
	target_include_directories(${target_name} PUBLIC ${Protobuf_INCLUDE_DIRS})
	target_link_libraries(${target_name} ${Protobuf_LIBRARIES})
	target_include_directories(${target_name} PUBLIC ${KINECT_INCLUDE_DIRS})
	target_link_libraries( ${target_name} "${KINECT_LIBRARY_DIR}/k4a.lib")
	target_include_directories(${target_name} PUBLIC ${REALSENSE_INCLUDE_DIRS})
	target_link_libraries( ${target_name} "${REALSENSE_LIBRARY_DIR}/realsense2.lib")
	target_include_directories(${target_name} PUBLIC ${STRUCTURE_INCLUDE_DIRS})
	target_link_libraries( ${target_name} "${STRUCTURE_LIBRARY_DIR}/Structure.lib")
	target_include_directories(${target_name} PUBLIC ${FFMPEG_INCLUDE_DIRS})
	target_link_libraries( ${target_name} "${FFMPEG_LIBRARY_DIR}/avcodec.lib" "${FFMPEG_LIBRARY_DIR}/avdevice.lib" "${FFMPEG_LIBRARY_DIR}/avfilter.lib" "${FFMPEG_LIBRARY_DIR}/avformat.lib" "${FFMPEG_LIBRARY_DIR}/avutil.lib" "${FFMPEG_LIBRARY_DIR}/postproc.lib" "${FFMPEG_LIBRARY_DIR}/swresample.lib" "${FFMPEG_LIBRARY_DIR}/swscale.lib")

	if(MSVC AND ${MSVC_VERSION} GREATER_EQUAL 1915)
	# You must acknowledge that you understand MSVC resolved a byte alignment issue in this compiler
	# We get this due to using Eigen objects and allocating those objects with make_shared
		target_compile_definitions( ${target_name} PRIVATE _ENABLE_EXTENDED_ALIGNED_STORAGE )
	endif()
	set_target_properties( ${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

ENDFOREACH()


########################### compiler definitions ##################
message("\n Developertools on \n")
target_compile_definitions(realDeal PRIVATE DEVELOPERTOOLS)



####################### post build events, mostly copy############
list(GET target_names 0 firstname) #get the first member of target names
message("Post build dll copy events. NOTE: Only executed for realdeal \n")

#copy opencv dll into build folder
add_custom_command(TARGET ${firstname} POST_BUILD       
COMMAND ${CMAKE_COMMAND} -E copy_if_different 
    "${PROJECT_SOURCE_DIR}/3rdparty/opencv/x64/vc15/bin/opencv_world410.dll"     
     $<TARGET_FILE_DIR:${firstname}>)               
#python
add_custom_command(TARGET ${firstname} POST_BUILD     
COMMAND ${CMAKE_COMMAND} -E copy_if_different  
    "${PROJECT_SOURCE_DIR}/3rdparty/Python/python36.dll"     
     $<TARGET_FILE_DIR:${firstname}>)            
#protobuf 
add_custom_command(TARGET ${firstname} POST_BUILD        
COMMAND ${CMAKE_COMMAND} -E copy_if_different             
    "${PROJECT_SOURCE_DIR}/3rdparty/protobuf/bin/protoc.exe"                           
 	 $<TARGET_FILE_DIR:${firstname}>)    
#Kinect 
add_custom_command(TARGET ${firstname} POST_BUILD        
COMMAND ${CMAKE_COMMAND} -E copy_if_different             
    "${KINECT_DLL_DIR}/k4a.dll"                           
 	 $<TARGET_FILE_DIR:${firstname}>)   
add_custom_command(TARGET ${firstname} POST_BUILD        
COMMAND ${CMAKE_COMMAND} -E copy_if_different             
    "${KINECT_DLL_DIR}/depthengine_2_0.dll"                           
 	 $<TARGET_FILE_DIR:${firstname}>)   
#realsense 
add_custom_command(TARGET ${firstname} POST_BUILD        
COMMAND ${CMAKE_COMMAND} -E copy_if_different             
    "${REALSENSE_DLL_DIR}/realsense2.dll"                           
 	 $<TARGET_FILE_DIR:${firstname}>)          
#structure 
add_custom_command(TARGET ${firstname} POST_BUILD        
COMMAND ${CMAKE_COMMAND} -E copy_if_different             
    "${STRUCTURE_DLL_DIR}/Structure.dll"                           
 	 $<TARGET_FILE_DIR:${firstname}>)      
#ffmpeg 
add_custom_command(TARGET ${firstname} POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_if_different
	"${FFMPEG_DLL_DIR}/avcodec-58.dll"
	$<TARGET_FILE_DIR:${firstname}>)
add_custom_command(TARGET ${firstname} POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_if_different
	"${FFMPEG_DLL_DIR}/avdevice-58.dll"
	$<TARGET_FILE_DIR:${firstname}>)
add_custom_command(TARGET ${firstname} POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_if_different
	"${FFMPEG_DLL_DIR}/avfilter-7.dll"
	$<TARGET_FILE_DIR:${firstname}>)
add_custom_command(TARGET ${firstname} POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_if_different
	"${FFMPEG_DLL_DIR}/avformat-58.dll"
	$<TARGET_FILE_DIR:${firstname}>)
add_custom_command(TARGET ${firstname} POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_if_different
	"${FFMPEG_DLL_DIR}/avutil-56.dll"
	$<TARGET_FILE_DIR:${firstname}>)
add_custom_command(TARGET ${firstname} POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_if_different
	"${FFMPEG_DLL_DIR}/postproc-55.dll"
	$<TARGET_FILE_DIR:${firstname}>)
add_custom_command(TARGET ${firstname} POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_if_different
	"${FFMPEG_DLL_DIR}/swresample-3.dll"
	$<TARGET_FILE_DIR:${firstname}>)
add_custom_command(TARGET ${firstname} POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_if_different
	"${FFMPEG_DLL_DIR}/swscale-5.dll"
	$<TARGET_FILE_DIR:${firstname}>)


#copy resources into build folder!
#copy shaders 
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_PATH})
file(GLOB shaders ${PROJECT_SOURCE_DIR}/Resources/shader_outside/*.txt)
foreach( shader ${shaders} )
configure_file(${shader} ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_PATH} COPYONLY )
endforeach( shader )

#copy textures 
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/textures)
file(GLOB textures ${PROJECT_SOURCE_DIR}/Resources/textures_outside/*.png)
foreach( texture ${textures} )
configure_file(${texture} ${CMAKE_CURRENT_BINARY_DIR}/textures COPYONLY )
endforeach( texture )

#copy the config.txt
configure_file(config.txt ${CMAKE_CURRENT_BINARY_DIR} COPYONLY )


#copy language files 
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/languageFolder)
file(GLOB languageFiles ${PROJECT_SOURCE_DIR}/Resources/language_outside/*.txt)
foreach( languageFile ${languageFiles} )
configure_file(${languageFile} ${CMAKE_CURRENT_BINARY_DIR}/languageFolder COPYONLY )
endforeach( languageFile )



