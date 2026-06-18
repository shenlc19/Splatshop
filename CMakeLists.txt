cmake_minimum_required(VERSION 3.22)

project(SplatEditor)
set(CMAKE_CXX_STANDARD 23)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)


# add_subdirectory(./libs/openvr-master)

include(cmake/common.cmake)

##############################################
## SUBPROJECT: Splat Editor
##############################################

set(TARGET_NAME SplatEditor) 

file(GLOB SplatEditor_SOURCES CONFIGURE_DEPENDS 
	src/*.*
	src/actions/*.*
	src/GPUPrefixSums/*.*
	src/GPUSorting/*.*
	src/gui/*.*
	src/inpaint/*.*
	# Temporarily disabled while maintaining only the Lama inpaint module.
	# src/moge/*.*
	src/render/*.*
	src/scene/*.*
	src/update/*.*
	src/loader/*.*
	src/writer/*.*
	include/*.*
)

if (WIN32)
	list(APPEND SplatEditor_SOURCES
		src/win32/GLRenderer_win32.cpp
		src/win32/OpenVRHelper.cpp
	)
elseif (UNIX)
	list(APPEND SplatEditor_SOURCES
		src/unix/GLRenderer_linux.cpp
	)
endif (WIN32)

add_executable(${TARGET_NAME} ${SplatEditor_SOURCES})

if (WIN32)
	# set workdir to ../$(SolutionDir), so that .cu files can be monitored/reloaded/compiled
	set_property(TARGET ${TARGET_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$(SolutionDir)..")
endif (WIN32)

target_include_directories(${TARGET_NAME} PRIVATE 
	./include
	./src
)

# add some third-party libraries
ADD_CUDA(${TARGET_NAME})
ADD_OPENGL(${TARGET_NAME})
ADD_GLM(${TARGET_NAME})
ADD_IMGUI(${TARGET_NAME})
ADD_IMPLOT(${TARGET_NAME})
ADD_IMGUIZMO(${TARGET_NAME})
target_include_directories(${TARGET_NAME} PRIVATE ./include)
target_include_directories(${TARGET_NAME} PRIVATE ./libs)

# Temporarily disabled while maintaining only the Lama inpaint module.
# set(MOGE_GS_PIPELINE_SOURCE "E:/projects/MoGe/cpp/moge_gs_pipeline.cpp" CACHE FILEPATH "MoGe GS C++ pipeline source")
# if(NOT EXISTS "${MOGE_GS_PIPELINE_SOURCE}")
# 	message(WARNING "MoGe GS pipeline source not found: ${MOGE_GS_PIPELINE_SOURCE}")
# endif()
# target_compile_definitions(${TARGET_NAME} PRIVATE SPLATSHOP_MOGE_GS_PIPELINE_SOURCE="${MOGE_GS_PIPELINE_SOURCE}")

if(WIN32 AND NOT OpenCV_DIR AND EXISTS "E:/libs/opencv/build")
	set(OpenCV_DIR "E:/libs/opencv/build" CACHE PATH "Path to OpenCV CMake config" FORCE)
endif()
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

set(OPENCV_RUNTIME_DIR "" CACHE PATH "Path to OpenCV runtime DLL directory")
if(WIN32 AND NOT OPENCV_RUNTIME_DIR AND EXISTS "E:/libs/opencv/build/x64/vc16/bin")
	set(OPENCV_RUNTIME_DIR "E:/libs/opencv/build/x64/vc16/bin" CACHE PATH "Path to OpenCV runtime DLL directory" FORCE)
endif()

set(ONNXRUNTIME_ROOT "$ENV{ORT_ROOT}" CACHE PATH "Path to ONNX Runtime C/C++ SDK root")
if(NOT ONNXRUNTIME_ROOT)
	set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "Path to ONNX Runtime C/C++ SDK root" FORCE)
endif()
if(WIN32 AND NOT ONNXRUNTIME_ROOT AND EXISTS "E:/libs/onnxruntime-win-x64-gpu-1.23.2")
	set(ONNXRUNTIME_ROOT "E:/libs/onnxruntime-win-x64-gpu-1.23.2" CACHE PATH "Path to ONNX Runtime C/C++ SDK root" FORCE)
endif()

set(MOGE_CPP_DIR "E:/projects/MoGe/cpp" CACHE PATH "MoGe C++ export API include directory")
set(MOGE_GS_EXPORT_LIB "E:/projects/MoGe/build/moge_gs_export.lib" CACHE FILEPATH "MoGe GS export static library")

target_include_directories(${TARGET_NAME} PRIVATE "${MOGE_CPP_DIR}")

target_link_libraries(${TARGET_NAME} "${MOGE_GS_EXPORT_LIB}")

set(TORCH_CUDA_DLL_DIR "$ENV{TORCH_CUDA_DLL_DIR}" CACHE PATH "Path to PyTorch CUDA/cuDNN runtime DLL directory")
if(WIN32 AND NOT TORCH_CUDA_DLL_DIR AND EXISTS "D:/miniconda/envs/moge/Lib/site-packages/torch/lib")
	set(TORCH_CUDA_DLL_DIR "D:/miniconda/envs/moge/Lib/site-packages/torch/lib" CACHE PATH "Path to PyTorch CUDA/cuDNN runtime DLL directory" FORCE)
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
	PATHS "${ONNXRUNTIME_ROOT}/include" NO_DEFAULT_PATH)
find_library(ONNXRUNTIME_LIBRARY onnxruntime
	PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
	message(FATAL_ERROR "Could not find ONNX Runtime headers/library under ${ONNXRUNTIME_ROOT}")
endif()

target_include_directories(${TARGET_NAME} PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(${TARGET_NAME} ${OpenCV_LIBS} "${ONNXRUNTIME_LIBRARY}")

if(WIN32)
	set_property(TARGET ${TARGET_NAME} PROPERTY VS_DEBUGGER_ENVIRONMENT
		"PATH=${OPENCV_RUNTIME_DIR}\\;${ONNXRUNTIME_ROOT}/lib\\;${TORCH_CUDA_DLL_DIR}\\;C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin\\;%PATH%")

	add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll" "$(OutDir)"
		COMMENT "POST BUILD: copying onnxruntime.dll")

	foreach(ORT_PROVIDER_DLL
		onnxruntime_providers_shared.dll
		onnxruntime_providers_cuda.dll
		onnxruntime_providers_tensorrt.dll
	)
		if(EXISTS "${ONNXRUNTIME_ROOT}/lib/${ORT_PROVIDER_DLL}")
			add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
				COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ONNXRUNTIME_ROOT}/lib/${ORT_PROVIDER_DLL}" "$(OutDir)"
				COMMENT "POST BUILD: copying ${ORT_PROVIDER_DLL}")
		endif()
	endforeach()
endif()

# I give up, let's just include OpenVR like this.
target_include_directories(${TARGET_NAME} PRIVATE ./libs/openvr/headers)
if (WIN32)
	target_link_libraries(${TARGET_NAME} ../libs/openvr/lib/win64/openvr_api)
	add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy "$(SolutionDir)../libs/openvr/bin/win64/openvr_api.dll" "$(OutDir)"
		COMMENT "POST BUILD: copying openvr folder")
else()
	add_compile_definitions(NO_OPENVR)
endif (WIN32)

if (UNIX)
	find_package(TBB REQUIRED)
	target_link_libraries(${TARGET_NAME} TBB::tbb)
endif (UNIX)

##############################################
## SOME FINAL STUFF
##############################################

# set startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT SplatEditor)
