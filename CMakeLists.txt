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
	src/infinidepth/*.*
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

set(INFINIDEPTH_CUDA_SOURCES
	${CMAKE_CURRENT_SOURCE_DIR}/src/infinidepth/infinidepth_depthsensor_kernels.cu
	${CMAKE_CURRENT_SOURCE_DIR}/src/infinidepth/infinidepth_gs_sparse.cu
	${CMAKE_CURRENT_SOURCE_DIR}/src/infinidepth/infinidepth_input_processing_kernels.cu
)
get_filename_component(SPLATSHOP_CUDA_BIN_DIR "${CUDAToolkit_NVCC_EXECUTABLE}" DIRECTORY)
get_filename_component(SPLATSHOP_CUDA_ROOT_DIR "${SPLATSHOP_CUDA_BIN_DIR}" DIRECTORY)
set(SPLATSHOP_CUDA_INCLUDE_DIR "${SPLATSHOP_CUDA_ROOT_DIR}/include")
set(INFINIDEPTH_CUDA_OBJECTS)
foreach(INFINIDEPTH_CUDA_SOURCE ${INFINIDEPTH_CUDA_SOURCES})
	get_filename_component(INFINIDEPTH_CUDA_NAME ${INFINIDEPTH_CUDA_SOURCE} NAME_WE)
	set(INFINIDEPTH_CUDA_OBJECT ${CMAKE_CURRENT_BINARY_DIR}/infinidepth_cuda/${INFINIDEPTH_CUDA_NAME}.obj)
	add_custom_command(
		OUTPUT ${INFINIDEPTH_CUDA_OBJECT}
		COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/infinidepth_cuda
		COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
			-c ${INFINIDEPTH_CUDA_SOURCE}
			-o ${INFINIDEPTH_CUDA_OBJECT}
			-std=c++17
			-Xcompiler=/MD
			-I${CMAKE_CURRENT_SOURCE_DIR}/src/infinidepth
			-I${SPLATSHOP_CUDA_INCLUDE_DIR}
		DEPENDS ${INFINIDEPTH_CUDA_SOURCE}
		VERBATIM
	)
	list(APPEND INFINIDEPTH_CUDA_OBJECTS ${INFINIDEPTH_CUDA_OBJECT})
endforeach()
target_sources(${TARGET_NAME} PRIVATE ${INFINIDEPTH_CUDA_OBJECTS})

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
target_compile_definitions(${TARGET_NAME} PRIVATE ORT_API_MANUAL_INIT)
target_link_libraries(${TARGET_NAME} ${OpenCV_LIBS} "${ONNXRUNTIME_LIBRARY}")

if(WIN32)
	target_link_libraries(${TARGET_NAME} windowscodecs ole32)
	target_link_options(${TARGET_NAME} PRIVATE "/DELAYLOAD:onnxruntime.dll")
	target_link_libraries(${TARGET_NAME} delayimp)
endif()

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
