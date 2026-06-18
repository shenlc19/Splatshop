#include "MogeGsPredictor.h"

#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

int run_moge_gs_pipeline_integrated(
	const std::string& mogeModelPath,
	const std::string& gsModelPath,
	const std::string& imagePath,
	int64_t numTokens,
	const std::string& outputPlyPath,
	int cudaDeviceId,
	int resizeTo,
	MogeGsPredictorTimings* timings
);

namespace {

using Clock = std::chrono::steady_clock;

std::string env_or_default(const char* name, std::string fallback){
	const char* value = std::getenv(name);
	if(value && value[0] != '\0'){
		return value;
	}
	return fallback;
}

double elapsed_ms(Clock::time_point start, Clock::time_point end){
	return std::chrono::duration<double, std::milli>(end - start).count();
}

#ifdef _WIN32
std::wstring widen_utf8(const std::string& path){
	int count = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
	if(count <= 0){
		return std::wstring(path.begin(), path.end());
	}

	std::wstring wide(count - 1, L'\0');
	MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wide.data(), count);
	return wide;
}

void prepend_process_path(const std::string& directory){
	if(directory.empty() || !std::filesystem::exists(directory)){
		return;
	}

	std::wstring directoryW = widen_utf8(directory);
	SetDllDirectoryW(directoryW.c_str());

	DWORD pathChars = GetEnvironmentVariableW(L"PATH", nullptr, 0);
	std::wstring oldPath;
	if(pathChars > 0){
		oldPath.resize(pathChars - 1);
		GetEnvironmentVariableW(L"PATH", oldPath.data(), pathChars);
	}

	std::wstring newPath = directoryW;
	if(!oldPath.empty()){
		newPath += L";";
		newPath += oldPath;
	}
	SetEnvironmentVariableW(L"PATH", newPath.c_str());
}

void configure_moge_dll_search_path(){
	prepend_process_path(env_or_default("TORCH_CUDA_DLL_DIR", R"(D:\miniconda\envs\moge\Lib\site-packages\torch\lib)"));
	prepend_process_path(env_or_default("SPLATSHOP_TORCH_CUDA_DLL_DIR", R"(D:\miniconda\envs\moge\Lib\site-packages\torch\lib)"));
	prepend_process_path(env_or_default("ONNXRUNTIME_ROOT", R"(E:\libs\onnxruntime-win-x64-gpu-1.23.2)") + std::string(R"(\lib)"));
	prepend_process_path(R"(C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin)");
}
#else
void configure_moge_dll_search_path(){}
#endif

} // namespace

int MogeGsPredictor::predictToPly(
	const std::string& imagePath,
	const std::string& outputPlyPath,
	MogeGsPredictorTimings* timings
){
	MogeGsPredictorTimings localTimings;
	MogeGsPredictorTimings& t = timings ? *timings : localTimings;
	t = MogeGsPredictorTimings{};
	const auto totalBegin = Clock::now();
	try{
		std::string mogeModelPath = env_or_default(
			"SPLATSHOP_MOGE_CORE_MODEL",
			R"(E:\projects\MoGe\workspace\implicit_head_debug\checkpoint\00050000_core_dynamic_tokens.onnx)"
		);
		std::string gsModelPath = env_or_default(
			"SPLATSHOP_MOGE_GS_MODEL",
			R"(E:\projects\MoGe\workspace\infinidepth_gs_raw_dynamic.onnx)"
		);
		int64_t numTokens = std::stoll(env_or_default("SPLATSHOP_MOGE_NUM_TOKENS", "1800"));
		int cudaDeviceId = std::stoi(env_or_default("SPLATSHOP_MOGE_CUDA_DEVICE", "0"));
		int resizeTo = std::stoi(env_or_default("SPLATSHOP_MOGE_RESIZE_TO", "512"));
		configure_moge_dll_search_path();

		if(!std::filesystem::exists(mogeModelPath)){
			std::cerr << "MoGe GS prediction skipped: MoGe model not found: " << mogeModelPath << "\n";
			return -1;
		}
		if(!std::filesystem::exists(gsModelPath)){
			std::cerr << "MoGe GS prediction skipped: GS model not found: " << gsModelPath << "\n";
			return -1;
		}
		if(!std::filesystem::exists(imagePath)){
			std::cerr << "MoGe GS prediction skipped: image not found: " << imagePath << "\n";
			return -1;
		}

		std::filesystem::path outputPath(outputPlyPath);
		if(outputPath.has_parent_path()){
			std::filesystem::create_directories(outputPath.parent_path());
		}

		std::cout << "MoGe GS prediction started: " << outputPlyPath << "\n";
		int result = run_moge_gs_pipeline_integrated(
			mogeModelPath,
			gsModelPath,
			imagePath,
			numTokens,
			outputPlyPath,
			cudaDeviceId,
			resizeTo,
			&t
		);
		t.totalMs = elapsed_ms(totalBegin, Clock::now());
		if(result == 0){
			std::cout << "MoGe GS prediction saved: " << outputPlyPath << "\n";
			std::cout << "MoGe GS prediction timings: total " << t.totalMs
				<< " ms, model load " << t.modelLoadMs
				<< " ms, preprocess " << t.preprocessMs
				<< " ms, MoGe image-grid ONNX " << t.mogeImageGridOnnxMs
				<< " ms, MoGe postprocess/sparse coords " << t.mogePostprocessAndSparseCoordsMs
				<< " ms, MoGe sparse ONNX " << t.mogeSparseOnnxMs
				<< " ms, GS raw ONNX " << t.gsRawOnnxMs
				<< " ms, adapter/filter " << t.adapterAndFilterMs
				<< " ms, result save " << t.resultSaveMs << " ms\n";
		}else{
			std::cerr << "MoGe GS prediction failed with exit code " << result << ".\n";
		}
		return result;
	}catch(const std::exception& e){
		t.totalMs = elapsed_ms(totalBegin, Clock::now());
		std::cerr << "MoGe GS prediction failed: " << e.what() << "\n";
		return 1;
	}
}
