#include "InfiniDepthGsPredictor.h"

#include "infinidepth_depthsensor_onnx.h"
#include "infinidepth_gs_pipeline.h"
#include "infinidepth_input_processing.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

#define CUDA_CHECK(expr) \
	do { \
		cudaError_t err = (expr); \
		if(err != cudaSuccess){ \
			throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
		} \
	} while(0)

std::string env_or_default(const char* name, std::string fallback){
	const char* value = std::getenv(name);
	if(value && value[0] != '\0'){
		return value;
	}
	return fallback;
}

int env_int_or_default(const char* name, int fallback){
	const char* value = std::getenv(name);
	return value && value[0] != '\0' ? std::stoi(value) : fallback;
}

int64_t env_i64_or_default(const char* name, int64_t fallback){
	const char* value = std::getenv(name);
	return value && value[0] != '\0' ? std::stoll(value) : fallback;
}

bool env_bool_or_default(const char* name, bool fallback){
	const char* value = std::getenv(name);
	if(!value || value[0] == '\0'){
		return fallback;
	}
	std::string text = value;
	return text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON";
}

double elapsed_ms(Clock::time_point start, Clock::time_point end){
	return std::chrono::duration<double, std::milli>(end - start).count();
}

#ifdef _WIN32
std::wstring widen_utf8(const std::string& text){
	if(text.empty()){
		return {};
	}
	int size = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, nullptr, 0);
	if(size <= 0){
		throw std::runtime_error("Failed to convert path to UTF-16.");
	}
	std::wstring wide(static_cast<size_t>(size), L'\0');
	MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, wide.data(), size);
	if(!wide.empty() && wide.back() == L'\0'){
		wide.pop_back();
	}
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

void configure_dll_search_path(){
	prepend_process_path(env_or_default("INFINIDEPTH_ONNXRUNTIME_DLL_DIR", R"(E:\libs\onnxruntime-win-x64-gpu-1.23.2\lib)"));
	prepend_process_path(env_or_default("INFINIDEPTH_CUDA_DLL_DIR", R"(C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin)"));
	prepend_process_path(env_or_default("INFINIDEPTH_TORCH_DLL_DIR", R"(D:\miniconda\envs\moge\Lib\site-packages\torch\lib)"));
}
#else
void configure_dll_search_path(){}
#endif

struct SessionKey{
	std::string encoderModelPath;
	std::string decoderModelPath;
	std::string gsModelPath;
	int cudaDeviceId = 0;
	int64_t chunkSize = 100000;
	int64_t samplePointNum = 2000000;
	bool disableOptimizations = true;
	std::string cudnnConvAlgoSearch = "HEURISTIC";
	bool cudnnConvUseMaxWorkspace = false;
	bool cudaUseTf32 = true;

	bool operator==(const SessionKey& other) const{
		return encoderModelPath == other.encoderModelPath
			&& decoderModelPath == other.decoderModelPath
			&& gsModelPath == other.gsModelPath
			&& cudaDeviceId == other.cudaDeviceId
			&& chunkSize == other.chunkSize
			&& samplePointNum == other.samplePointNum
			&& disableOptimizations == other.disableOptimizations
			&& cudnnConvAlgoSearch == other.cudnnConvAlgoSearch
			&& cudnnConvUseMaxWorkspace == other.cudnnConvUseMaxWorkspace
			&& cudaUseTf32 == other.cudaUseTf32;
	}
};

struct RunOptions{
	int inputHeight = 768;
	int inputWidth = 1024;
	int promptSamples = 1500;
	bool enableDepthNoiseFilter = false;
};

struct CachedInfiniDepthSession{
	SessionKey key;
	infinidepth::InfiniDepthDepthSensorOnnx depthSensor;
	infinidepth::InfiniDepthGsPipeline gsPipeline;

	CachedInfiniDepthSession(const SessionKey& sessionKey)
		: key(sessionKey),
		  depthSensor(sessionKey.encoderModelPath, sessionKey.decoderModelPath, make_depth_options(sessionKey)),
		  gsPipeline(sessionKey.gsModelPath, make_gs_options(sessionKey))
	{
	}

	static infinidepth::RuntimeOptions make_depth_options(const SessionKey& key){
		infinidepth::RuntimeOptions options;
		options.device_id = key.cudaDeviceId;
		options.decoder_chunk_size = key.chunkSize;
		options.disable_graph_optimizations = key.disableOptimizations;
		options.cudnn_conv_algo_search = key.cudnnConvAlgoSearch;
		options.cudnn_conv_use_max_workspace = key.cudnnConvUseMaxWorkspace;
		options.cuda_use_tf32 = key.cudaUseTf32;
		return options;
	}

	static infinidepth::GsPipelineOptions make_gs_options(const SessionKey& key){
		infinidepth::GsPipelineOptions options;
		options.device_id = key.cudaDeviceId;
		options.disable_graph_optimizations = key.disableOptimizations;
		options.cudnn_conv_algo_search = key.cudnnConvAlgoSearch;
		options.cudnn_conv_use_max_workspace = key.cudnnConvUseMaxWorkspace;
		options.cuda_use_tf32 = key.cudaUseTf32;
		return options;
	}

	void run(
		const RunOptions& options,
		const std::string& imagePath,
		const std::string& depthPath,
		const std::string& cameraJsonPath,
		const std::string& outputPlyPath,
		InfiniDepthGsTimings& timings
	){
		const auto inputBegin = Clock::now();
		infinidepth::InputProcessingOptions inputOptions;
		inputOptions.image_path = imagePath;
		inputOptions.depth_path = depthPath;
		inputOptions.camera_json_path = cameraJsonPath;
		inputOptions.target_height = options.inputHeight;
		inputOptions.target_width = options.inputWidth;
		inputOptions.prompt_samples = options.promptSamples;
		inputOptions.enable_depth_noise_filter = options.enableDepthNoiseFilter;
		infinidepth::ProcessedGsInputsCuda processed = infinidepth::ProcessRawGsInputsCuda(inputOptions);
		timings.inputProcessingMs = elapsed_ms(inputBegin, Clock::now());

		try{
			const auto depthBegin = Clock::now();
			infinidepth::InferenceForGsInputs depthInputs;
			depthInputs.image = processed.image;
			depthInputs.intrinsics = processed.intrinsics;
			depthInputs.gt_depth = processed.gt;
			depthInputs.gt_depth_mask = processed.gt_mask;
			depthInputs.prompt_depth = processed.prompt;
			depthInputs.prompt_mask = processed.prompt_mask;
			depthInputs.batch = processed.batch;
			depthInputs.height = processed.height;
			depthInputs.width = processed.width;
			depthInputs.sample_point_num = key.samplePointNum;
			infinidepth::InferenceForGsOutputs depthOutputs = depthSensor.InferenceForGs(depthInputs);
			timings.depthInferenceMs = elapsed_ms(depthBegin, Clock::now());

			const auto gsBegin = Clock::now();
			infinidepth::GsPipelineInputs gsInputs;
			gsInputs.image = processed.image;
			gsInputs.intrinsics = processed.intrinsics;
			gsInputs.extrinsics = processed.extrinsics;
			gsInputs.depth_outputs = &depthOutputs;
			gsInputs.batch = processed.batch;
			gsInputs.height = processed.height;
			gsInputs.width = processed.width;
			infinidepth::GsPipelineResult result = gsPipeline.RunAndExportPly(gsInputs, outputPlyPath);
			timings.gsInferenceAndExportMs = elapsed_ms(gsBegin, Clock::now());
			timings.gaussianCount = result.gaussian_count;
			timings.removedCount = result.removed_count;
		}catch(...){
			infinidepth::FreeProcessedGsInputsCuda(&processed);
			throw;
		}
		infinidepth::FreeProcessedGsInputsCuda(&processed);
	}
};

std::unique_ptr<CachedInfiniDepthSession>& cached_session(){
	static std::unique_ptr<CachedInfiniDepthSession> session;
	return session;
}

SessionKey make_session_key(){
	SessionKey key;
	key.encoderModelPath = env_or_default(
		"SPLATSHOP_INFINIDEPTH_ENCODER",
		R"(E:\projects\InfiniDepth\workspace\infinidepth_depthsensor_encoder.onnx)");
	key.decoderModelPath = env_or_default(
		"SPLATSHOP_INFINIDEPTH_DECODER",
		R"(E:\projects\InfiniDepth\workspace\infinidepth_depthsensor_decoder.onnx)");
	key.gsModelPath = env_or_default(
		"SPLATSHOP_INFINIDEPTH_GS_MODEL",
		R"(E:\projects\MoGe\workspace\gs_predictor_dynamic.onnx)");
	key.cudaDeviceId = env_int_or_default("SPLATSHOP_INFINIDEPTH_CUDA_DEVICE", 0);
	key.chunkSize = env_i64_or_default("SPLATSHOP_INFINIDEPTH_CHUNK_SIZE", 100000);
	key.samplePointNum = env_i64_or_default("SPLATSHOP_INFINIDEPTH_SAMPLE_POINT_NUM", 2000000);
	key.disableOptimizations = !env_bool_or_default("SPLATSHOP_INFINIDEPTH_ENABLE_OPTIMIZATIONS", false);
	key.cudnnConvAlgoSearch = env_or_default("SPLATSHOP_INFINIDEPTH_CUDNN_CONV_ALGO_SEARCH", "HEURISTIC");
	key.cudnnConvUseMaxWorkspace = env_bool_or_default("SPLATSHOP_INFINIDEPTH_CUDNN_CONV_USE_MAX_WORKSPACE", false);
	key.cudaUseTf32 = env_bool_or_default("SPLATSHOP_INFINIDEPTH_CUDA_USE_TF32", true);
	return key;
}

RunOptions make_run_options(){
	RunOptions options;
	options.inputHeight = env_int_or_default("SPLATSHOP_INFINIDEPTH_INPUT_HEIGHT", 768);
	options.inputWidth = env_int_or_default("SPLATSHOP_INFINIDEPTH_INPUT_WIDTH", 1024);
	options.promptSamples = env_int_or_default("SPLATSHOP_INFINIDEPTH_PROMPT_SAMPLES", 1500);
	options.enableDepthNoiseFilter = env_bool_or_default("SPLATSHOP_INFINIDEPTH_ENABLE_DEPTH_NOISE_FILTER", false);
	return options;
}

bool cache_sessions_enabled(){
	return env_bool_or_default("SPLATSHOP_INFINIDEPTH_CACHE_SESSIONS", true);
}

void validate_file(const std::string& path, const char* label){
	if(path.empty() || !std::filesystem::exists(path)){
		throw std::runtime_error(std::string(label) + " not found: " + path);
	}
}

} // namespace

int InfiniDepthGsPredictor::predictToPly(
	const std::string& imagePath,
	const std::string& depthPath,
	const std::string& cameraJsonPath,
	const std::string& outputPlyPath,
	InfiniDepthGsTimings* timings
){
	InfiniDepthGsTimings localTimings;
	InfiniDepthGsTimings& t = timings ? *timings : localTimings;
	t = InfiniDepthGsTimings{};
	const auto totalBegin = Clock::now();

	try{
		configure_dll_search_path();
		Ort::InitApi();

		SessionKey key = make_session_key();
		RunOptions runOptions = make_run_options();

		validate_file(key.encoderModelPath, "InfiniDepth encoder model");
		validate_file(key.decoderModelPath, "InfiniDepth decoder model");
		validate_file(key.gsModelPath, "InfiniDepth GS model");
		validate_file(imagePath, "InfiniDepth input image");
		validate_file(depthPath, "InfiniDepth input depth");
		validate_file(cameraJsonPath, "InfiniDepth camera JSON");

		std::filesystem::path outputPath(outputPlyPath);
		if(outputPath.has_parent_path()){
			std::filesystem::create_directories(outputPath.parent_path());
		}

		CUDA_CHECK(cudaSetDevice(key.cudaDeviceId));

		bool useCache = cache_sessions_enabled();
		t.cacheSessions = useCache;
		if(useCache){
			std::unique_ptr<CachedInfiniDepthSession>& session = cached_session();
			if(!session || !(session->key == key)){
				std::cout << "Loading cached InfiniDepth ONNX models:\n"
					<< "  encoder: " << key.encoderModelPath << "\n"
					<< "  decoder: " << key.decoderModelPath << "\n"
					<< "  gsModel: " << key.gsModelPath << "\n"
					<< "  cudnnConvAlgoSearch: " << key.cudnnConvAlgoSearch << "\n"
					<< "  cudnnConvUseMaxWorkspace: " << key.cudnnConvUseMaxWorkspace << "\n"
					<< "  cudaUseTf32: " << key.cudaUseTf32 << "\n";
				const auto loadBegin = Clock::now();
				session = std::make_unique<CachedInfiniDepthSession>(key);
				t.modelLoadMs = elapsed_ms(loadBegin, Clock::now());
				t.modelsLoadedThisCall = true;
			}

			std::cout << "InfiniDepth GS prediction started: " << outputPlyPath << "\n";
			session->run(runOptions, imagePath, depthPath, cameraJsonPath, outputPlyPath, t);
		}else{
			cached_session().reset();
			std::cout << "Loading temporary InfiniDepth ONNX models:\n"
				<< "  encoder: " << key.encoderModelPath << "\n"
				<< "  decoder: " << key.decoderModelPath << "\n"
				<< "  gsModel: " << key.gsModelPath << "\n"
				<< "  cudnnConvAlgoSearch: " << key.cudnnConvAlgoSearch << "\n"
				<< "  cudnnConvUseMaxWorkspace: " << key.cudnnConvUseMaxWorkspace << "\n"
				<< "  cudaUseTf32: " << key.cudaUseTf32 << "\n";
			const auto loadBegin = Clock::now();
			std::unique_ptr<CachedInfiniDepthSession> session = std::make_unique<CachedInfiniDepthSession>(key);
			t.modelLoadMs = elapsed_ms(loadBegin, Clock::now());
			t.modelsLoadedThisCall = true;

			std::cout << "InfiniDepth GS prediction started: " << outputPlyPath << "\n";
			session->run(runOptions, imagePath, depthPath, cameraJsonPath, outputPlyPath, t);
			session.reset();
			CUDA_CHECK(cudaDeviceSynchronize());
		}

		t.totalMs = elapsed_ms(totalBegin, Clock::now());

		std::cout << "InfiniDepth GS prediction saved: " << outputPlyPath << "\n";
		std::cout << "InfiniDepth GS timings: total " << t.totalMs
			<< " ms, model load " << t.modelLoadMs
			<< " ms, input processing " << t.inputProcessingMs
			<< " ms, depth inference " << t.depthInferenceMs
			<< " ms, GS inference/export " << t.gsInferenceAndExportMs
			<< " ms, gaussians " << t.gaussianCount
			<< ", removed " << t.removedCount << "\n";
		return 0;
	}catch(const std::exception& e){
		t.totalMs = elapsed_ms(totalBegin, Clock::now());
		std::cerr << "InfiniDepth GS prediction failed: " << e.what() << "\n";
		return 1;
	}
}

void InfiniDepthGsPredictor::clearCache(){
	cached_session().reset();
}
