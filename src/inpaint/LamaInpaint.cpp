#include "LamaInpaint.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kMaskThreshold = 96;
constexpr int kAlphaThreshold = 128;
constexpr int kMinComponentArea = 1000;
constexpr int kEdgeComponentArea = 3000;
constexpr int kCloseKernel = 11;
constexpr int kDilateKernel = 31;

std::string env_or_default(const char* name, std::string fallback){
	const char* value = std::getenv(name);
	if(value && value[0] != '\0'){
		return value;
	}
	return fallback;
}

cv::Mat load_rgb(const std::string& path){
	cv::Mat input = cv::imread(path, cv::IMREAD_UNCHANGED);
	if(input.empty()){
		throw std::runtime_error("Failed to read image: " + path);
	}

	cv::Mat bgr;
	if(input.channels() == 4){
		cv::cvtColor(input, bgr, cv::COLOR_BGRA2BGR);
	}else if(input.channels() == 3){
		bgr = input;
	}else if(input.channels() == 1){
		cv::cvtColor(input, bgr, cv::COLOR_GRAY2BGR);
	}else{
		throw std::runtime_error("Unsupported image channel count: " + std::to_string(input.channels()));
	}

	cv::Mat rgb;
	cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
	return rgb;
}

cv::Mat load_alpha_missing(const std::string& path, cv::Size size){
	cv::Mat input = cv::imread(path, cv::IMREAD_UNCHANGED);
	if(input.empty() || input.channels() != 4){
		return cv::Mat();
	}

	std::vector<cv::Mat> channels;
	cv::split(input, channels);
	cv::Mat missing;
	cv::threshold(channels[3], missing, kAlphaThreshold - 1, 255, cv::THRESH_BINARY_INV);
	if(missing.size() != size){
		cv::resize(missing, missing, size, 0.0, 0.0, cv::INTER_NEAREST);
	}
	return missing;
}

cv::Mat preprocess_mask(const std::string& imagePath, const std::string& maskPath, cv::Size imageSize){
	cv::Mat mask = cv::imread(maskPath, cv::IMREAD_GRAYSCALE);
	if(mask.empty()){
		throw std::runtime_error("Failed to read mask: " + maskPath);
	}
	if(mask.size() != imageSize){
		cv::resize(mask, mask, imageSize, 0.0, 0.0, cv::INTER_NEAREST);
	}

	cv::Mat binary;
	cv::threshold(mask, binary, kMaskThreshold - 1, 255, cv::THRESH_BINARY);

	cv::Mat alphaMissing = load_alpha_missing(imagePath, imageSize);
	if(!alphaMissing.empty()){
		cv::bitwise_or(binary, alphaMissing, binary);
	}

	cv::Mat labels;
	cv::Mat stats;
	cv::Mat centroids;
	int count = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);
	cv::Mat cleaned = cv::Mat::zeros(binary.size(), CV_8U);

	for(int label = 1; label < count; label++){
		int area = stats.at<int>(label, cv::CC_STAT_AREA);
		int x = stats.at<int>(label, cv::CC_STAT_LEFT);
		int y = stats.at<int>(label, cv::CC_STAT_TOP);
		int w = stats.at<int>(label, cv::CC_STAT_WIDTH);
		int h = stats.at<int>(label, cv::CC_STAT_HEIGHT);
		bool touchesEdge = x == 0 || y == 0 || x + w == imageSize.width || y + h == imageSize.height;
		int requiredArea = touchesEdge ? kEdgeComponentArea : kMinComponentArea;
		if(area >= requiredArea){
			cleaned.setTo(255, labels == label);
		}
	}

	cv::morphologyEx(cleaned, cleaned, cv::MORPH_CLOSE, cv::Mat::ones(kCloseKernel, kCloseKernel, CV_8U));
	cv::dilate(cleaned, cleaned, cv::Mat::ones(kDilateKernel, kDilateKernel, CV_8U));
	return cleaned;
}

bool is_fp32_named_model(const std::string& modelPath){
	return modelPath.find("fp32") != std::string::npos;
}

#ifdef _WIN32
std::wstring widen_path(const std::string& path){
	int count = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
	if(count <= 0){
		return std::wstring(path.begin(), path.end());
	}

	std::wstring wide(count - 1, L'\0');
	MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wide.data(), count);
	return wide;
}
#endif

struct LamaSession{
	Ort::Env env;
	Ort::Session session;
	Ort::MemoryInfo memoryInfo;
	std::vector<const char*> inputNames;
	std::vector<const char*> outputNames;
	int modelSize = 512;
	std::string provider = "cuda";

	LamaSession(const std::string& modelPath)
		: env(ORT_LOGGING_LEVEL_WARNING, "splatshop_lama_inpaint"),
		  session(create_session(modelPath)),
		  memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
	{
		modelSize = infer_square_model_size();
		bool fp32Model = is_fp32_named_model(modelPath);
		inputNames = fp32Model ? std::vector<const char*>{"image", "mask"} : std::vector<const char*>{"l_image_", "l_mask_"};
		outputNames = fp32Model ? std::vector<const char*>{"output"} : std::vector<const char*>{"clamp"};
	}

	void configure_execution_provider(Ort::SessionOptions& session_options, const std::string& provider) {
		if (provider == "cpu") {
			std::cout << "provider CPUExecutionProvider\n";
			return;
		}

		OrtCUDAProviderOptions cuda_options{};
		cuda_options.device_id = 0;
		session_options.AppendExecutionProvider_CUDA(cuda_options);
		std::cout << "provider CUDAExecutionProvider\n";
	}

	Ort::Session create_session(const std::string& modelPath){
		Ort::SessionOptions options;
		options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		configure_execution_provider(options, provider);
#ifdef _WIN32
		std::wstring modelPathW = widen_path(modelPath);
		return Ort::Session(env, modelPathW.c_str(), options);
#else
		return Ort::Session(env, modelPath.c_str(), options);
#endif
	}

	int infer_square_model_size(){
		Ort::TypeInfo typeInfo = session.GetInputTypeInfo(0);
		auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
		std::vector<int64_t> shape = tensorInfo.GetShape();
		if(shape.size() != 4 || shape[2] <= 0 || shape[3] <= 0 || shape[2] != shape[3]){
			throw std::runtime_error("Expected a fixed square ONNX input shape [N,C,S,S].");
		}
		return static_cast<int>(shape[2]);
	}

	std::vector<float> rgb_to_nchw(const cv::Mat& rgb){
		cv::Mat resized;
		cv::resize(rgb, resized, cv::Size(modelSize, modelSize), 0.0, 0.0, cv::INTER_AREA);

		std::vector<float> tensor(3 * modelSize * modelSize);
		int plane = modelSize * modelSize;
		for(int y = 0; y < modelSize; y++){
			const cv::Vec3b* row = resized.ptr<cv::Vec3b>(y);
			for(int x = 0; x < modelSize; x++){
				int idx = y * modelSize + x;
				tensor[idx] = row[x][0] / 255.0f;
				tensor[plane + idx] = row[x][1] / 255.0f;
				tensor[2 * plane + idx] = row[x][2] / 255.0f;
			}
		}
		return tensor;
	}

	std::vector<float> mask_to_nchw(const cv::Mat& mask){
		cv::Mat resized;
		cv::resize(mask, resized, cv::Size(modelSize, modelSize), 0.0, 0.0, cv::INTER_NEAREST);

		std::vector<float> tensor(modelSize * modelSize);
		for(int y = 0; y < modelSize; y++){
			const uint8_t* row = resized.ptr<uint8_t>(y);
			for(int x = 0; x < modelSize; x++){
				int idx = y * modelSize + x;
				tensor[idx] = row[x] > 0 ? 1.0f : 0.0f;
			}
		}
		return tensor;
	}

	cv::Mat output_to_rgb(const float* output){
		cv::Mat rgb(modelSize, modelSize, CV_8UC3);
		int plane = modelSize * modelSize;
		for(int y = 0; y < modelSize; y++){
			cv::Vec3b* row = rgb.ptr<cv::Vec3b>(y);
			for(int x = 0; x < modelSize; x++){
				int idx = y * modelSize + x;
				row[x][0] = static_cast<uint8_t>(std::clamp(output[idx], 0.0f, 255.0f));
				row[x][1] = static_cast<uint8_t>(std::clamp(output[plane + idx], 0.0f, 255.0f));
				row[x][2] = static_cast<uint8_t>(std::clamp(output[2 * plane + idx], 0.0f, 255.0f));
			}
		}
		return rgb;
	}

	cv::Mat run_onnx(const cv::Mat& modelRgb, const cv::Mat& modelMask){
		std::vector<float> imageTensor = rgb_to_nchw(modelRgb);
		std::vector<float> maskTensor = mask_to_nchw(modelMask);

		std::array<int64_t, 4> imageShape = {1, 3, modelSize, modelSize};
		std::array<int64_t, 4> maskShape = {1, 1, modelSize, modelSize};

		Ort::Value imageValue = Ort::Value::CreateTensor<float>(
			memoryInfo, imageTensor.data(), imageTensor.size(), imageShape.data(), imageShape.size());
		Ort::Value maskValue = Ort::Value::CreateTensor<float>(
			memoryInfo, maskTensor.data(), maskTensor.size(), maskShape.data(), maskShape.size());
		std::array<Ort::Value, 2> inputs = {std::move(imageValue), std::move(maskValue)};

		std::vector<Ort::Value> outputs = session.Run(
			Ort::RunOptions{nullptr},
			inputNames.data(),
			inputs.data(),
			inputs.size(),
			outputNames.data(),
			outputNames.size());

		return output_to_rgb(outputs[0].GetTensorData<float>());
	}

	cv::Mat run_full(const cv::Mat& rgb, const cv::Mat& mask){
		cv::Mat prediction = run_onnx(rgb, mask);
		cv::Mat predictionResized;
		cv::resize(prediction, predictionResized, rgb.size(), 0.0, 0.0, cv::INTER_CUBIC);

		cv::Mat result = rgb.clone();
		predictionResized.copyTo(result, mask);
		return result;
	}

	cv::Rect mask_bbox_or_full(const cv::Mat& mask, int context){
		std::vector<cv::Point> points;
		cv::findNonZero(mask, points);
		if(points.empty()){
			return cv::Rect(0, 0, mask.cols, mask.rows);
		}

		cv::Rect bbox = cv::boundingRect(points);
		int x0 = std::max(0, bbox.x - context);
		int y0 = std::max(0, bbox.y - context);
		int x1 = std::min(mask.cols, bbox.x + bbox.width + context);
		int y1 = std::min(mask.rows, bbox.y + bbox.height + context);
		return cv::Rect(x0, y0, x1 - x0, y1 - y0);
	}

	cv::Mat run_crop(const cv::Mat& rgb, const cv::Mat& mask, int context){
		cv::Rect roi = mask_bbox_or_full(mask, context);
		cv::Mat rgbCrop = rgb(roi).clone();
		cv::Mat maskCrop = mask(roi).clone();

		cv::Mat modelRgb;
		cv::Mat modelMask;
		cv::resize(rgbCrop, modelRgb, cv::Size(modelSize, modelSize), 0.0, 0.0, cv::INTER_AREA);
		cv::resize(maskCrop, modelMask, cv::Size(modelSize, modelSize), 0.0, 0.0, cv::INTER_NEAREST);
		cv::Mat prediction = run_onnx(modelRgb, modelMask);

		cv::Mat predCrop;
		cv::resize(prediction, predCrop, roi.size(), 0.0, 0.0, cv::INTER_CUBIC);

		cv::Mat result = rgb.clone();
		predCrop.copyTo(result(roi), maskCrop);
		return result;
	}

	std::vector<int> tile_starts(int length, int tile, int stride){
		std::vector<int> starts;
		if(length <= tile){
			starts.push_back(0);
			return starts;
		}

		for(int start = 0; start + tile < length; start += stride){
			starts.push_back(start);
		}
		if(starts.empty() || starts.back() != length - tile){
			starts.push_back(length - tile);
		}
		return starts;
	}

	cv::Mat pad_rgb(const cv::Mat& crop){
		if(crop.cols == modelSize && crop.rows == modelSize){
			return crop.clone();
		}
		if(crop.cols > modelSize || crop.rows > modelSize){
			cv::Mat resized;
			cv::resize(crop, resized, cv::Size(modelSize, modelSize), 0.0, 0.0, cv::INTER_AREA);
			return resized;
		}

		cv::Mat padded;
		cv::copyMakeBorder(crop, padded, 0, modelSize - crop.rows, 0, modelSize - crop.cols, cv::BORDER_REFLECT);
		return padded;
	}

	cv::Mat pad_mask(const cv::Mat& crop){
		if(crop.cols == modelSize && crop.rows == modelSize){
			return crop.clone();
		}
		if(crop.cols > modelSize || crop.rows > modelSize){
			cv::Mat resized;
			cv::resize(crop, resized, cv::Size(modelSize, modelSize), 0.0, 0.0, cv::INTER_NEAREST);
			return resized;
		}

		cv::Mat padded;
		cv::copyMakeBorder(crop, padded, 0, modelSize - crop.rows, 0, modelSize - crop.cols, cv::BORDER_CONSTANT, cv::Scalar(0));
		return padded;
	}

	cv::Mat run_tile(const cv::Mat& rgb, const cv::Mat& mask, int stride){
		cv::Mat acc = cv::Mat::zeros(rgb.size(), CV_32FC3);
		cv::Mat count = cv::Mat::zeros(rgb.size(), CV_32FC1);

		for(int y : tile_starts(rgb.rows, modelSize, stride)){
			for(int x : tile_starts(rgb.cols, modelSize, stride)){
				int w = std::min(modelSize, rgb.cols - x);
				int h = std::min(modelSize, rgb.rows - y);
				cv::Rect roi(x, y, w, h);
				cv::Mat maskCrop = mask(roi);
				if(cv::countNonZero(maskCrop) == 0){
					continue;
				}

				cv::Mat prediction = run_onnx(pad_rgb(rgb(roi)), pad_mask(maskCrop));
				cv::Mat predCrop = prediction(cv::Rect(0, 0, w, h)).clone();
				cv::Mat predFloat;
				predCrop.convertTo(predFloat, CV_32FC3);

				cv::Mat maskFloat;
				maskCrop.convertTo(maskFloat, CV_32FC1, 1.0 / 255.0);

				cv::Mat accRoi = acc(roi);
				cv::Mat countRoi = count(roi);
				std::vector<cv::Mat> channels;
				cv::split(predFloat, channels);
				for(cv::Mat& channel : channels){
					channel = channel.mul(maskFloat);
				}
				cv::Mat weightedPred;
				cv::merge(channels, weightedPred);
				accRoi += weightedPred;
				countRoi += maskFloat;
			}
		}

		cv::Mat result = rgb.clone();
		for(int y = 0; y < rgb.rows; y++){
			const float* countRow = count.ptr<float>(y);
			const cv::Vec3f* accRow = acc.ptr<cv::Vec3f>(y);
			cv::Vec3b* resultRow = result.ptr<cv::Vec3b>(y);
			const uint8_t* maskRow = mask.ptr<uint8_t>(y);
			for(int x = 0; x < rgb.cols; x++){
				if(maskRow[x] > 0 && countRow[x] > 0.0f){
					cv::Vec3f value = accRow[x] / countRow[x];
					resultRow[x][0] = static_cast<uint8_t>(std::clamp(value[0], 0.0f, 255.0f));
					resultRow[x][1] = static_cast<uint8_t>(std::clamp(value[1], 0.0f, 255.0f));
					resultRow[x][2] = static_cast<uint8_t>(std::clamp(value[2], 0.0f, 255.0f));
				}
			}
		}
		return result;
	}
};

std::unique_ptr<LamaSession>& cached_session(){
	static std::unique_ptr<LamaSession> session;
	return session;
}

double elapsed_ms(Clock::time_point start, Clock::time_point end){
	return std::chrono::duration<double, std::milli>(end - start).count();
}

} // namespace

int LamaInpaint::inpaintFramebuffer(
	const std::string& imagePath,
	const std::string& maskPath,
	const std::string& outputPath,
	LamaInpaintTimings* timings
){
	LamaInpaintTimings localTimings;
	LamaInpaintTimings& t = timings ? *timings : localTimings;
	t = LamaInpaintTimings{};
	const auto totalBegin = Clock::now();
	try{
		std::string modelPath = env_or_default("SPLATSHOP_LAMA_ONNX_MODEL", R"(E:\projects\sd_models\onnx\carve_lama_fp32_1024_cuda_export.onnx)");
		std::string mode = env_or_default("SPLATSHOP_LAMA_MODE", "full");
		int context = std::stoi(env_or_default("SPLATSHOP_LAMA_CONTEXT", "128"));
		int stride = std::stoi(env_or_default("SPLATSHOP_LAMA_STRIDE", "768"));

		if(!std::filesystem::exists(modelPath)){
			std::cerr << "Framebuffer inpainting skipped: model not found: " << modelPath << "\n";
			return -1;
		}

		std::unique_ptr<LamaSession>& session = cached_session();
		if(!session){
			std::cout << "Loading LaMa ONNX model: " << modelPath << "\n";
			const auto loadBegin = Clock::now();
			session = std::make_unique<LamaSession>(modelPath);
			t.modelLoadMs = elapsed_ms(loadBegin, Clock::now());
			t.modelLoadedThisCall = true;
		}

		const auto inputBegin = Clock::now();
		cv::Mat rgb = load_rgb(imagePath);
		cv::Mat mask = preprocess_mask(imagePath, maskPath, rgb.size());
		t.inputLoadAndPreprocessMs = elapsed_ms(inputBegin, Clock::now());

		cv::Mat resultRgb;
		const auto inferenceBegin = Clock::now();
		if(mode == "full"){
			resultRgb = session->run_full(rgb, mask);
		}else if(mode == "crop"){
			resultRgb = session->run_crop(rgb, mask, context);
		}else if(mode == "tile"){
			resultRgb = session->run_tile(rgb, mask, stride);
		}else{
			throw std::runtime_error("Unknown inpaint mode: " + mode);
		}
		t.inferenceMs = elapsed_ms(inferenceBegin, Clock::now());

		const auto saveBegin = Clock::now();
		cv::Mat resultBgr;
		cv::cvtColor(resultRgb, resultBgr, cv::COLOR_RGB2BGR);
		if(!cv::imwrite(outputPath, resultBgr)){
			throw std::runtime_error("Failed to write output: " + outputPath);
		}
		t.resultSaveMs = elapsed_ms(saveBegin, Clock::now());
		t.totalMs = elapsed_ms(totalBegin, Clock::now());

		std::cout << "Framebuffer inpainting saved: " << outputPath << "\n";
		std::cout << "Framebuffer inpainting timings: total " << t.totalMs
			<< " ms, model load " << t.modelLoadMs
			<< " ms, input/preprocess " << t.inputLoadAndPreprocessMs
			<< " ms, inference " << t.inferenceMs
			<< " ms, result save " << t.resultSaveMs << " ms\n";
		return 0;
	}catch(const std::exception& e){
		t.totalMs = elapsed_ms(totalBegin, Clock::now());
		std::cerr << "Framebuffer inpainting failed: " << e.what() << "\n";
		return 1;
	}
}
