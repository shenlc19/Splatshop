#include "infinidepth_depthsensor_onnx.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cuda_runtime.h>

#include "infinidepth_depthsensor_kernels.cuh"

namespace infinidepth {
namespace {

#define CUDA_CHECK(expr)                                                        \
  do {                                                                         \
    cudaError_t err = (expr);                                                   \
    if (err != cudaSuccess) {                                                   \
      throw std::runtime_error(std::string("CUDA error: ") +                  \
                               cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)

Ort::SessionOptions MakeSessionOptions(const RuntimeOptions& options) {
  Ort::SessionOptions session_options;
  if (options.disable_graph_optimizations) {
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    session_options.DisableMemPattern();
  } else {
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  }

  Ort::CUDAProviderOptions cuda_options;
  cuda_options.Update({
      {"device_id", std::to_string(options.device_id)},
      {"cudnn_conv_algo_search", options.cudnn_conv_algo_search},
      {"cudnn_conv_use_max_workspace", options.cudnn_conv_use_max_workspace ? "1" : "0"},
      {"use_tf32", options.cuda_use_tf32 ? "1" : "0"},
  });
  session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);
  return session_options;
}

Ort::MemoryInfo MakeCudaMemoryInfo(int device_id) {
  return Ort::MemoryInfo("Cuda", OrtAllocatorType::OrtDeviceAllocator, device_id, OrtMemTypeDefault);
}

Ort::Value MakeCudaTensor(
    const Ort::MemoryInfo& memory_info,
    void* data,
    size_t bytes,
    const std::vector<int64_t>& shape,
    ONNXTensorElementDataType dtype) {
  return Ort::Value::CreateTensor(
      memory_info,
      data,
      bytes,
      shape.data(),
      shape.size(),
      dtype);
}

size_t Numel(const std::vector<int64_t>& shape) {
  size_t n = 1;
  for (int64_t dim : shape) n *= static_cast<size_t>(dim);
  return n;
}

void CopyDeviceBuffer(void** dst, const void* src, size_t bytes) {
  CUDA_CHECK(cudaMalloc(dst, bytes));
  CUDA_CHECK(cudaMemcpy(*dst, src, bytes, cudaMemcpyDeviceToDevice));
}

std::wstring WidenPath(const std::string& path) {
  if (path.empty()) return std::wstring();
  int size = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
  if (size <= 0) {
    throw std::runtime_error("Failed to convert UTF-8 path to wide string.");
  }
  std::wstring wide(static_cast<size_t>(size), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wide.data(), size);
  if (!wide.empty() && wide.back() == L'\0') wide.pop_back();
  return wide;
}

}  // namespace

InferenceOutputs::~InferenceOutputs() {
  if (pred_depth) cudaFree(pred_depth);
  if (pred_disparity) cudaFree(pred_disparity);
  if (dino_tokens) cudaFree(dino_tokens);
}

InferenceOutputs::InferenceOutputs(InferenceOutputs&& other) noexcept {
  *this = std::move(other);
}

InferenceOutputs& InferenceOutputs::operator=(InferenceOutputs&& other) noexcept {
  if (this == &other) return *this;
  if (pred_depth) cudaFree(pred_depth);
  if (pred_disparity) cudaFree(pred_disparity);
  if (dino_tokens) cudaFree(dino_tokens);

  reference_meta_host = std::move(other.reference_meta_host);
  pred_depth = other.pred_depth;
  pred_disparity = other.pred_disparity;
  dino_tokens = other.dino_tokens;
  pred_shape = std::move(other.pred_shape);
  dino_tokens_shape = std::move(other.dino_tokens_shape);

  other.pred_depth = nullptr;
  other.pred_disparity = nullptr;
  other.dino_tokens = nullptr;
  return *this;
}

InferenceForGsOutputs::~InferenceForGsOutputs() {
  if (depthmap) cudaFree(depthmap);
  if (dino_tokens) cudaFree(dino_tokens);
  if (query_3d_uniform_coord) cudaFree(query_3d_uniform_coord);
  if (pred_depth_3d) cudaFree(pred_depth_3d);
}

InferenceForGsOutputs::InferenceForGsOutputs(InferenceForGsOutputs&& other) noexcept {
  *this = std::move(other);
}

InferenceForGsOutputs& InferenceForGsOutputs::operator=(InferenceForGsOutputs&& other) noexcept {
  if (this == &other) return *this;
  if (depthmap) cudaFree(depthmap);
  if (dino_tokens) cudaFree(dino_tokens);
  if (query_3d_uniform_coord) cudaFree(query_3d_uniform_coord);
  if (pred_depth_3d) cudaFree(pred_depth_3d);

  depthmap = other.depthmap;
  dino_tokens = other.dino_tokens;
  query_3d_uniform_coord = other.query_3d_uniform_coord;
  pred_depth_3d = other.pred_depth_3d;
  depthmap_shape = std::move(other.depthmap_shape);
  dino_tokens_shape = std::move(other.dino_tokens_shape);
  query_3d_uniform_coord_shape = std::move(other.query_3d_uniform_coord_shape);
  pred_depth_3d_shape = std::move(other.pred_depth_3d_shape);
  reference_meta_host = std::move(other.reference_meta_host);

  other.depthmap = nullptr;
  other.dino_tokens = nullptr;
  other.query_3d_uniform_coord = nullptr;
  other.pred_depth_3d = nullptr;
  return *this;
}

InfiniDepthDepthSensorOnnx::InfiniDepthDepthSensorOnnx(
    const std::string& encoder_onnx_path,
    const std::string& decoder_onnx_path,
    RuntimeOptions options)
    : env_(ORT_LOGGING_LEVEL_WARNING, "infinidepth_depthsensor"),
      session_options_(MakeSessionOptions(options)),
      cuda_memory_info_(MakeCudaMemoryInfo(options.device_id)),
      encoder_session_(env_, WidenPath(encoder_onnx_path).c_str(), session_options_),
      decoder_session_(env_, WidenPath(decoder_onnx_path).c_str(), session_options_),
      options_(options) {
  if (options_.decoder_chunk_size <= 0) {
    throw std::invalid_argument("decoder_chunk_size must be positive.");
  }
}

InferenceOutputs InfiniDepthDepthSensorOnnx::Infer(const InferenceInputs& inputs) {
  if (!inputs.image || !inputs.query_coord || !inputs.gt_depth || !inputs.gt_depth_mask ||
      !inputs.prompt_depth || !inputs.prompt_mask) {
    throw std::invalid_argument("InferenceInputs contains a null device pointer.");
  }
  if (inputs.batch <= 0 || inputs.height <= 0 || inputs.width <= 0 || inputs.num_queries <= 0) {
    throw std::invalid_argument("InferenceInputs has invalid shape values.");
  }

  PreparedDepthSensorInputs prepared = PrepareDepthSensorInputsCuda(
      inputs.gt_depth,
      inputs.gt_depth_mask,
      inputs.prompt_depth,
      inputs.prompt_mask,
      inputs.batch,
      inputs.height,
      inputs.width);

  InferenceOutputs outputs;
  outputs.reference_meta_host.resize(static_cast<size_t>(inputs.batch));
  CUDA_CHECK(cudaMemcpy(
      outputs.reference_meta_host.data(),
      prepared.reference_meta,
      sizeof(float) * static_cast<size_t>(inputs.batch),
      cudaMemcpyDeviceToHost));

  std::vector<int64_t> image_shape{inputs.batch, 3, inputs.height, inputs.width};
  std::vector<int64_t> prompt_shape{inputs.batch, 1, inputs.height, inputs.width};

  Ort::Value image_value = MakeCudaTensor(
      cuda_memory_info_,
      const_cast<float*>(inputs.image),
      sizeof(float) * Numel(image_shape),
      image_shape,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  Ort::Value prompt_depth_value = MakeCudaTensor(
      cuda_memory_info_,
      prepared.prompt_depth,
      sizeof(float) * Numel(prompt_shape),
      prompt_shape,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  Ort::Value prompt_mask_value = MakeCudaTensor(
      cuda_memory_info_,
      prepared.prompt_mask,
      sizeof(uint8_t) * Numel(prompt_shape),
      prompt_shape,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);

  Ort::IoBinding encoder_binding(encoder_session_);
  encoder_binding.BindInput("image", image_value);
  encoder_binding.BindInput("prompt_depth", prompt_depth_value);
  encoder_binding.BindInput("prompt_mask", prompt_mask_value);
  encoder_binding.BindOutput("feat", cuda_memory_info_);
  encoder_binding.BindOutput("basic_feat", cuda_memory_info_);
  encoder_binding.BindOutput("dino_tokens", cuda_memory_info_);

  Ort::RunOptions run_options;
  encoder_session_.Run(run_options, encoder_binding);
  std::vector<Ort::Value> encoder_outputs = encoder_binding.GetOutputValues();
  if (encoder_outputs.size() != 3) {
    FreePreparedDepthSensorInputs(&prepared);
    throw std::runtime_error("Encoder ONNX did not return feat, basic_feat, dino_tokens.");
  }

  Ort::Value& feat_value = encoder_outputs[0];
  Ort::Value& basic_feat_value = encoder_outputs[1];
  Ort::Value& dino_tokens_value = encoder_outputs[2];
  std::vector<int64_t> feat_shape = feat_value.GetTensorTypeAndShapeInfo().GetShape();
  std::vector<int64_t> basic_feat_shape = basic_feat_value.GetTensorTypeAndShapeInfo().GetShape();
  outputs.dino_tokens_shape = dino_tokens_value.GetTensorTypeAndShapeInfo().GetShape();

  CopyDeviceBuffer(
      &outputs.dino_tokens,
      dino_tokens_value.GetTensorData<float>(),
      sizeof(float) * Numel(outputs.dino_tokens_shape));

  std::vector<int64_t> pred_shape{inputs.batch, inputs.num_queries, 1};
  outputs.pred_shape = pred_shape;
  CUDA_CHECK(cudaMalloc(&outputs.pred_disparity, sizeof(float) * Numel(pred_shape)));
  CUDA_CHECK(cudaMalloc(&outputs.pred_depth, sizeof(float) * Numel(pred_shape)));

  int64_t offset = 0;
  while (offset < inputs.num_queries) {
    int64_t chunk = std::min(options_.decoder_chunk_size, inputs.num_queries - offset);
    std::vector<int64_t> query_shape{inputs.batch, chunk, 2};
    float* query_chunk_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(
        &query_chunk_buffer,
        sizeof(float) * static_cast<size_t>(inputs.batch * chunk * 2)));
    GatherQueryChunkCuda(
        inputs.query_coord,
        query_chunk_buffer,
        inputs.batch,
        inputs.num_queries,
        offset,
        chunk);

    Ort::Value query_value = MakeCudaTensor(
        cuda_memory_info_,
        query_chunk_buffer,
        sizeof(float) * Numel(query_shape),
        query_shape,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    Ort::IoBinding decoder_binding(decoder_session_);
    decoder_binding.BindInput("feat", feat_value);
    decoder_binding.BindInput("basic_feat", basic_feat_value);
    decoder_binding.BindInput("query_coord", query_value);
    decoder_binding.BindOutput("pred", cuda_memory_info_);

    decoder_session_.Run(run_options, decoder_binding);
    std::vector<Ort::Value> decoder_outputs = decoder_binding.GetOutputValues();
    if (decoder_outputs.size() != 1) {
      FreePreparedDepthSensorInputs(&prepared);
      throw std::runtime_error("Decoder ONNX did not return pred.");
    }

    const float* pred_chunk = decoder_outputs[0].GetTensorData<float>();
    ScatterPredChunkCuda(
        pred_chunk,
        static_cast<float*>(outputs.pred_disparity),
        inputs.batch,
        inputs.num_queries,
        offset,
        chunk);
    CUDA_CHECK(cudaFree(query_chunk_buffer));

    offset += chunk;
  }

  PostprocessDepthSensorCuda(
      static_cast<const float*>(outputs.pred_disparity),
      prepared.reference_meta,
      static_cast<float*>(outputs.pred_depth),
      static_cast<float*>(outputs.pred_disparity),
      inputs.batch,
      inputs.num_queries);

  FreePreparedDepthSensorInputs(&prepared);
  return outputs;
}

InferenceForGsOutputs InfiniDepthDepthSensorOnnx::InferenceForGs(const InferenceForGsInputs& inputs) {
  if (!inputs.image || !inputs.intrinsics || !inputs.gt_depth || !inputs.gt_depth_mask ||
      !inputs.prompt_depth || !inputs.prompt_mask) {
    throw std::invalid_argument("InferenceForGsInputs contains a null device pointer.");
  }
  if (inputs.batch <= 0 || inputs.height <= 0 || inputs.width <= 0 || inputs.sample_point_num <= 0) {
    throw std::invalid_argument("InferenceForGsInputs has invalid shape values.");
  }

  int64_t dense_queries = inputs.height * inputs.width;
  float* dense_query = nullptr;
  CUDA_CHECK(cudaMalloc(
      &dense_query,
      sizeof(float) * static_cast<size_t>(inputs.batch * dense_queries * 2)));
  MakeDenseQueryCoordCuda(dense_query, inputs.batch, inputs.height, inputs.width);

  InferenceInputs dense_inputs;
  dense_inputs.image = inputs.image;
  dense_inputs.query_coord = dense_query;
  dense_inputs.gt_depth = inputs.gt_depth;
  dense_inputs.gt_depth_mask = inputs.gt_depth_mask;
  dense_inputs.prompt_depth = inputs.prompt_depth;
  dense_inputs.prompt_mask = inputs.prompt_mask;
  dense_inputs.batch = inputs.batch;
  dense_inputs.height = inputs.height;
  dense_inputs.width = inputs.width;
  dense_inputs.num_queries = dense_queries;

  InferenceOutputs dense_outputs = Infer(dense_inputs);
  CUDA_CHECK(cudaFree(dense_query));

  std::vector<float> intrinsics_host(static_cast<size_t>(inputs.batch * 9));
  CUDA_CHECK(cudaMemcpy(
      intrinsics_host.data(),
      inputs.intrinsics,
      sizeof(float) * intrinsics_host.size(),
      cudaMemcpyDeviceToHost));

  float* query_3d_uniform = nullptr;
  CUDA_CHECK(cudaMalloc(
      &query_3d_uniform,
      sizeof(float) * static_cast<size_t>(inputs.batch * inputs.sample_point_num * 2)));

  int64_t pixels = inputs.height * inputs.width;
  for (int64_t b = 0; b < inputs.batch; ++b) {
    const float* depth_b = static_cast<const float*>(dense_outputs.pred_depth) + b * pixels;
    const float* k = intrinsics_host.data() + b * 9;
    UniformCoordResult sampled = Make3DUniformCoordTriangleCuda(
        depth_b,
        static_cast<int>(inputs.height),
        static_cast<int>(inputs.width),
        k[0],
        k[4],
        k[2],
        k[5],
        inputs.sample_point_num);
    CUDA_CHECK(cudaMemcpy(
        query_3d_uniform + b * inputs.sample_point_num * 2,
        sampled.coords_yx,
        sizeof(float) * static_cast<size_t>(inputs.sample_point_num * 2),
        cudaMemcpyDeviceToDevice));
    FreeUniformCoordResult(&sampled);
  }

  InferenceInputs uniform_inputs;
  uniform_inputs.image = inputs.image;
  uniform_inputs.query_coord = query_3d_uniform;
  uniform_inputs.gt_depth = inputs.gt_depth;
  uniform_inputs.gt_depth_mask = inputs.gt_depth_mask;
  uniform_inputs.prompt_depth = inputs.prompt_depth;
  uniform_inputs.prompt_mask = inputs.prompt_mask;
  uniform_inputs.batch = inputs.batch;
  uniform_inputs.height = inputs.height;
  uniform_inputs.width = inputs.width;
  uniform_inputs.num_queries = inputs.sample_point_num;

  InferenceOutputs uniform_outputs = Infer(uniform_inputs);

  InferenceForGsOutputs outputs;
  outputs.depthmap = dense_outputs.pred_depth;
  outputs.dino_tokens = dense_outputs.dino_tokens;
  outputs.query_3d_uniform_coord = query_3d_uniform;
  outputs.pred_depth_3d = uniform_outputs.pred_depth;
  outputs.depthmap_shape = {inputs.batch, 1, inputs.height, inputs.width};
  outputs.dino_tokens_shape = std::move(dense_outputs.dino_tokens_shape);
  outputs.query_3d_uniform_coord_shape = {inputs.batch, inputs.sample_point_num, 2};
  outputs.pred_depth_3d_shape = {inputs.batch, inputs.sample_point_num, 1};
  outputs.reference_meta_host = std::move(dense_outputs.reference_meta_host);

  dense_outputs.pred_depth = nullptr;
  dense_outputs.dino_tokens = nullptr;
  uniform_outputs.pred_depth = nullptr;
  return outputs;
}

}  // namespace infinidepth
