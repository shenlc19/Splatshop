#include "infinidepth_gs_pipeline.h"

#include "infinidepth_gs_ply.cuh"
#include "infinidepth_gs_sparse.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

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

struct OrtNameSet {
  std::vector<Ort::AllocatedStringPtr> storage;
  std::vector<const char*> names;
};

OrtNameSet GetInputNames(Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator) {
  OrtNameSet result;
  for (size_t i = 0; i < session.GetInputCount(); ++i) {
    result.storage.emplace_back(session.GetInputNameAllocated(i, allocator));
    result.names.push_back(result.storage.back().get());
  }
  return result;
}

OrtNameSet GetOutputNames(Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator) {
  OrtNameSet result;
  for (size_t i = 0; i < session.GetOutputCount(); ++i) {
    result.storage.emplace_back(session.GetOutputNameAllocated(i, allocator));
    result.names.push_back(result.storage.back().get());
  }
  return result;
}

bool HasName(const std::vector<const char*>& names, const std::string& expected) {
  return std::any_of(names.begin(), names.end(), [&](const char* name) {
    return expected == name;
  });
}

void RequireNames(
    const std::vector<const char*>& names,
    const std::vector<std::string>& expected,
    const std::string& model_name) {
  for (const std::string& name : expected) {
    if (!HasName(names, name)) {
      throw std::runtime_error(model_name + " ONNX is missing expected input/output: " + name);
    }
  }
}

Ort::Value& GetOutputByName(
    std::vector<Ort::Value>& values,
    const std::vector<const char*>& names,
    const std::string& expected) {
  for (size_t i = 0; i < names.size(); ++i) {
    if (expected == names[i]) return values[i];
  }
  throw std::runtime_error("Missing ONNX output: " + expected);
}

Ort::SessionOptions MakeSessionOptions(const GsPipelineOptions& options) {
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
    const std::vector<int64_t>& shape) {
  return Ort::Value::CreateTensor(
      memory_info,
      data,
      bytes,
      shape.data(),
      shape.size(),
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
}

size_t Numel(const std::vector<int64_t>& shape) {
  size_t n = 1;
  for (int64_t dim : shape) n *= static_cast<size_t>(dim);
  return n;
}

std::wstring WidenPath(const std::string& path) {
  if (path.empty()) return {};
  int size = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
  if (size <= 0) {
    throw std::runtime_error("Failed to convert UTF-8 path to wide string.");
  }
  std::wstring wide(static_cast<size_t>(size), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wide.data(), size);
  if (!wide.empty() && wide.back() == L'\0') wide.pop_back();
  return wide;
}

std::vector<float> CopyDeviceF32(const float* device, size_t count) {
  std::vector<float> host(count);
  CUDA_CHECK(cudaMemcpy(host.data(), device, count * sizeof(float), cudaMemcpyDeviceToHost));
  return host;
}

std::vector<float> MakeIdentityExtrinsicsHost() {
  return {
      1.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 1.0f,
  };
}

}  // namespace

InfiniDepthGsPipeline::InfiniDepthGsPipeline(std::string gs_onnx_path, GsPipelineOptions options)
    : env_(ORT_LOGGING_LEVEL_WARNING, "infinidepth_gs_pipeline"),
      session_options_(MakeSessionOptions(options)),
      cuda_memory_info_(MakeCudaMemoryInfo(options.device_id)),
      gs_session_(env_, WidenPath(gs_onnx_path).c_str(), session_options_),
      options_(options) {}

GsPipelineResult InfiniDepthGsPipeline::RunAndExportPly(
    const GsPipelineInputs& inputs,
    const std::filesystem::path& output_ply_path) {
  if (!inputs.image || !inputs.intrinsics || !inputs.extrinsics || !inputs.depth_outputs) {
    throw std::invalid_argument("GsPipelineInputs contains a null pointer.");
  }
  if (inputs.batch != 1) {
    throw std::invalid_argument("The current GS sparse export path supports batch size 1.");
  }
  if (inputs.height <= 0 || inputs.width <= 0) {
    throw std::invalid_argument("GsPipelineInputs has invalid image shape.");
  }

  const InferenceForGsOutputs& depth = *inputs.depth_outputs;
  std::vector<int64_t> image_shape{1, 3, inputs.height, inputs.width};
  std::vector<int64_t> depth_shape{1, 1, inputs.height, inputs.width};
  std::vector<int64_t> intrinsics_shape{1, 3, 3};
  std::vector<int64_t> extrinsics_shape{1, 4, 4};
  std::vector<int64_t> dino_shape = depth.dino_tokens_shape;

  Ort::AllocatorWithDefaultOptions allocator;
  OrtNameSet gs_inputs = GetInputNames(gs_session_, allocator);
  OrtNameSet gs_outputs = GetOutputNames(gs_session_, allocator);
  RequireNames(gs_inputs.names, {"image", "depthmap", "dino_tokens", "intrinsics", "extrinsics"}, "GS predictor");
  RequireNames(gs_outputs.names, {"means", "harmonics", "opacities", "scales", "rotations"}, "GS predictor");

  Ort::Value image_value = MakeCudaTensor(
      cuda_memory_info_,
      const_cast<float*>(inputs.image),
      sizeof(float) * Numel(image_shape),
      image_shape);
  Ort::Value depth_value = MakeCudaTensor(
      cuda_memory_info_,
      depth.depthmap,
      sizeof(float) * Numel(depth_shape),
      depth_shape);
  Ort::Value dino_value = MakeCudaTensor(
      cuda_memory_info_,
      depth.dino_tokens,
      sizeof(float) * Numel(dino_shape),
      dino_shape);
  Ort::Value intrinsics_value = MakeCudaTensor(
      cuda_memory_info_,
      const_cast<float*>(inputs.intrinsics),
      sizeof(float) * 9,
      intrinsics_shape);
  Ort::Value extrinsics_value = MakeCudaTensor(
      cuda_memory_info_,
      const_cast<float*>(inputs.extrinsics),
      sizeof(float) * 16,
      extrinsics_shape);

  Ort::IoBinding binding(gs_session_);
  binding.BindInput("image", image_value);
  binding.BindInput("depthmap", depth_value);
  binding.BindInput("dino_tokens", dino_value);
  binding.BindInput("intrinsics", intrinsics_value);
  binding.BindInput("extrinsics", extrinsics_value);
  binding.BindOutput("means", cuda_memory_info_);
  binding.BindOutput("harmonics", cuda_memory_info_);
  binding.BindOutput("opacities", cuda_memory_info_);
  binding.BindOutput("scales", cuda_memory_info_);
  binding.BindOutput("rotations", cuda_memory_info_);

  Ort::RunOptions run_options;
  gs_session_.Run(run_options, binding);
  std::vector<Ort::Value> gs_values = binding.GetOutputValues();
  if (gs_values.size() != gs_outputs.names.size()) {
    throw std::runtime_error("GS predictor returned an unexpected number of outputs.");
  }

  Ort::Value& harmonics_value = GetOutputByName(gs_values, gs_outputs.names, "harmonics");
  Ort::Value& opacities_value = GetOutputByName(gs_values, gs_outputs.names, "opacities");
  Ort::Value& scales_value = GetOutputByName(gs_values, gs_outputs.names, "scales");
  Ort::Value& rotations_value = GetOutputByName(gs_values, gs_outputs.names, "rotations");

  int64_t sample_count = depth.query_3d_uniform_coord_shape[1];
  SparseGaussiansCuda sparse = BuildSparseUniformGaussiansCuda(
      static_cast<const float*>(depth.query_3d_uniform_coord),
      static_cast<const float*>(depth.pred_depth_3d),
      inputs.intrinsics,
      inputs.extrinsics,
      harmonics_value.GetTensorData<float>(),
      opacities_value.GetTensorData<float>(),
      scales_value.GetTensorData<float>(),
      rotations_value.GetTensorData<float>(),
      static_cast<int>(inputs.height),
      static_cast<int>(inputs.width),
      sample_count);

  int64_t removed = 0;
  SparseGaussiansCuda filtered = FilterGaussiansByScalePercentileCuda(
      sparse,
      options_.filter_by_scale_percentile,
      options_.scale_min_percentile,
      options_.scale_max_percentile,
      &removed);

  std::vector<float> intrinsics_host = CopyDeviceF32(inputs.intrinsics, 9);
  std::vector<float> extrinsics_host = CopyDeviceF32(inputs.extrinsics, 16);
  if (extrinsics_host.empty()) {
    extrinsics_host = MakeIdentityExtrinsicsHost();
  }

  ExportSparseGaussiansPly(
      output_ply_path,
      filtered,
      intrinsics_host.data(),
      extrinsics_host.data(),
      static_cast<int>(inputs.height),
      static_cast<int>(inputs.width));

  GsPipelineResult result;
  result.gaussian_count = filtered.count;
  result.removed_count = removed;

  FreeSparseGaussiansCuda(&filtered);
  FreeSparseGaussiansCuda(&sparse);
  return result;
}

}  // namespace infinidepth
