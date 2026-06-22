#pragma once

#include "infinidepth_depthsensor_onnx.h"

#include <filesystem>
#include <string>

namespace infinidepth {

struct GsPipelineOptions {
  int device_id = 0;
  bool disable_graph_optimizations = true;
  std::string cudnn_conv_algo_search = "HEURISTIC";
  bool cudnn_conv_use_max_workspace = false;
  bool cuda_use_tf32 = true;
  bool filter_by_scale_percentile = true;
  float scale_min_percentile = 1.0f;
  float scale_max_percentile = 99.0f;
};

struct GsPipelineInputs {
  const float* image = nullptr;       // CUDA [1, 3, H, W]
  const float* intrinsics = nullptr;  // CUDA [1, 3, 3], pixel units
  const float* extrinsics = nullptr;  // CUDA [1, 4, 4], camera-to-world
  const InferenceForGsOutputs* depth_outputs = nullptr;
  int64_t batch = 1;
  int64_t height = 0;
  int64_t width = 0;
};

struct GsPipelineResult {
  int64_t gaussian_count = 0;
  int64_t removed_count = 0;
};

class InfiniDepthGsPipeline {
 public:
  explicit InfiniDepthGsPipeline(std::string gs_onnx_path, GsPipelineOptions options = {});

  GsPipelineResult RunAndExportPly(
      const GsPipelineInputs& inputs,
      const std::filesystem::path& output_ply_path);

 private:
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  Ort::MemoryInfo cuda_memory_info_;
  Ort::Session gs_session_;
  GsPipelineOptions options_;
};

}  // namespace infinidepth
