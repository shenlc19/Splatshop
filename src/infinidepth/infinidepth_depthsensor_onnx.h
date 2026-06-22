#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#ifndef ORT_API_MANUAL_INIT
#define ORT_API_MANUAL_INIT
#endif
#include <onnxruntime_cxx_api.h>

namespace infinidepth {

struct TensorShape {
  int64_t b = 1;
  int64_t c = 1;
  int64_t h = 1;
  int64_t w = 1;
};

struct InferenceInputs {
  // All pointers are CUDA device pointers.
  const float* image = nullptr;          // [B, 3, H, W], RGB, 0..1
  const float* query_coord = nullptr;    // [B, N, 2], order (y, x), range [-1, 1]
  const float* gt_depth = nullptr;       // [B, 1, H, W], disparity, not metric depth
  const uint8_t* gt_depth_mask = nullptr;// [B, 1, H, W], bool-like 0/1
  const float* prompt_depth = nullptr;   // [B, 1, H, W], disparity, not metric depth
  const uint8_t* prompt_mask = nullptr;  // [B, 1, H, W], bool-like 0/1

  int64_t batch = 1;
  int64_t height = 0;
  int64_t width = 0;
  int64_t num_queries = 0;
};

struct InferenceForGsInputs {
  // All pointers are CUDA device pointers.
  const float* image = nullptr;           // [B, 3, H, W], RGB, 0..1
  const float* intrinsics = nullptr;      // [B, 3, 3]
  const float* gt_depth = nullptr;        // [B, 1, H, W], disparity, not metric depth
  const uint8_t* gt_depth_mask = nullptr; // [B, 1, H, W], bool-like 0/1
  const float* prompt_depth = nullptr;    // [B, 1, H, W], disparity, not metric depth
  const uint8_t* prompt_mask = nullptr;   // [B, 1, H, W], bool-like 0/1

  int64_t batch = 1;
  int64_t height = 0;
  int64_t width = 0;
  int64_t sample_point_num = 2000000;
};

struct InferenceOutputs {
  // CUDA device buffers owned by this struct.
  std::vector<float> reference_meta_host;  // [B], copied back for debugging/logging.
  void* pred_depth = nullptr;              // float [B, N, 1]
  void* pred_disparity = nullptr;          // float [B, N, 1]
  void* dino_tokens = nullptr;             // float [B, T, C]

  std::vector<int64_t> pred_shape;
  std::vector<int64_t> dino_tokens_shape;

  ~InferenceOutputs();
  InferenceOutputs() = default;
  InferenceOutputs(const InferenceOutputs&) = delete;
  InferenceOutputs& operator=(const InferenceOutputs&) = delete;
  InferenceOutputs(InferenceOutputs&& other) noexcept;
  InferenceOutputs& operator=(InferenceOutputs&& other) noexcept;
};

struct InferenceForGsOutputs {
  // CUDA device buffers owned by this struct.
  void* depthmap = nullptr;                // float [B, 1, H, W]
  void* dino_tokens = nullptr;             // float [B, T, C]
  void* query_3d_uniform_coord = nullptr;  // float [B, N, 2], order (y, x)
  void* pred_depth_3d = nullptr;           // float [B, N, 1]

  std::vector<int64_t> depthmap_shape;
  std::vector<int64_t> dino_tokens_shape;
  std::vector<int64_t> query_3d_uniform_coord_shape;
  std::vector<int64_t> pred_depth_3d_shape;
  std::vector<float> reference_meta_host;

  ~InferenceForGsOutputs();
  InferenceForGsOutputs() = default;
  InferenceForGsOutputs(const InferenceForGsOutputs&) = delete;
  InferenceForGsOutputs& operator=(const InferenceForGsOutputs&) = delete;
  InferenceForGsOutputs(InferenceForGsOutputs&& other) noexcept;
  InferenceForGsOutputs& operator=(InferenceForGsOutputs&& other) noexcept;
};

struct RuntimeOptions {
  int device_id = 0;
  int64_t decoder_chunk_size = 10000;
  bool disable_graph_optimizations = true;
  std::string cudnn_conv_algo_search = "HEURISTIC";
  bool cudnn_conv_use_max_workspace = false;
  bool cuda_use_tf32 = true;
};

class InfiniDepthDepthSensorOnnx {
 public:
  InfiniDepthDepthSensorOnnx(
      const std::string& encoder_onnx_path,
      const std::string& decoder_onnx_path,
      RuntimeOptions options = {});

  InferenceOutputs Infer(const InferenceInputs& inputs);
  InferenceForGsOutputs InferenceForGs(const InferenceForGsInputs& inputs);

 private:
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  Ort::MemoryInfo cuda_memory_info_;
  Ort::Session encoder_session_;
  Ort::Session decoder_session_;
  RuntimeOptions options_;
};

}  // namespace infinidepth
