#pragma once

#include <cstdint>
#include <string>

namespace infinidepth {

struct ProcessedGsInputsCuda {
  float* image = nullptr;       // [1, 3, H, W]
  float* intrinsics = nullptr;  // [1, 3, 3]
  float* extrinsics = nullptr;  // [1, 4, 4]
  float* gt = nullptr;          // [1, 1, H, W], disparity
  uint8_t* gt_mask = nullptr;   // [1, 1, H, W]
  float* prompt = nullptr;      // [1, 1, H, W], sparse disparity
  uint8_t* prompt_mask = nullptr;
  int64_t batch = 1;
  int64_t height = 0;
  int64_t width = 0;
  int64_t original_height = 0;
  int64_t original_width = 0;
};

struct InputProcessingOptions {
  std::string image_path;
  std::string depth_path;
  std::string camera_json_path;
  int target_height = 768;
  int target_width = 1024;
  int prompt_samples = 1500;
  float min_prompt = 1.0f;
  float max_prompt = 100.0f;
  bool enable_depth_noise_filter = false;
  float filter_std_threshold = 0.8f;
  float filter_median_threshold = 0.5f;
  float filter_gradient_threshold = 0.5f;
  int filter_min_neighbors = 5;
};

ProcessedGsInputsCuda ProcessRawGsInputsCuda(const InputProcessingOptions& options);
void FreeProcessedGsInputsCuda(ProcessedGsInputsCuda* inputs);

}  // namespace infinidepth
