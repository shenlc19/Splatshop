#pragma once

#include <cstdint>

namespace infinidepth {

struct InputProcessingCudaOptions {
  int prompt_samples = 1500;
  float min_prompt = 1.0f;
  float max_prompt = 100.0f;
  bool enable_depth_noise_filter = false;
  float filter_std_threshold = 0.8f;
  float filter_median_threshold = 0.5f;
  float filter_gradient_threshold = 0.5f;
  int filter_min_neighbors = 5;
};

void ResizeImageToNchwCuda(
    const uint8_t* image_rgb,
    int in_height,
    int in_width,
    float* image_nchw,
    int out_height,
    int out_width);

void ResizeDepthNearestCuda(
    const float* depth,
    int in_height,
    int in_width,
    float* resized_depth,
    int out_height,
    int out_width);

void FilterDepthNoiseCuda(
    float* depth,
    int height,
    int width,
    const InputProcessingCudaOptions& options);

void BuildDepthAndPromptCuda(
    const float* depth,
    int height,
    int width,
    const InputProcessingCudaOptions& options,
    float* gt_disp,
    uint8_t* gt_mask,
    float* prompt_disp,
    uint8_t* prompt_mask);

}  // namespace infinidepth
