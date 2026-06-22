#include "infinidepth_input_processing_kernels.cuh"

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

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

struct SumSq {
  float sum;
  float sq;
  int count;
};

struct SumSqOp {
  const float* depth;
  float min_prompt;
  float max_prompt;

  __host__ __device__ SumSq operator()(int idx) const {
    float z = depth[idx];
    bool valid = isfinite(z) && z > min_prompt && z < max_prompt;
    return valid ? SumSq{z, z * z, 1} : SumSq{0.0f, 0.0f, 0};
  }
};

struct SumSqReduce {
  __host__ __device__ SumSq operator()(const SumSq& a, const SumSq& b) const {
    return SumSq{a.sum + b.sum, a.sq + b.sq, a.count + b.count};
  }
};

struct ValidDepthIndex {
  const float* depth;
  float min_prompt;
  float max_prompt;

  __host__ __device__ bool operator()(int idx) const {
    float z = depth[idx];
    return isfinite(z) && z > min_prompt && z < max_prompt && z > 0.1f;
  }
};

__device__ float Median25(float values[25]) {
  for (int i = 0; i <= 12; ++i) {
    int min_idx = i;
    for (int j = i + 1; j < 25; ++j) {
      if (values[j] < values[min_idx]) min_idx = j;
    }
    float tmp = values[i];
    values[i] = values[min_idx];
    values[min_idx] = tmp;
  }
  return values[12];
}

__global__ void ResizeImageToNchwKernel(
    const uint8_t* image_rgb,
    int in_height,
    int in_width,
    float* image_nchw,
    int out_height,
    int out_width,
    int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int c = idx / (out_height * out_width);
  int rem = idx - c * out_height * out_width;
  int y = rem / out_width;
  int x = rem - y * out_width;

  float scale_y = static_cast<float>(in_height) / static_cast<float>(out_height);
  float scale_x = static_cast<float>(in_width) / static_cast<float>(out_width);
  float src_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
  float src_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
  int y0 = max(0, min(in_height - 1, static_cast<int>(floorf(src_y))));
  int x0 = max(0, min(in_width - 1, static_cast<int>(floorf(src_x))));
  int y1 = max(0, min(in_height - 1, y0 + 1));
  int x1 = max(0, min(in_width - 1, x0 + 1));
  float wy = src_y - static_cast<float>(y0);
  float wx = src_x - static_cast<float>(x0);
  auto at = [&](int yy, int xx) {
    return static_cast<float>(image_rgb[(static_cast<int64_t>(yy) * in_width + xx) * 3 + c]) / 255.0f;
  };
  image_nchw[idx] =
      (1.0f - wx) * (1.0f - wy) * at(y0, x0) +
      wx * (1.0f - wy) * at(y0, x1) +
      (1.0f - wx) * wy * at(y1, x0) +
      wx * wy * at(y1, x1);
}

__global__ void ResizeDepthNearestKernel(
    const float* depth,
    int in_height,
    int in_width,
    float* resized_depth,
    int out_height,
    int out_width,
    int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int y = idx / out_width;
  int x = idx - y * out_width;
  int sy = min(in_height - 1, static_cast<int>(floorf((static_cast<float>(y) + 0.5f) * in_height / out_height)));
  int sx = min(in_width - 1, static_cast<int>(floorf((static_cast<float>(x) + 0.5f) * in_width / out_width)));
  resized_depth[idx] = depth[static_cast<int64_t>(sy) * in_width + sx];
}

__global__ void ThresholdDepthKernel(
    float* depth,
    uint8_t* mask,
    int total,
    float min_prompt,
    float max_prompt,
    float lo,
    float hi) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  float z = depth[idx];
  bool valid = isfinite(z) && z > min_prompt && z < max_prompt && z >= lo && z <= hi;
  mask[idx] = valid ? 1 : 0;
  if (!valid) depth[idx] = 0.0f;
}

__global__ void MedianConsistencyKernel(
    const float* depth,
    const uint8_t* in_mask,
    uint8_t* out_mask,
    int height,
    int width,
    float threshold,
    int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  if (!in_mask[idx]) {
    out_mask[idx] = 0;
    return;
  }
  int y = idx / width;
  int x = idx - y * width;
  float values[25];
  int n = 0;
  for (int dy = -2; dy <= 2; ++dy) {
    int yy = max(0, min(height - 1, y + dy));
    for (int dx = -2; dx <= 2; ++dx) {
      int xx = max(0, min(width - 1, x + dx));
      values[n++] = depth[yy * width + xx];
    }
  }
  float med = Median25(values);
  float rel = fabsf(depth[idx] - med) / (med + 1.0e-6f);
  out_mask[idx] = rel < threshold ? 1 : 0;
}

__global__ void GradientAndNeighborKernel(
    const float* depth,
    const uint8_t* in_mask,
    uint8_t* out_mask,
    int height,
    int width,
    float gradient_threshold,
    int min_neighbors,
    int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  if (!in_mask[idx]) {
    out_mask[idx] = 0;
    return;
  }
  int y = idx / width;
  int x = idx - y * width;
  auto at = [&](int yy, int xx) {
    yy = max(0, min(height - 1, yy));
    xx = max(0, min(width - 1, xx));
    return depth[yy * width + xx];
  };
  float dx = -at(y - 1, x - 1) - 2.0f * at(y, x - 1) - at(y + 1, x - 1) +
              at(y - 1, x + 1) + 2.0f * at(y, x + 1) + at(y + 1, x + 1);
  float dy = -at(y - 1, x - 1) - 2.0f * at(y - 1, x) - at(y - 1, x + 1) +
              at(y + 1, x - 1) + 2.0f * at(y + 1, x) + at(y + 1, x + 1);
  float norm_grad = sqrtf(dx * dx + dy * dy) / (depth[idx] + 1.0e-6f);
  int neighbors = 0;
  for (int oy = -1; oy <= 1; ++oy) {
    int yy = y + oy;
    if (yy < 0 || yy >= height) continue;
    for (int ox = -1; ox <= 1; ++ox) {
      int xx = x + ox;
      if (xx < 0 || xx >= width) continue;
      neighbors += in_mask[yy * width + xx] ? 1 : 0;
    }
  }
  out_mask[idx] = (norm_grad <= gradient_threshold && neighbors >= min_neighbors) ? 1 : 0;
}

__global__ void ApplyMaskKernel(float* depth, const uint8_t* mask, int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  if (!mask[idx]) depth[idx] = 0.0f;
}

__global__ void BuildGtDisparityKernel(
    const float* depth,
    float* gt_disp,
    uint8_t* gt_mask,
    int total,
    float min_prompt,
    float max_prompt) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  float z = depth[idx];
  bool valid = isfinite(z) && z > min_prompt && z < max_prompt;
  gt_mask[idx] = valid ? 1 : 0;
  gt_disp[idx] = valid ? 1.0f / z : 0.0f;
}

__global__ void ClearPromptKernel(float* prompt_disp, uint8_t* prompt_mask, int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  prompt_disp[idx] = 0.0f;
  prompt_mask[idx] = 0;
}

__global__ void MarkPromptSamplesKernel(
    const int* valid_indices,
    int valid_count,
    int sample_count,
    const float* gt_disp,
    float* prompt_disp,
    uint8_t* prompt_mask) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= sample_count) return;
  int src = static_cast<int>((static_cast<int64_t>(k) * valid_count) / sample_count);
  src = min(src, valid_count - 1);
  int idx = valid_indices[src];
  prompt_disp[idx] = gt_disp[idx];
  prompt_mask[idx] = 1;
}

}  // namespace

void ResizeImageToNchwCuda(
    const uint8_t* image_rgb,
    int in_height,
    int in_width,
    float* image_nchw,
    int out_height,
    int out_width) {
  int total = 3 * out_height * out_width;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  ResizeImageToNchwKernel<<<blocks, threads>>>(image_rgb, in_height, in_width, image_nchw, out_height, out_width, total);
  CUDA_CHECK(cudaGetLastError());
}

void ResizeDepthNearestCuda(
    const float* depth,
    int in_height,
    int in_width,
    float* resized_depth,
    int out_height,
    int out_width) {
  int total = out_height * out_width;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  ResizeDepthNearestKernel<<<blocks, threads>>>(depth, in_height, in_width, resized_depth, out_height, out_width, total);
  CUDA_CHECK(cudaGetLastError());
}

void FilterDepthNoiseCuda(
    float* depth,
    int height,
    int width,
    const InputProcessingCudaOptions& options) {
  int total = height * width;
  auto begin = thrust::make_counting_iterator(0);
  SumSq stats = thrust::transform_reduce(
      thrust::device,
      begin,
      begin + total,
      SumSqOp{depth, options.min_prompt, options.max_prompt},
      SumSq{0.0f, 0.0f, 0},
      SumSqReduce{});
  if (stats.count <= 0) return;
  float mean = stats.sum / static_cast<float>(stats.count);
  float var = fmaxf(stats.sq / static_cast<float>(stats.count) - mean * mean, 0.0f);
  float stddev = sqrtf(var);
  float lo = mean - options.filter_std_threshold * stddev;
  float hi = mean + options.filter_std_threshold * stddev;

  uint8_t* mask_a = nullptr;
  uint8_t* mask_b = nullptr;
  CUDA_CHECK(cudaMalloc(&mask_a, total * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&mask_b, total * sizeof(uint8_t)));
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  ThresholdDepthKernel<<<blocks, threads>>>(depth, mask_a, total, options.min_prompt, options.max_prompt, lo, hi);
  CUDA_CHECK(cudaGetLastError());
  MedianConsistencyKernel<<<blocks, threads>>>(depth, mask_a, mask_b, height, width, options.filter_median_threshold, total);
  CUDA_CHECK(cudaGetLastError());
  ApplyMaskKernel<<<blocks, threads>>>(depth, mask_b, total);
  CUDA_CHECK(cudaGetLastError());
  GradientAndNeighborKernel<<<blocks, threads>>>(
      depth,
      mask_b,
      mask_a,
      height,
      width,
      options.filter_gradient_threshold,
      options.filter_min_neighbors,
      total);
  CUDA_CHECK(cudaGetLastError());
  ApplyMaskKernel<<<blocks, threads>>>(depth, mask_a, total);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaFree(mask_a));
  CUDA_CHECK(cudaFree(mask_b));
}

void BuildDepthAndPromptCuda(
    const float* depth,
    int height,
    int width,
    const InputProcessingCudaOptions& options,
    float* gt_disp,
    uint8_t* gt_mask,
    float* prompt_disp,
    uint8_t* prompt_mask) {
  int total = height * width;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  BuildGtDisparityKernel<<<blocks, threads>>>(depth, gt_disp, gt_mask, total, options.min_prompt, options.max_prompt);
  CUDA_CHECK(cudaGetLastError());
  ClearPromptKernel<<<blocks, threads>>>(prompt_disp, prompt_mask, total);
  CUDA_CHECK(cudaGetLastError());

  int valid_count = static_cast<int>(thrust::count_if(
      thrust::device,
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(total),
      ValidDepthIndex{depth, options.min_prompt, options.max_prompt}));
  if (valid_count <= 0) return;

  int sample_count = std::min(valid_count, std::max(options.prompt_samples, 0));
  if (sample_count <= 0) return;
  int* valid_indices = nullptr;
  CUDA_CHECK(cudaMalloc(&valid_indices, static_cast<size_t>(valid_count) * sizeof(int)));
  auto out_end = thrust::copy_if(
      thrust::device,
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(total),
      thrust::device_pointer_cast(valid_indices),
      ValidDepthIndex{depth, options.min_prompt, options.max_prompt});
  (void)out_end;

  int sample_blocks = (sample_count + threads - 1) / threads;
  MarkPromptSamplesKernel<<<sample_blocks, threads>>>(
      valid_indices, valid_count, sample_count, gt_disp, prompt_disp, prompt_mask);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaFree(valid_indices));
}

}  // namespace infinidepth
