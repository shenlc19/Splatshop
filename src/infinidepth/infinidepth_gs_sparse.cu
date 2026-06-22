#include "infinidepth_gs_sparse.cuh"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cmath>
#include <limits>
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

__device__ float SampleBilinearZeroPadding(
    const float* data,
    int height,
    int width,
    int channels,
    float y_norm,
    float x_norm,
    int channel) {
  float src_x = ((x_norm + 1.0f) * static_cast<float>(width) - 1.0f) * 0.5f;
  float src_y = ((y_norm + 1.0f) * static_cast<float>(height) - 1.0f) * 0.5f;

  int x0 = static_cast<int>(floorf(src_x));
  int y0 = static_cast<int>(floorf(src_y));
  int x1 = x0 + 1;
  int y1 = y0 + 1;
  float wx = src_x - static_cast<float>(x0);
  float wy = src_y - static_cast<float>(y0);

  float value = 0.0f;
  if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
    value += (1.0f - wx) * (1.0f - wy) * data[(static_cast<int64_t>(y0) * width + x0) * channels + channel];
  }
  if (x1 >= 0 && x1 < width && y0 >= 0 && y0 < height) {
    value += wx * (1.0f - wy) * data[(static_cast<int64_t>(y0) * width + x1) * channels + channel];
  }
  if (x0 >= 0 && x0 < width && y1 >= 0 && y1 < height) {
    value += (1.0f - wx) * wy * data[(static_cast<int64_t>(y1) * width + x0) * channels + channel];
  }
  if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
    value += wx * wy * data[(static_cast<int64_t>(y1) * width + x1) * channels + channel];
  }
  return value;
}

__global__ void BuildSparseGaussiansKernel(
    const float* query_yx,
    const float* pred_depth,
    const float* intrinsics,
    const float* extrinsics,
    const float* dense_harmonics,
    const float* dense_opacities,
    const float* dense_scales,
    const float* dense_rotations,
    float* means,
    float* harmonics,
    float* opacities,
    float* scales,
    float* rotations,
    int height,
    int width,
    int64_t count) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;

  float y_norm = query_yx[i * 2 + 0];
  float x_norm = query_yx[i * 2 + 1];
  float py = ((y_norm + 1.0f) * (static_cast<float>(height) * 0.5f)) - 0.5f;
  float px = ((x_norm + 1.0f) * (static_cast<float>(width) * 0.5f)) - 0.5f;
  float depth = pred_depth[i];

  float fx = intrinsics[0];
  float fy = intrinsics[4];
  float cx = intrinsics[2];
  float cy = intrinsics[5];

  float x_cam = (px - cx) / fx * depth;
  float y_cam = (py - cy) / fy * depth;
  float z_cam = depth;

  means[i * 3 + 0] = extrinsics[0] * x_cam + extrinsics[1] * y_cam + extrinsics[2] * z_cam + extrinsics[3];
  means[i * 3 + 1] = extrinsics[4] * x_cam + extrinsics[5] * y_cam + extrinsics[6] * z_cam + extrinsics[7];
  means[i * 3 + 2] = extrinsics[8] * x_cam + extrinsics[9] * y_cam + extrinsics[10] * z_cam + extrinsics[11];

  opacities[i] = SampleBilinearZeroPadding(dense_opacities, height, width, 1, y_norm, x_norm, 0);
  for (int c = 0; c < 3; ++c) {
    scales[i * 3 + c] = SampleBilinearZeroPadding(dense_scales, height, width, 3, y_norm, x_norm, c);
  }
  for (int c = 0; c < 27; ++c) {
    harmonics[i * 27 + c] = SampleBilinearZeroPadding(dense_harmonics, height, width, 27, y_norm, x_norm, c);
  }

  float norm_sq = 0.0f;
  float rotation[4];
  for (int c = 0; c < 4; ++c) {
    rotation[c] = SampleBilinearZeroPadding(dense_rotations, height, width, 4, y_norm, x_norm, c);
    norm_sq += rotation[c] * rotation[c];
  }
  float inv_norm = 1.0f / (sqrtf(norm_sq) + 1.0e-8f);
  for (int c = 0; c < 4; ++c) {
    rotations[i * 4 + c] = rotation[c] * inv_norm;
  }
}

__global__ void MaxScaleKernel(const float* scales, float* max_scale, int64_t count) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  max_scale[i] = fmaxf(scales[i * 3 + 0], fmaxf(scales[i * 3 + 1], scales[i * 3 + 2]));
}

__global__ void MarkScaleKeepKernel(const float* max_scale, int* keep, float lo, float hi, int64_t count) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  float value = max_scale[i];
  keep[i] = (value >= lo && value <= hi) ? 1 : 0;
}

__global__ void CompactSparseGaussiansKernel(
    const float* in_means,
    const float* in_harmonics,
    const float* in_opacities,
    const float* in_scales,
    const float* in_rotations,
    const int* keep,
    const int* prefix,
    float* out_means,
    float* out_harmonics,
    float* out_opacities,
    float* out_scales,
    float* out_rotations,
    int64_t count) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count || keep[i] == 0) return;

  int64_t dst = static_cast<int64_t>(prefix[i]);
  for (int c = 0; c < 3; ++c) {
    out_means[dst * 3 + c] = in_means[i * 3 + c];
    out_scales[dst * 3 + c] = in_scales[i * 3 + c];
  }
  out_opacities[dst] = in_opacities[i];
  for (int c = 0; c < 27; ++c) {
    out_harmonics[dst * 27 + c] = in_harmonics[i * 27 + c];
  }
  for (int c = 0; c < 4; ++c) {
    out_rotations[dst * 4 + c] = in_rotations[i * 4 + c];
  }
}

}  // namespace

void FreeSparseGaussiansCuda(SparseGaussiansCuda* result) {
  if (!result) return;
  if (result->means) cudaFree(result->means);
  if (result->harmonics) cudaFree(result->harmonics);
  if (result->opacities) cudaFree(result->opacities);
  if (result->scales) cudaFree(result->scales);
  if (result->rotations) cudaFree(result->rotations);
  result->means = nullptr;
  result->harmonics = nullptr;
  result->opacities = nullptr;
  result->scales = nullptr;
  result->rotations = nullptr;
  result->count = 0;
}

SparseGaussiansCuda BuildSparseUniformGaussiansCuda(
    const float* query_yx,
    const float* pred_depth,
    const float* intrinsics,
    const float* extrinsics,
    const float* dense_harmonics,
    const float* dense_opacities,
    const float* dense_scales,
    const float* dense_rotations,
    int height,
    int width,
    int64_t count) {
  SparseGaussiansCuda result;
  result.count = count;
  CUDA_CHECK(cudaMalloc(&result.means, static_cast<size_t>(count) * 3 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&result.harmonics, static_cast<size_t>(count) * 27 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&result.opacities, static_cast<size_t>(count) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&result.scales, static_cast<size_t>(count) * 3 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&result.rotations, static_cast<size_t>(count) * 4 * sizeof(float)));

  constexpr int threads = 256;
  int blocks = static_cast<int>((count + threads - 1) / threads);
  BuildSparseGaussiansKernel<<<blocks, threads>>>(
      query_yx,
      pred_depth,
      intrinsics,
      extrinsics,
      dense_harmonics,
      dense_opacities,
      dense_scales,
      dense_rotations,
      result.means,
      result.harmonics,
      result.opacities,
      result.scales,
      result.rotations,
      height,
      width,
      count);
  CUDA_CHECK(cudaGetLastError());
  return result;
}

SparseGaussiansCuda FilterGaussiansByScalePercentileCuda(
    const SparseGaussiansCuda& input,
    bool enabled,
    float min_percentile,
    float max_percentile,
    int64_t* removed_count) {
  if (removed_count) *removed_count = 0;
  if (input.count <= 0) return {};

  int64_t count = input.count;
  constexpr int threads = 256;
  int blocks = static_cast<int>((count + threads - 1) / threads);

  float* max_scale = nullptr;
  float* sorted_scale = nullptr;
  int* keep = nullptr;
  int* prefix = nullptr;
  CUDA_CHECK(cudaMalloc(&max_scale, static_cast<size_t>(count) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&sorted_scale, static_cast<size_t>(count) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&keep, static_cast<size_t>(count) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&prefix, static_cast<size_t>(count) * sizeof(int)));

  MaxScaleKernel<<<blocks, threads>>>(input.scales, max_scale, count);
  CUDA_CHECK(cudaGetLastError());

  constexpr float kAllMin = -std::numeric_limits<float>::max();
  constexpr float kAllMax = std::numeric_limits<float>::max();
  float lo = kAllMin;
  float hi = kAllMax;
  if (enabled) {
    CUDA_CHECK(cudaMemcpy(sorted_scale, max_scale, static_cast<size_t>(count) * sizeof(float), cudaMemcpyDeviceToDevice));
    thrust::device_ptr<float> sorted_begin(sorted_scale);
    thrust::sort(thrust::device, sorted_begin, sorted_begin + count);

    float q_min = std::max(0.0f, std::min(1.0f, min_percentile * 0.01f));
    float q_max = std::max(0.0f, std::min(1.0f, max_percentile * 0.01f));
    float pos_lo = q_min * static_cast<float>(count - 1);
    float pos_hi = q_max * static_cast<float>(count - 1);
    int lo0 = static_cast<int>(std::floor(pos_lo));
    int lo1 = static_cast<int>(std::ceil(pos_lo));
    int hi0 = static_cast<int>(std::floor(pos_hi));
    int hi1 = static_cast<int>(std::ceil(pos_hi));

    float lo_pair[2] = {};
    float hi_pair[2] = {};
    CUDA_CHECK(cudaMemcpy(lo_pair, sorted_scale + lo0, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(lo_pair + 1, sorted_scale + lo1, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hi_pair, sorted_scale + hi0, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hi_pair + 1, sorted_scale + hi1, sizeof(float), cudaMemcpyDeviceToHost));

    float lo_t = pos_lo - static_cast<float>(lo0);
    float hi_t = pos_hi - static_cast<float>(hi0);
    lo = lo_pair[0] + (lo_pair[1] - lo_pair[0]) * lo_t;
    hi = hi_pair[0] + (hi_pair[1] - hi_pair[0]) * hi_t;
  }

  MarkScaleKeepKernel<<<blocks, threads>>>(max_scale, keep, lo, hi, count);
  CUDA_CHECK(cudaGetLastError());

  thrust::device_ptr<int> keep_begin(keep);
  thrust::device_ptr<int> prefix_begin(prefix);
  thrust::exclusive_scan(thrust::device, keep_begin, keep_begin + count, prefix_begin);

  int last_keep = 0;
  int last_prefix = 0;
  CUDA_CHECK(cudaMemcpy(&last_keep, keep + count - 1, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&last_prefix, prefix + count - 1, sizeof(int), cudaMemcpyDeviceToHost));
  int64_t kept_count = static_cast<int64_t>(last_prefix + last_keep);

  if (kept_count == 0) {
    kept_count = count;
    MarkScaleKeepKernel<<<blocks, threads>>>(max_scale, keep, kAllMin, kAllMax, count);
    CUDA_CHECK(cudaGetLastError());
    thrust::exclusive_scan(thrust::device, keep_begin, keep_begin + count, prefix_begin);
  }

  SparseGaussiansCuda output;
  output.count = kept_count;
  CUDA_CHECK(cudaMalloc(&output.means, static_cast<size_t>(kept_count) * 3 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&output.harmonics, static_cast<size_t>(kept_count) * 27 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&output.opacities, static_cast<size_t>(kept_count) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&output.scales, static_cast<size_t>(kept_count) * 3 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&output.rotations, static_cast<size_t>(kept_count) * 4 * sizeof(float)));

  CompactSparseGaussiansKernel<<<blocks, threads>>>(
      input.means,
      input.harmonics,
      input.opacities,
      input.scales,
      input.rotations,
      keep,
      prefix,
      output.means,
      output.harmonics,
      output.opacities,
      output.scales,
      output.rotations,
      count);
  CUDA_CHECK(cudaGetLastError());

  if (removed_count) *removed_count = count - kept_count;
  cudaFree(max_scale);
  cudaFree(sorted_scale);
  cudaFree(keep);
  cudaFree(prefix);
  return output;
}

}  // namespace infinidepth
