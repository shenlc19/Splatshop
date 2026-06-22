#pragma once

#include <cstdint>

namespace infinidepth {

struct SparseGaussiansCuda {
  float* means = nullptr;
  float* harmonics = nullptr;
  float* opacities = nullptr;
  float* scales = nullptr;
  float* rotations = nullptr;
  int64_t count = 0;
};

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
    int64_t count);

SparseGaussiansCuda FilterGaussiansByScalePercentileCuda(
    const SparseGaussiansCuda& input,
    bool enabled = true,
    float min_percentile = 1.0f,
    float max_percentile = 99.0f,
    int64_t* removed_count = nullptr);

void FreeSparseGaussiansCuda(SparseGaussiansCuda* result);

}  // namespace infinidepth
