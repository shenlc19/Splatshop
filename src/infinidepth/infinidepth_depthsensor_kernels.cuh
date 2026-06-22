#pragma once

#include <cstdint>

namespace infinidepth {

struct PreparedDepthSensorInputs {
  float* prompt_depth = nullptr;    // [B, 1, H, W], warped
  uint8_t* prompt_mask = nullptr;   // [B, 1, H, W], warped mask
  float* reference_meta = nullptr;  // [B]
};

struct UniformCoordResult {
  float* coords_yx = nullptr;  // [N, 2], order (y, x), range [-1, 1]
  int64_t count = 0;
};

PreparedDepthSensorInputs PrepareDepthSensorInputsCuda(
    const float* gt_depth,
    const uint8_t* gt_depth_mask,
    const float* prompt_depth,
    const uint8_t* prompt_mask,
    int64_t batch,
    int64_t height,
    int64_t width);

void PostprocessDepthSensorCuda(
    const float* pred,
    const float* reference_meta,
    float* pred_depth,
    float* pred_disparity,
    int64_t batch,
    int64_t num_queries);

void GatherQueryChunkCuda(
    const float* query_coord,
    float* query_chunk,
    int64_t batch,
    int64_t num_queries,
    int64_t start,
    int64_t chunk);

void ScatterPredChunkCuda(
    const float* pred_chunk,
    float* pred,
    int64_t batch,
    int64_t num_queries,
    int64_t start,
    int64_t chunk);

void MakeDenseQueryCoordCuda(
    float* query_coord,
    int64_t batch,
    int64_t height,
    int64_t width);

UniformCoordResult Make3DUniformCoordTriangleCuda(
    const float* depth,
    int height,
    int width,
    float fx,
    float fy,
    float cx,
    float cy,
    int64_t sample_count);

void FreeUniformCoordResult(UniformCoordResult* result);

void FreePreparedDepthSensorInputs(PreparedDepthSensorInputs* prepared);

}  // namespace infinidepth
