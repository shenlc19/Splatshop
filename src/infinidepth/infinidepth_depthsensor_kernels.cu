#include "infinidepth_depthsensor_kernels.cuh"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

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

constexpr float kEps = 1.0e-6f;

struct FaceScore {
  float score;
  int i0;
  int j0;
  int i1;
  int j1;
  int i2;
  int j2;
};

__global__ void CountValidKernel(
    const uint8_t* mask,
    const float* values,
    int64_t pixels,
    bool require_positive,
    int* count) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= pixels) return;
  bool valid = mask[idx] != 0;
  if (require_positive) valid = valid && values[idx] > 0.0f;
  if (valid) atomicAdd(count, 1);
}

__global__ void CompactValidKernel(
    const float* values,
    const uint8_t* mask,
    int64_t pixels,
    bool require_positive,
    float* compact,
    int* count) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= pixels) return;
  bool valid = mask[idx] != 0;
  if (require_positive) valid = valid && values[idx] > 0.0f;
  if (!valid) return;
  int out = atomicAdd(count, 1);
  compact[out] = values[idx];
}

__global__ void WarpPromptKernel(
    const float* prompt_depth,
    const uint8_t* prompt_mask,
    const float* reference_meta,
    float* out_depth,
    uint8_t* out_mask,
    int64_t pixels_per_batch,
    int64_t total_pixels) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_pixels) return;
  int64_t b = idx / pixels_per_batch;
  float median = fmaxf(reference_meta[b], 1.0e-2f);
  float depth = prompt_depth[idx];
  out_depth[idx] = depth / median;
  out_mask[idx] = (depth >= 0.0f && prompt_mask[idx] != 0) ? 1 : 0;
}

__global__ void PostprocessKernel(
    const float* pred,
    const float* reference_meta,
    float* pred_depth,
    float* pred_disparity,
    int64_t num_queries,
    int64_t total) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int64_t b = idx / num_queries;
  float unwarped = pred[idx] * fmaxf(reference_meta[b], 1.0e-3f);
  pred_disparity[idx] = unwarped;
  pred_depth[idx] = 1.0f / fmaxf(unwarped, 5.0e-3f);
}

__global__ void GatherQueryChunkKernel(
    const float* query_coord,
    float* query_chunk,
    int64_t num_queries,
    int64_t start,
    int64_t chunk,
    int64_t total) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int64_t b = idx / (chunk * 2);
  int64_t rem = idx % (chunk * 2);
  int64_t q = rem / 2;
  int64_t xy = rem % 2;
  query_chunk[idx] = query_coord[(b * num_queries + start + q) * 2 + xy];
}

__global__ void ScatterPredChunkKernel(
    const float* pred_chunk,
    float* pred,
    int64_t num_queries,
    int64_t start,
    int64_t chunk,
    int64_t total) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int64_t b = idx / chunk;
  int64_t q = idx % chunk;
  pred[b * num_queries + start + q] = pred_chunk[idx];
}

__global__ void MakeDenseQueryCoordKernel(
    float* query_coord,
    int64_t height,
    int64_t width,
    int64_t pixels,
    int64_t total) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int64_t p = idx % pixels;
  int64_t y = p / width;
  int64_t x = p - y * width;
  float yf = ((static_cast<float>(y) + 0.5f) / fmaxf(static_cast<float>(height), 1.0f)) * 2.0f - 1.0f;
  float xf = ((static_cast<float>(x) + 0.5f) / fmaxf(static_cast<float>(width), 1.0f)) * 2.0f - 1.0f;
  query_coord[idx * 2 + 0] = yf;
  query_coord[idx * 2 + 1] = xf;
}

__device__ inline float3 VertexFromDepth(
    const float* depth,
    int y,
    int x,
    int width,
    float fx,
    float fy,
    float cx,
    float cy) {
  int idx = y * width + x;
  float z = depth[idx];
  return make_float3(
      (static_cast<float>(x) - cx) / fx * z,
      (static_cast<float>(y) - cy) / fy * z,
      z);
}

__device__ inline float3 Sub3(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 Cross3(float3 a, float3 b) {
  return make_float3(
      a.y * b.z - a.z * b.y,
      a.z * b.x - a.x * b.z,
      a.x * b.y - a.y * b.x);
}

__global__ void ScoreFacesKernel(
    const float* depth,
    FaceScore* scores,
    int height,
    int width,
    float fx,
    float fy,
    float cx,
    float cy) {
  int cells = (height - 1) * (width - 1);
  int face_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (face_id >= cells * 2) return;

  int cell = face_id / 2;
  int tri = face_id - cell * 2;
  int y = cell / (width - 1);
  int x = cell - y * (width - 1);

  int i0 = y;
  int j0 = x;
  int i1 = y + 1;
  int j1 = x;
  int i2 = y;
  int j2 = x + 1;
  if (tri == 1) {
    i0 = y + 1;
    j0 = x + 1;
    i1 = y;
    j1 = x + 1;
    i2 = y + 1;
    j2 = x;
  }

  float3 a = VertexFromDepth(depth, i0, j0, width, fx, fy, cx, cy);
  float3 b = VertexFromDepth(depth, i1, j1, width, fx, fy, cx, cy);
  float3 c = VertexFromDepth(depth, i2, j2, width, fx, fy, cx, cy);

  float zmin = fminf(a.z, fminf(b.z, c.z));
  float zmax = fmaxf(a.z, fmaxf(b.z, c.z));
  float3 cr = Cross3(Sub3(b, a), Sub3(c, a));
  float area = 0.5f * sqrtf(cr.x * cr.x + cr.y * cr.y + cr.z * cr.z);

  bool keep = isfinite(area) && zmin > 0.0f && zmax < 90.0f && zmax / fmaxf(zmin, 1.0e-9f) < 1.05f;
  if (!keep) area = 0.0f;
  scores[face_id] = FaceScore{area, i0, j0, i1, j1, i2, j2};
}

std::vector<float> MakeBaryTemplate(size_t limit = 2000) {
  std::vector<float> bary;
  bary.reserve(limit * 3);
  for (int subdiv = 1; subdiv <= 50 && bary.size() / 3 < limit; ++subdiv) {
    for (int a = 0; a <= subdiv && bary.size() / 3 < limit; ++a) {
      for (int b = 0; b <= subdiv - a && bary.size() / 3 < limit; ++b) {
        int c = subdiv - a - b;
        float w0 = static_cast<float>(a) / subdiv;
        float w1 = static_cast<float>(b) / subdiv;
        float w2 = static_cast<float>(c) / subdiv;
        if (std::abs(w0 - 1.0f / 3.0f) < 0.01f && std::abs(w1 - 1.0f / 3.0f) < 0.01f) {
          continue;
        }
        bary.push_back(w0);
        bary.push_back(w1);
        bary.push_back(w2);
      }
    }
  }
  return bary;
}

void AppendCoordFromFace(
    std::vector<float>& coords_yx,
    const FaceScore& face,
    float w0,
    float w1,
    float w2,
    int height,
    int width) {
  float i = w0 * face.i0 + w1 * face.i1 + w2 * face.i2;
  float j = w0 * face.j0 + w1 * face.j1 + w2 * face.j2;
  float x = 2.0f * ((j + 0.5f) / static_cast<float>(width)) - 1.0f;
  float y = 2.0f * ((i + 0.5f) / static_cast<float>(height)) - 1.0f;
  coords_yx.push_back(y);
  coords_yx.push_back(x);
}

int CountValid(
    const float* values,
    const uint8_t* mask,
    int64_t pixels,
    bool require_positive) {
  int* d_count = nullptr;
  CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));
  int threads = 256;
  int blocks = static_cast<int>((pixels + threads - 1) / threads);
  CountValidKernel<<<blocks, threads>>>(mask, values, pixels, require_positive, d_count);
  CUDA_CHECK(cudaGetLastError());
  int h_count = 0;
  CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_count));
  return h_count;
}

float MedianOfMaskedValues(
    const float* values,
    const uint8_t* mask,
    int64_t pixels,
    bool require_positive) {
  int count = CountValid(values, mask, pixels, require_positive);
  if (count <= 0) {
    return 1.0f;
  }

  float* compact = nullptr;
  int* d_count = nullptr;
  CUDA_CHECK(cudaMalloc(&compact, sizeof(float) * count));
  CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

  int threads = 256;
  int blocks = static_cast<int>((pixels + threads - 1) / threads);
  CompactValidKernel<<<blocks, threads>>>(values, mask, pixels, require_positive, compact, d_count);
  CUDA_CHECK(cudaGetLastError());

  thrust::device_ptr<float> begin(compact);
  thrust::sort(begin, begin + count);

  int median_idx = count / 2;
  float median = 1.0f;
  if (count % 2 == 1) {
    CUDA_CHECK(cudaMemcpy(&median, compact + median_idx, sizeof(float), cudaMemcpyDeviceToHost));
  } else {
    float pair[2];
    CUDA_CHECK(cudaMemcpy(pair, compact + median_idx - 1, 2 * sizeof(float), cudaMemcpyDeviceToHost));
    median = 0.5f * (pair[0] + pair[1]);
  }

  CUDA_CHECK(cudaFree(compact));
  CUDA_CHECK(cudaFree(d_count));
  return median;
}

}  // namespace

PreparedDepthSensorInputs PrepareDepthSensorInputsCuda(
    const float* gt_depth,
    const uint8_t* gt_depth_mask,
    const float* prompt_depth,
    const uint8_t* prompt_mask,
    int64_t batch,
    int64_t height,
    int64_t width) {
  if (batch <= 0 || height <= 0 || width <= 0) {
    throw std::invalid_argument("Invalid input shape for PrepareDepthSensorInputsCuda.");
  }

  int64_t pixels = height * width;
  PreparedDepthSensorInputs prepared;
  CUDA_CHECK(cudaMalloc(&prepared.prompt_depth, sizeof(float) * batch * pixels));
  CUDA_CHECK(cudaMalloc(&prepared.prompt_mask, sizeof(uint8_t) * batch * pixels));
  CUDA_CHECK(cudaMalloc(&prepared.reference_meta, sizeof(float) * batch));

  std::vector<float> reference_host(batch, 1.0f);
  for (int64_t b = 0; b < batch; ++b) {
    const float* prompt_b = prompt_depth + b * pixels;
    const uint8_t* prompt_mask_b = prompt_mask + b * pixels;
    const float* gt_b = gt_depth + b * pixels;
    const uint8_t* gt_mask_b = gt_depth_mask + b * pixels;

    int prompt_count = CountValid(prompt_b, prompt_mask_b, pixels, false);
    float median = 1.0f;
    if (prompt_count <= 5) {
      median = MedianOfMaskedValues(gt_b, gt_mask_b, pixels, false);
    } else {
      median = MedianOfMaskedValues(prompt_b, prompt_mask_b, pixels, false);
      if (median <= 1.0e-2f) {
        median = MedianOfMaskedValues(gt_b, gt_mask_b, pixels, false);
      }
    }
    reference_host[b] = median;
  }

  CUDA_CHECK(cudaMemcpy(
      prepared.reference_meta,
      reference_host.data(),
      sizeof(float) * batch,
      cudaMemcpyHostToDevice));

  int64_t total = batch * pixels;
  int threads = 256;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  WarpPromptKernel<<<blocks, threads>>>(
      prompt_depth,
      prompt_mask,
      prepared.reference_meta,
      prepared.prompt_depth,
      prepared.prompt_mask,
      pixels,
      total);
  CUDA_CHECK(cudaGetLastError());

  return prepared;
}

void PostprocessDepthSensorCuda(
    const float* pred,
    const float* reference_meta,
    float* pred_depth,
    float* pred_disparity,
    int64_t batch,
    int64_t num_queries) {
  int64_t total = batch * num_queries;
  int threads = 256;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  PostprocessKernel<<<blocks, threads>>>(
      pred,
      reference_meta,
      pred_depth,
      pred_disparity,
      num_queries,
      total);
  CUDA_CHECK(cudaGetLastError());
}

void GatherQueryChunkCuda(
    const float* query_coord,
    float* query_chunk,
    int64_t batch,
    int64_t num_queries,
    int64_t start,
    int64_t chunk) {
  int64_t total = batch * chunk * 2;
  int threads = 256;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  GatherQueryChunkKernel<<<blocks, threads>>>(
      query_coord, query_chunk, num_queries, start, chunk, total);
  CUDA_CHECK(cudaGetLastError());
}

void ScatterPredChunkCuda(
    const float* pred_chunk,
    float* pred,
    int64_t batch,
    int64_t num_queries,
    int64_t start,
    int64_t chunk) {
  int64_t total = batch * chunk;
  int threads = 256;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  ScatterPredChunkKernel<<<blocks, threads>>>(
      pred_chunk, pred, num_queries, start, chunk, total);
  CUDA_CHECK(cudaGetLastError());
}

void MakeDenseQueryCoordCuda(
    float* query_coord,
    int64_t batch,
    int64_t height,
    int64_t width) {
  if (!query_coord || batch <= 0 || height <= 0 || width <= 0) {
    throw std::invalid_argument("Invalid input for MakeDenseQueryCoordCuda.");
  }
  int64_t pixels = height * width;
  int64_t total = batch * pixels;
  int threads = 256;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  MakeDenseQueryCoordKernel<<<blocks, threads>>>(query_coord, height, width, pixels, total);
  CUDA_CHECK(cudaGetLastError());
}

UniformCoordResult Make3DUniformCoordTriangleCuda(
    const float* depth,
    int height,
    int width,
    float fx,
    float fy,
    float cx,
    float cy,
    int64_t sample_count) {
  if (!depth || height <= 1 || width <= 1 || sample_count <= 0) {
    throw std::invalid_argument("Invalid input for Make3DUniformCoordTriangleCuda.");
  }

  int64_t num_faces = static_cast<int64_t>(height - 1) * (width - 1) * 2;
  FaceScore* d_scores = nullptr;
  CUDA_CHECK(cudaMalloc(&d_scores, static_cast<size_t>(num_faces) * sizeof(FaceScore)));

  int threads = 256;
  int blocks = static_cast<int>((num_faces + threads - 1) / threads);
  ScoreFacesKernel<<<blocks, threads>>>(depth, d_scores, height, width, fx, fy, cx, cy);
  CUDA_CHECK(cudaGetLastError());

  std::vector<FaceScore> scores(static_cast<size_t>(num_faces));
  CUDA_CHECK(cudaMemcpy(scores.data(), d_scores, scores.size() * sizeof(FaceScore), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_scores));

  std::vector<FaceScore> faces;
  faces.reserve(scores.size());
  for (const FaceScore& score : scores) {
    if (score.score > 0.0f) faces.push_back(score);
  }
  if (faces.empty()) {
    throw std::runtime_error("Make3DUniformCoordTriangleCuda: all faces were pruned.");
  }

  int64_t valid_faces = static_cast<int64_t>(faces.size());
  std::vector<float> coords_yx;
  coords_yx.reserve(static_cast<size_t>(sample_count) * 2);

  if (sample_count <= valid_faces) {
    double total_area = 0.0;
    for (const FaceScore& face : faces) {
      total_area += face.score;
    }
    std::vector<double> cdf(faces.size());
    double running = 0.0;
    for (size_t i = 0; i < faces.size(); ++i) {
      running += static_cast<double>(faces[i].score) / total_area;
      cdf[i] = running;
    }
    for (int64_t k = 0; k < sample_count; ++k) {
      double q = (static_cast<double>(k) + 0.5) / static_cast<double>(sample_count);
      auto it = std::lower_bound(cdf.begin(), cdf.end(), q);
      size_t face_idx = std::min(static_cast<size_t>(it - cdf.begin()), faces.size() - 1);
      AppendCoordFromFace(coords_yx, faces[face_idx], 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f, height, width);
    }
  } else {
    for (const FaceScore& face : faces) {
      AppendCoordFromFace(coords_yx, face, 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f, height, width);
    }

    int64_t remaining = sample_count - valid_faces;
    double sqrt_sum = 0.0;
    for (const FaceScore& face : faces) {
      sqrt_sum += std::sqrt(std::max(face.score, 0.0f));
    }
    std::vector<int64_t> extra_counts(faces.size(), 0);
    std::vector<std::pair<double, size_t>> fractions;
    fractions.reserve(faces.size());
    int64_t assigned = 0;
    for (size_t i = 0; i < faces.size(); ++i) {
      double raw = std::sqrt(std::max(faces[i].score, 0.0f)) / sqrt_sum * static_cast<double>(remaining);
      int64_t base = static_cast<int64_t>(std::floor(raw));
      extra_counts[i] = base;
      assigned += base;
      fractions.emplace_back(raw - static_cast<double>(base), i);
    }
    std::sort(fractions.begin(), fractions.end(), [](const auto& a, const auto& b) {
      return a.first > b.first;
    });
    for (int64_t i = 0; i < remaining - assigned; ++i) {
      extra_counts[fractions[static_cast<size_t>(i)].second] += 1;
    }

    const std::vector<float> bary = MakeBaryTemplate();
    size_t bary_count = bary.size() / 3;
    for (size_t face_idx = 0; face_idx < faces.size() &&
                                static_cast<int64_t>(coords_yx.size() / 2) < sample_count;
         ++face_idx) {
      int64_t count = extra_counts[face_idx];
      for (int64_t k = 0; k < count && static_cast<int64_t>(coords_yx.size() / 2) < sample_count; ++k) {
        size_t bi = static_cast<size_t>(k) % bary_count;
        AppendCoordFromFace(
            coords_yx,
            faces[face_idx],
            bary[bi * 3 + 0],
            bary[bi * 3 + 1],
            bary[bi * 3 + 2],
            height,
            width);
      }
    }
  }

  coords_yx.resize(static_cast<size_t>(sample_count) * 2);

  UniformCoordResult result;
  result.count = sample_count;
  CUDA_CHECK(cudaMalloc(&result.coords_yx, coords_yx.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(result.coords_yx, coords_yx.data(), coords_yx.size() * sizeof(float), cudaMemcpyHostToDevice));
  return result;
}

void FreeUniformCoordResult(UniformCoordResult* result) {
  if (!result) return;
  if (result->coords_yx) CUDA_CHECK(cudaFree(result->coords_yx));
  result->coords_yx = nullptr;
  result->count = 0;
}

void FreePreparedDepthSensorInputs(PreparedDepthSensorInputs* prepared) {
  if (!prepared) return;
  if (prepared->prompt_depth) CUDA_CHECK(cudaFree(prepared->prompt_depth));
  if (prepared->prompt_mask) CUDA_CHECK(cudaFree(prepared->prompt_mask));
  if (prepared->reference_meta) CUDA_CHECK(cudaFree(prepared->reference_meta));
  prepared->prompt_depth = nullptr;
  prepared->prompt_mask = nullptr;
  prepared->reference_meta = nullptr;
}

}  // namespace infinidepth
