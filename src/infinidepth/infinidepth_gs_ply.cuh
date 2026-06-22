#pragma once

#include "infinidepth_gs_sparse.cuh"

#include <filesystem>

namespace infinidepth {

void ExportSparseGaussiansPly(
    const std::filesystem::path& path,
    const SparseGaussiansCuda& gaussians,
    const float* pixel_intrinsics_host,
    const float* extrinsics_host,
    int height,
    int width);

}  // namespace infinidepth
