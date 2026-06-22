#include "infinidepth_gs_ply.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

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

std::vector<float> CopyDeviceF32(const float* device, size_t count) {
  std::vector<float> host(count);
  CUDA_CHECK(cudaMemcpy(host.data(), device, count * sizeof(float), cudaMemcpyDeviceToHost));
  return host;
}

float ClampFloat(float value, float lo, float hi) {
  return std::max(lo, std::min(value, hi));
}

void WriteF32(std::ofstream& file, float value) {
  file.write(reinterpret_cast<const char*>(&value), sizeof(float));
}

void WriteU32(std::ofstream& file, uint32_t value) {
  file.write(reinterpret_cast<const char*>(&value), sizeof(uint32_t));
}

void WriteU8(std::ofstream& file, uint8_t value) {
  file.write(reinterpret_cast<const char*>(&value), sizeof(uint8_t));
}

}  // namespace

void ExportSparseGaussiansPly(
    const std::filesystem::path& path,
    const SparseGaussiansCuda& gaussians,
    const float* pixel_intrinsics_host,
    const float* extrinsics_host,
    int height,
    int width) {
  if (gaussians.count <= 0) {
    throw std::runtime_error("Cannot export empty sparse gaussians.");
  }
  size_t count = static_cast<size_t>(gaussians.count);

  std::vector<float> means = CopyDeviceF32(gaussians.means, count * 3);
  std::vector<float> harmonics = CopyDeviceF32(gaussians.harmonics, count * 27);
  std::vector<float> opacities = CopyDeviceF32(gaussians.opacities, count);
  std::vector<float> scales = CopyDeviceF32(gaussians.scales, count * 3);
  std::vector<float> rotations = CopyDeviceF32(gaussians.rotations, count * 4);

  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open PLY path: " + path.string());
  }

  file << "ply\n";
  file << "format binary_little_endian 1.0\n";
  file << "element vertex " << count << "\n";
  file << "property float x\n";
  file << "property float y\n";
  file << "property float z\n";
  file << "property float nx\n";
  file << "property float ny\n";
  file << "property float nz\n";
  file << "property float f_dc_0\n";
  file << "property float f_dc_1\n";
  file << "property float f_dc_2\n";
  file << "property float opacity\n";
  file << "property float scale_0\n";
  file << "property float scale_1\n";
  file << "property float scale_2\n";
  file << "property float rot_0\n";
  file << "property float rot_1\n";
  file << "property float rot_2\n";
  file << "property float rot_3\n";
  file << "element image_size 2\n";
  file << "property uint image_size\n";
  file << "element intrinsic 9\n";
  file << "property float intrinsic\n";
  file << "element extrinsic 16\n";
  file << "property float extrinsic\n";
  file << "element color_space 1\n";
  file << "property uchar color_space\n";
  file << "end_header\n";

  for (size_t i = 0; i < count; ++i) {
    WriteF32(file, means[i * 3 + 0]);
    WriteF32(file, means[i * 3 + 1]);
    WriteF32(file, means[i * 3 + 2]);
    WriteF32(file, 0.0f);
    WriteF32(file, 0.0f);
    WriteF32(file, 0.0f);
    WriteF32(file, harmonics[i * 27 + 0]);
    WriteF32(file, harmonics[i * 27 + 9]);
    WriteF32(file, harmonics[i * 27 + 18]);

    float opacity = ClampFloat(opacities[i], 1.0e-6f, 1.0f - 1.0e-6f);
    WriteF32(file, std::log(opacity / (1.0f - opacity)));
    WriteF32(file, std::log(std::max(scales[i * 3 + 0], 1.0e-10f)));
    WriteF32(file, std::log(std::max(scales[i * 3 + 1], 1.0e-10f)));
    WriteF32(file, std::log(std::max(scales[i * 3 + 2], 1.0e-10f)));
    WriteF32(file, rotations[i * 4 + 0]);
    WriteF32(file, rotations[i * 4 + 1]);
    WriteF32(file, rotations[i * 4 + 2]);
    WriteF32(file, rotations[i * 4 + 3]);
  }

  WriteU32(file, static_cast<uint32_t>(width));
  WriteU32(file, static_cast<uint32_t>(height));
  for (int i = 0; i < 9; ++i) {
    WriteF32(file, pixel_intrinsics_host[i]);
  }
  for (int i = 0; i < 16; ++i) {
    WriteF32(file, extrinsics_host[i]);
  }
  WriteU8(file, 1);
}

}  // namespace infinidepth
