#include "infinidepth_input_processing.h"
#include "infinidepth_input_processing_kernels.cuh"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <wincodec.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <regex>
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

struct ImageRgb {
  std::vector<uint8_t> rgb;
  int width = 0;
  int height = 0;
};

struct CameraJson {
  std::vector<float> c2w;
  float fx = 0.0f;
  float fy = 0.0f;
  float cx = 0.0f;
  float cy = 0.0f;
  int width = 0;
  int height = 0;
};

std::wstring WidenUtf8(const std::string& text) {
  if (text.empty()) return {};
  int size = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, nullptr, 0);
  if (size <= 0) throw std::runtime_error("Failed to convert path to UTF-16.");
  std::wstring wide(static_cast<size_t>(size), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, wide.data(), size);
  if (!wide.empty() && wide.back() == L'\0') wide.pop_back();
  return wide;
}

uint16_t ReadU16(std::ifstream& in) {
  uint8_t b[2];
  in.read(reinterpret_cast<char*>(b), 2);
  return static_cast<uint16_t>(b[0] | (b[1] << 8));
}

uint32_t ReadU32(std::ifstream& in) {
  uint8_t b[4];
  in.read(reinterpret_cast<char*>(b), 4);
  return static_cast<uint32_t>(b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24));
}

std::vector<int64_t> ParseNpyShape(const std::string& header) {
  size_t l = header.find('(');
  size_t r = header.find(')', l);
  if (l == std::string::npos || r == std::string::npos) throw std::runtime_error("Invalid npy shape header.");
  std::string inside = header.substr(l + 1, r - l - 1);
  std::vector<int64_t> shape;
  size_t pos = 0;
  while (pos < inside.size()) {
    while (pos < inside.size() && (inside[pos] == ' ' || inside[pos] == ',')) ++pos;
    if (pos >= inside.size()) break;
    size_t end = pos;
    while (end < inside.size() && ((inside[end] >= '0' && inside[end] <= '9') || inside[end] == '-')) ++end;
    if (end > pos) shape.push_back(std::stoll(inside.substr(pos, end - pos)));
    pos = end + 1;
  }
  return shape;
}

std::vector<float> LoadNpyDepthF32(const std::string& path, int* height, int* width) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Failed to open depth npy: " + path);
  char magic[6];
  in.read(magic, 6);
  if (std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error("Not an npy file: " + path);
  uint8_t major = 0, minor = 0;
  in.read(reinterpret_cast<char*>(&major), 1);
  in.read(reinterpret_cast<char*>(&minor), 1);
  (void)minor;
  uint32_t header_len = major == 1 ? ReadU16(in) : ReadU32(in);
  std::string header(header_len, '\0');
  in.read(header.data(), header_len);
  if (header.find("'fortran_order': True") != std::string::npos) {
    throw std::runtime_error("Fortran-order npy depth is not supported: " + path);
  }
  if (header.find("'<f4'") == std::string::npos && header.find("\"<f4\"") == std::string::npos &&
      header.find("'<f8'") == std::string::npos && header.find("\"<f8\"") == std::string::npos) {
    throw std::runtime_error("Only float32/float64 npy depth is supported: " + path);
  }
  bool f64 = header.find("<f8") != std::string::npos;
  std::vector<int64_t> shape = ParseNpyShape(header);
  while (shape.size() > 2 && shape.front() == 1) shape.erase(shape.begin());
  if (shape.size() != 2) throw std::runtime_error("Depth npy must resolve to shape [H, W].");
  *height = static_cast<int>(shape[0]);
  *width = static_cast<int>(shape[1]);
  size_t count = static_cast<size_t>(*height) * *width;
  std::vector<float> depth(count);
  if (f64) {
    std::vector<double> tmp(count);
    in.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(tmp.size() * sizeof(double)));
    for (size_t i = 0; i < count; ++i) depth[i] = static_cast<float>(tmp[i]);
  } else {
    in.read(reinterpret_cast<char*>(depth.data()), static_cast<std::streamsize>(depth.size() * sizeof(float)));
  }
  if (!in) throw std::runtime_error("Failed to read npy depth payload: " + path);
  return depth;
}

std::string ReadAsciiToken(std::ifstream& in) {
  std::string token;
  while (in >> token) {
    if (!token.empty() && token[0] == '#') {
      std::string ignored;
      std::getline(in, ignored);
      continue;
    }
    return token;
  }
  throw std::runtime_error("Unexpected end of PFM header.");
}

std::vector<float> LoadPfmDepthF32(const std::string& path, int* height, int* width) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Failed to open depth pfm: " + path);

  std::string header = ReadAsciiToken(in);
  if (header != "Pf") {
    throw std::runtime_error("Only single-channel Pf depth files are supported: " + path);
  }

  *width = std::stoi(ReadAsciiToken(in));
  *height = std::stoi(ReadAsciiToken(in));
  float scale = std::stof(ReadAsciiToken(in));
  if (*width <= 0 || *height <= 0) {
    throw std::runtime_error("Invalid PFM dimensions: " + path);
  }

  in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  if (!in) throw std::runtime_error("Invalid PFM payload start: " + path);

  size_t count = static_cast<size_t>(*width) * static_cast<size_t>(*height);
  std::vector<float> depth(count);
  in.read(reinterpret_cast<char*>(depth.data()), static_cast<std::streamsize>(depth.size() * sizeof(float)));
  if (!in) throw std::runtime_error("Failed to read PFM depth payload: " + path);

  if (scale > 0.0f) {
    for (float& value : depth) {
      uint8_t* b = reinterpret_cast<uint8_t*>(&value);
      std::swap(b[0], b[3]);
      std::swap(b[1], b[2]);
    }
  }
  return depth;
}

std::vector<float> LoadDepthF32(const std::string& path, int* height, int* width) {
  std::string lower = path;
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });

  if (lower.size() >= 4 && lower.substr(lower.size() - 4) == ".pfm") {
    return LoadPfmDepthF32(path, height, width);
  }
  return LoadNpyDepthF32(path, height, width);
}

ImageRgb LoadImageWic(const std::string& path) {
  HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
  bool com_initialized = SUCCEEDED(hr);
  if (hr == RPC_E_CHANGED_MODE) com_initialized = false;
  else if (FAILED(hr)) throw std::runtime_error("CoInitializeEx failed.");

  IWICImagingFactory* factory = nullptr;
  IWICBitmapDecoder* decoder = nullptr;
  IWICBitmapFrameDecode* frame = nullptr;
  IWICFormatConverter* converter = nullptr;
  ImageRgb image;
  try {
    hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory));
    if (FAILED(hr)) throw std::runtime_error("Failed to create WIC factory.");
    std::wstring wide = WidenUtf8(path);
    hr = factory->CreateDecoderFromFilename(wide.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnLoad, &decoder);
    if (FAILED(hr)) throw std::runtime_error("Failed to decode image: " + path);
    hr = decoder->GetFrame(0, &frame);
    if (FAILED(hr)) throw std::runtime_error("Failed to get image frame: " + path);
    hr = factory->CreateFormatConverter(&converter);
    if (FAILED(hr)) throw std::runtime_error("Failed to create WIC converter.");
    hr = converter->Initialize(frame, GUID_WICPixelFormat24bppRGB, WICBitmapDitherTypeNone, nullptr, 0.0, WICBitmapPaletteTypeCustom);
    if (FAILED(hr)) throw std::runtime_error("Failed to convert image to RGB.");
    UINT w = 0, h = 0;
    converter->GetSize(&w, &h);
    image.width = static_cast<int>(w);
    image.height = static_cast<int>(h);
    image.rgb.resize(static_cast<size_t>(w) * h * 3);
    hr = converter->CopyPixels(nullptr, w * 3, static_cast<UINT>(image.rgb.size()), image.rgb.data());
    if (FAILED(hr)) throw std::runtime_error("Failed to copy image pixels.");
  } catch (...) {
    if (converter) converter->Release();
    if (frame) frame->Release();
    if (decoder) decoder->Release();
    if (factory) factory->Release();
    if (com_initialized) CoUninitialize();
    throw;
  }
  converter->Release();
  frame->Release();
  decoder->Release();
  factory->Release();
  if (com_initialized) CoUninitialize();
  return image;
}

std::string ReadTextFile(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Failed to open JSON file: " + path);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::vector<float> ExtractFloatsAfterKey(const std::string& text, const std::string& key, int expected) {
  size_t key_pos = text.find("\"" + key + "\"");
  if (key_pos == std::string::npos) return {};
  size_t start = text.find('[', key_pos);
  if (start == std::string::npos) return {};
  std::regex number_re(R"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)");
  std::vector<float> values;
  auto begin = std::sregex_iterator(text.begin() + static_cast<std::ptrdiff_t>(start), text.end(), number_re);
  auto end = std::sregex_iterator();
  for (auto it = begin; it != end && static_cast<int>(values.size()) < expected; ++it) {
    values.push_back(std::stof(it->str()));
  }
  if (static_cast<int>(values.size()) != expected) {
    throw std::runtime_error("Failed to parse " + key + " from camera JSON.");
  }
  return values;
}

float ExtractFloatAfterKey(const std::string& text, const std::string& key, bool required = true) {
  size_t key_pos = text.find("\"" + key + "\"");
  if (key_pos == std::string::npos) {
    if (required) throw std::runtime_error("Missing " + key + " in camera JSON.");
    return 0.0f;
  }
  size_t colon = text.find(':', key_pos);
  if (colon == std::string::npos) {
    throw std::runtime_error("Invalid scalar field " + key + " in camera JSON.");
  }
  std::regex number_re(R"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)");
  std::smatch match;
  std::string tail = text.substr(colon + 1);
  if (!std::regex_search(tail, match, number_re)) {
    throw std::runtime_error("Failed to parse scalar field " + key + " from camera JSON.");
  }
  return std::stof(match.str());
}

std::vector<float> MatMul4(const std::vector<float>& a, const std::vector<float>& b) {
  std::vector<float> out(16, 0.0f);
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      for (int k = 0; k < 4; ++k) out[r * 4 + c] += a[r * 4 + k] * b[k * 4 + c];
    }
  }
  return out;
}

std::vector<float> InvertRigid4(const std::vector<float>& m) {
  std::vector<float> inv(16, 0.0f);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) inv[r * 4 + c] = m[c * 4 + r];
  }
  for (int r = 0; r < 3; ++r) {
    inv[r * 4 + 3] = -(inv[r * 4 + 0] * m[3] + inv[r * 4 + 1] * m[7] + inv[r * 4 + 2] * m[11]);
  }
  inv[15] = 1.0f;
  return inv;
}

CameraJson LoadCameraJson(const std::string& path) {
  if (path.empty()) {
    throw std::invalid_argument("Raw input processing requires camera_json_path for intrinsics and extrinsics.");
  }
  std::string text = ReadTextFile(path);
  CameraJson camera;
  camera.fx = ExtractFloatAfterKey(text, "fxPixels");
  camera.fy = ExtractFloatAfterKey(text, "fyPixels");
  camera.cx = ExtractFloatAfterKey(text, "cxPixels");
  camera.cy = ExtractFloatAfterKey(text, "cyPixels");
  camera.width = static_cast<int>(std::lround(ExtractFloatAfterKey(text, "width")));
  camera.height = static_cast<int>(std::lround(ExtractFloatAfterKey(text, "height")));
  if (camera.fx <= 0.0f || camera.fy <= 0.0f || camera.width <= 0 || camera.height <= 0) {
    throw std::runtime_error("Camera JSON contains invalid intrinsics or framebuffer size.");
  }

  std::vector<float> c2w = ExtractFloatsAfterKey(text, "cameraWorldMatrix", 16);
  if (c2w.empty()) {
    std::vector<float> view = ExtractFloatsAfterKey(text, "viewMatrix", 16);
    c2w = InvertRigid4(view);
  }
  std::string lower = text;
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  if (lower.find("opengl") != std::string::npos) {
    std::vector<float> opencv_to_opengl = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, -1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, -1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    c2w = MatMul4(c2w, opencv_to_opengl);
  }
  camera.c2w = c2w;
  return camera;
}

}  // namespace

ProcessedGsInputsCuda ProcessRawGsInputsCuda(const InputProcessingOptions& options) {
  if (options.image_path.empty() || options.depth_path.empty()) {
    throw std::invalid_argument("Raw input processing requires image_path and depth_path.");
  }
  if (options.target_height <= 0 || options.target_width <= 0) {
    throw std::invalid_argument("Invalid target input size.");
  }

  ImageRgb image = LoadImageWic(options.image_path);
  CameraJson camera = LoadCameraJson(options.camera_json_path);
  int depth_h = 0;
  int depth_w = 0;
  std::vector<float> raw_depth = LoadDepthF32(options.depth_path, &depth_h, &depth_w);

  ProcessedGsInputsCuda out;
  out.batch = 1;
  out.height = options.target_height;
  out.width = options.target_width;
  out.original_height = image.height;
  out.original_width = image.width;

  const size_t image_bytes = image.rgb.size() * sizeof(uint8_t);
  const size_t raw_depth_bytes = raw_depth.size() * sizeof(float);
  const size_t pixels = static_cast<size_t>(options.target_height) * options.target_width;

  uint8_t* d_raw_image = nullptr;
  float* d_raw_depth = nullptr;
  float* d_resized_depth = nullptr;
  CUDA_CHECK(cudaMalloc(&d_raw_image, image_bytes));
  CUDA_CHECK(cudaMemcpy(d_raw_image, image.rgb.data(), image_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&d_raw_depth, raw_depth_bytes));
  CUDA_CHECK(cudaMemcpy(d_raw_depth, raw_depth.data(), raw_depth_bytes, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&out.image, sizeof(float) * pixels * 3));
  CUDA_CHECK(cudaMalloc(&d_resized_depth, sizeof(float) * pixels));
  CUDA_CHECK(cudaMalloc(&out.gt, sizeof(float) * pixels));
  CUDA_CHECK(cudaMalloc(&out.gt_mask, sizeof(uint8_t) * pixels));
  CUDA_CHECK(cudaMalloc(&out.prompt, sizeof(float) * pixels));
  CUDA_CHECK(cudaMalloc(&out.prompt_mask, sizeof(uint8_t) * pixels));

  ResizeImageToNchwCuda(
      d_raw_image,
      image.height,
      image.width,
      out.image,
      options.target_height,
      options.target_width);
  ResizeDepthNearestCuda(
      d_raw_depth,
      depth_h,
      depth_w,
      d_resized_depth,
      options.target_height,
      options.target_width);

  InputProcessingCudaOptions cuda_options;
  cuda_options.prompt_samples = options.prompt_samples;
  cuda_options.min_prompt = options.min_prompt;
  cuda_options.max_prompt = options.max_prompt;
  cuda_options.enable_depth_noise_filter = options.enable_depth_noise_filter;
  cuda_options.filter_std_threshold = options.filter_std_threshold;
  cuda_options.filter_median_threshold = options.filter_median_threshold;
  cuda_options.filter_gradient_threshold = options.filter_gradient_threshold;
  cuda_options.filter_min_neighbors = options.filter_min_neighbors;
  if (cuda_options.enable_depth_noise_filter) {
    FilterDepthNoiseCuda(d_resized_depth, options.target_height, options.target_width, cuda_options);
  }
  BuildDepthAndPromptCuda(
      d_resized_depth,
      options.target_height,
      options.target_width,
      cuda_options,
      out.gt,
      out.gt_mask,
      out.prompt,
      out.prompt_mask);

  float sx = static_cast<float>(options.target_width) / static_cast<float>(camera.width);
  float sy = static_cast<float>(options.target_height) / static_cast<float>(camera.height);
  std::vector<float> intrinsics = {
      camera.fx * sx, 0.0f, camera.cx * sx,
      0.0f, camera.fy * sy, camera.cy * sy,
      0.0f, 0.0f, 1.0f,
  };
  CUDA_CHECK(cudaMalloc(&out.intrinsics, sizeof(float) * intrinsics.size()));
  CUDA_CHECK(cudaMemcpy(out.intrinsics, intrinsics.data(), sizeof(float) * intrinsics.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&out.extrinsics, sizeof(float) * camera.c2w.size()));
  CUDA_CHECK(cudaMemcpy(out.extrinsics, camera.c2w.data(), sizeof(float) * camera.c2w.size(), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaFree(d_raw_image));
  CUDA_CHECK(cudaFree(d_raw_depth));
  CUDA_CHECK(cudaFree(d_resized_depth));
  return out;
}

void FreeProcessedGsInputsCuda(ProcessedGsInputsCuda* inputs) {
  if (!inputs) return;
  if (inputs->image) cudaFree(inputs->image);
  if (inputs->intrinsics) cudaFree(inputs->intrinsics);
  if (inputs->extrinsics) cudaFree(inputs->extrinsics);
  if (inputs->gt) cudaFree(inputs->gt);
  if (inputs->gt_mask) cudaFree(inputs->gt_mask);
  if (inputs->prompt) cudaFree(inputs->prompt);
  if (inputs->prompt_mask) cudaFree(inputs->prompt_mask);
  inputs->image = nullptr;
  inputs->intrinsics = nullptr;
  inputs->extrinsics = nullptr;
  inputs->gt = nullptr;
  inputs->gt_mask = nullptr;
  inputs->prompt = nullptr;
  inputs->prompt_mask = nullptr;
}

}  // namespace infinidepth
