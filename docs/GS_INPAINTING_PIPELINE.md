# Framebuffer Inpainting + GS Prediction Pipeline

This document describes the Splatshop framebuffer dump path that:

1. dumps the current framebuffer color, transparency mask, depth, and camera JSON;
2. inpaints the transparent/missing regions with LaMa ONNX;
3. runs the InfiniDepth DepthSensor encoder/decoder ONNX models;
4. runs the GS predictor ONNX model;
5. writes a sparse Gaussian `.ply`;
6. optionally loads that predicted `.ply` back into the canvas.

The implementation lives mostly in:

- `src/SplatEditor_render.h`
- `src/inpaint/LamaInpaint.cpp`
- `src/infinidepth/InfiniDepthGsPredictor.cpp`
- `src/infinidepth/infinidepth_*`
- `src/RuntimeDllSearchPath.cpp`

Do not run the ONNX/CUDA path with a random global `PATH`. ONNX Runtime GPU,
CUDA, and cuDNN/PyTorch DLLs must resolve to matching versions.

## Runtime DLL Setup

Known working dependency roots:

```powershell
$ortRoot = "E:\libs\onnxruntime-win-x64-gpu-1.23.2"
$cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
$torchLib = "D:\miniconda\envs\moge\Lib\site-packages\torch\lib"
$opencvBin = "E:\libs\opencv\build\x64\vc16\bin"
```

Set the runtime DLL directories before launching Splatshop:

```powershell
$env:ONNXRUNTIME_ROOT = $ortRoot
$env:CUDA_ROOT = $cudaRoot
$env:TORCH_LIB = $torchLib
$env:OPENCV_RUNTIME_DIR = $opencvBin

$env:INFINIDEPTH_ONNXRUNTIME_DLL_DIR = "$ortRoot\lib"
$env:INFINIDEPTH_CUDA_DLL_DIR = "$cudaRoot\bin"
$env:INFINIDEPTH_TORCH_DLL_DIR = $torchLib

$env:PATH = "$opencvBin;$ortRoot\lib;$cudaRoot\bin;$torchLib;$env:PATH"
```

At process startup Splatshop also calls `configureRuntimeDllSearchPath()`, which
registers those directories with Windows and prepends them to the process `PATH`.
This happens before `Ort::InitApi()` and before any LaMa or InfiniDepth ONNX
session is created.

The executable delay-loads `onnxruntime.dll`. This is intentional: it prevents
Windows from loading the wrong ONNX Runtime DLL before the process has installed
the intended DLL search path.

## Model Environment Variables

LaMa inpainting:

```powershell
$env:SPLATSHOP_LAMA_ONNX_MODEL = "E:\projects\sd_models\onnx\carve_lama_fp32_1024_cuda_export.onnx"
$env:SPLATSHOP_LAMA_MODE = "full"
```

InfiniDepth + GS prediction:

```powershell
$env:SPLATSHOP_INFINIDEPTH_ENCODER = "E:\projects\InfiniDepth\workspace\infinidepth_depthsensor_encoder.onnx"
$env:SPLATSHOP_INFINIDEPTH_DECODER = "E:\projects\InfiniDepth\workspace\infinidepth_depthsensor_decoder.onnx"
$env:SPLATSHOP_INFINIDEPTH_GS_MODEL = "E:\projects\MoGe\workspace\gs_predictor_dynamic.onnx"
```

Runtime size and batching:

```powershell
$env:SPLATSHOP_INFINIDEPTH_CUDA_DEVICE = "0"
$env:SPLATSHOP_INFINIDEPTH_INPUT_HEIGHT = "768"
$env:SPLATSHOP_INFINIDEPTH_INPUT_WIDTH = "1024"
$env:SPLATSHOP_INFINIDEPTH_PROMPT_SAMPLES = "1500"
$env:SPLATSHOP_INFINIDEPTH_CHUNK_SIZE = "100000"
$env:SPLATSHOP_INFINIDEPTH_SAMPLE_POINT_NUM = "2000000"
```

Optional toggles:

```powershell
$env:SPLATSHOP_INFINIDEPTH_ENABLE_OPTIMIZATIONS = "0"
$env:SPLATSHOP_INFINIDEPTH_ENABLE_DEPTH_NOISE_FILTER = "0"
$env:SPLATSHOP_INFINIDEPTH_CACHE_SESSIONS = "1"
$env:SPLATSHOP_INFINIDEPTH_AUTO_LOAD_PLY = "1"
```

ONNX Runtime CUDA provider settings:

```powershell
$env:SPLATSHOP_INFINIDEPTH_CUDNN_CONV_ALGO_SEARCH = "HEURISTIC"
$env:SPLATSHOP_INFINIDEPTH_CUDNN_CONV_USE_MAX_WORKSPACE = "0"
$env:SPLATSHOP_INFINIDEPTH_CUDA_USE_TF32 = "1"
```

These settings are part of the InfiniDepth session cache key. If they change
while the app is running, the next prediction creates fresh ONNX sessions.

`SPLATSHOP_INFINIDEPTH_CACHE_SESSIONS` controls ONNX session lifetime:

- `1`: keep encoder, decoder, and GS sessions cached after the first run. Faster repeat dumps, higher persistent GPU memory use.
- `0`: load sessions for each prediction and release them immediately. Slower repeat dumps, lower memory pressure for multi-view or constrained GPUs.

`SPLATSHOP_INFINIDEPTH_AUTO_LOAD_PLY` controls whether a successful predicted
PLY is automatically loaded into the current canvas:

- `1`: auto-load and select the generated layer.
- `0`: only write the `.ply` file.

## Recommended Launch Script

```powershell
$ortRoot = "E:\libs\onnxruntime-win-x64-gpu-1.23.2"
$cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
$torchLib = "D:\miniconda\envs\moge\Lib\site-packages\torch\lib"
$opencvBin = "E:\libs\opencv\build\x64\vc16\bin"

$env:ONNXRUNTIME_ROOT = $ortRoot
$env:CUDA_ROOT = $cudaRoot
$env:TORCH_LIB = $torchLib
$env:OPENCV_RUNTIME_DIR = $opencvBin
$env:INFINIDEPTH_ONNXRUNTIME_DLL_DIR = "$ortRoot\lib"
$env:INFINIDEPTH_CUDA_DLL_DIR = "$cudaRoot\bin"
$env:INFINIDEPTH_TORCH_DLL_DIR = $torchLib
$env:PATH = "$opencvBin;$ortRoot\lib;$cudaRoot\bin;$torchLib;$env:PATH"

$env:SPLATSHOP_LAMA_ONNX_MODEL = "E:\projects\sd_models\onnx\carve_lama_fp32_1024_cuda_export.onnx"
$env:SPLATSHOP_LAMA_MODE = "full"

$env:SPLATSHOP_INFINIDEPTH_ENCODER = "E:\projects\InfiniDepth\workspace\infinidepth_depthsensor_encoder.onnx"
$env:SPLATSHOP_INFINIDEPTH_DECODER = "E:\projects\InfiniDepth\workspace\infinidepth_depthsensor_decoder.onnx"
$env:SPLATSHOP_INFINIDEPTH_GS_MODEL = "E:\projects\MoGe\workspace\gs_predictor_dynamic.onnx"
$env:SPLATSHOP_INFINIDEPTH_CUDA_DEVICE = "0"
$env:SPLATSHOP_INFINIDEPTH_INPUT_HEIGHT = "768"
$env:SPLATSHOP_INFINIDEPTH_INPUT_WIDTH = "1024"
$env:SPLATSHOP_INFINIDEPTH_PROMPT_SAMPLES = "1500"
$env:SPLATSHOP_INFINIDEPTH_CHUNK_SIZE = "100000"
$env:SPLATSHOP_INFINIDEPTH_SAMPLE_POINT_NUM = "1000000"
$env:SPLATSHOP_INFINIDEPTH_CACHE_SESSIONS = "0"
$env:SPLATSHOP_INFINIDEPTH_AUTO_LOAD_PLY = "1"
$env:SPLATSHOP_INFINIDEPTH_CUDNN_CONV_ALGO_SEARCH = "HEURISTIC"
$env:SPLATSHOP_INFINIDEPTH_CUDNN_CONV_USE_MAX_WORKSPACE = "0"
$env:SPLATSHOP_INFINIDEPTH_CUDA_USE_TF32 = "1"

.\build-vs2022-x64\Release\SplatEditor.exe
```

For speed on a GPU with enough memory, change:

```powershell
$env:SPLATSHOP_INFINIDEPTH_CACHE_SESSIONS = "1"
$env:SPLATSHOP_INFINIDEPTH_SAMPLE_POINT_NUM = "2000000"
```

## Build Dependencies

Required:

- Visual Studio 2022 C++ toolchain
- CUDA Toolkit 12.4
- ONNX Runtime GPU 1.23.2 C/C++ package
- OpenCV, including the runtime DLL directory containing `opencv_world4120.dll`
- PyTorch environment containing cuDNN DLLs, especially `cudnn64_9.dll`

The CMake target links:

- ONNX Runtime import library
- CUDA driver/runtime libraries through the existing project CUDA setup
- `windowscodecs`
- `ole32`
- `delayimp`

The InfiniDepth CUDA kernels are built by explicit `nvcc -c` custom commands.
This avoids requiring the selected Visual Studio generator instance to have CUDA
MSBuild integration installed.

The executable has a normal startup dependency on `opencv_world4120.dll`. Unlike
`onnxruntime.dll`, it is not delay-loaded, so Windows must be able to find it
before `main()` runs. Put `OPENCV_RUNTIME_DIR` at the front of `PATH` before
launching the app.

## Configure and Build

```powershell
cmake -S . -B build-vs2022-x64
cmake --build build-vs2022-x64 --config Release --target SplatEditor -- /v:minimal /clp:ErrorsOnly
```

The built executable is:

```text
build-vs2022-x64\Release\SplatEditor.exe
```

## Framebuffer Dump Outputs

When the toolbar requests a framebuffer dump, Splatshop writes files under
`debug/`:

- `framebuffer_XXXX_color.png`
- `framebuffer_XXXX_transparent_mask.png`
- `framebuffer_XXXX_transparent_mask.pfm`
- `framebuffer_XXXX_depth.pfm`
- `framebuffer_XXXX_camera.json`
- `framebuffer_XXXX_inpainted.png`
- `framebuffer_XXXX_infinidepth_gs.ply`
- `framebuffer_XXXX_info.txt`

The depth `.pfm` is consumed directly by the integrated InfiniDepth input
processor. No Python `.pfm -> .npy` conversion is needed.

The camera JSON contains pixel intrinsics and the camera-to-world matrix. The
InfiniDepth input processor reads this JSON and applies the OpenGL-to-OpenCV
camera basis conversion expected by the GS predictor path.

## Automatic PLY Loading

After a successful InfiniDepth GS prediction, Splatshop can load the generated
PLY back into the scene automatically. This uses the same existing APIs as
drag/drop PLY loading:

- `GSPlyLoader::load(path)`
- `make_shared<SNSplats>(...)`
- `scene.world->children.push_back(...)`
- `setSelectedNode(...)`

The actual GPU upload still happens through the normal scene update path via
`SplatEditor::uploadSplats()`.

Disable this behavior with:

```powershell
$env:SPLATSHOP_INFINIDEPTH_AUTO_LOAD_PLY = "0"
```

## Common Failures

### Exit code `-1073741515`

This is `0xC0000135`, meaning Windows could not find a DLL before the app
started. Check the startup dependencies with:

```powershell
dumpbin /dependents .\build-vs2022-x64\Release\SplatEditor.exe
```

For this build, `opencv_world4120.dll` is a startup dependency. Make sure the
OpenCV runtime folder is first in `PATH`:

```powershell
$opencvBin = "E:\libs\opencv\build\x64\vc16\bin"
$env:PATH = "$opencvBin;$env:PATH"
.\build-vs2022-x64\Release\SplatEditor.exe
```

### Exit code `-1073741819`

This is `0xC0000005`, an access violation. In this pipeline it has usually meant
ONNX Runtime or CUDA provider DLLs were loaded in the wrong order or from a
wrong directory. Confirm:

```powershell
where.exe onnxruntime.dll
where.exe cudnn64_9.dll
```

The first `onnxruntime.dll` should come from:

```text
E:\libs\onnxruntime-win-x64-gpu-1.23.2\lib
```

`cudnn64_9.dll` should come from:

```text
D:\miniconda\envs\moge\Lib\site-packages\torch\lib
```

Also confirm the executable prints runtime DLL directory setup lines on startup.

### Missing `cudnn64_9.dll`

Set:

```powershell
$env:INFINIDEPTH_TORCH_DLL_DIR = "D:\miniconda\envs\moge\Lib\site-packages\torch\lib"
$env:PATH = "$env:INFINIDEPTH_TORCH_DLL_DIR;$env:PATH"
```

### CUDA out of memory

Use on-the-fly sessions and fewer samples:

```powershell
$env:SPLATSHOP_INFINIDEPTH_CACHE_SESSIONS = "0"
$env:SPLATSHOP_INFINIDEPTH_SAMPLE_POINT_NUM = "1000000"
```

Lower further if needed:

```powershell
$env:SPLATSHOP_INFINIDEPTH_SAMPLE_POINT_NUM = "500000"
```

### ONNX Runtime CUDA warnings

Warnings about shape nodes assigned to CPU or inserted `Memcpy` nodes can appear
during successful runs. Treat them as informational unless the process fails.

### cuDNN Conv Backend Failure

If GS prediction fails with a message like:

```text
CUDNN_FE failure 11: CUDNN_BACKEND_API_FAILED
Name:'/gaussian_head/gaussian_head.0/Conv'
```

the failure is inside ONNX Runtime's CUDA provider while running the GS ONNX
model, before PLY writing or canvas loading. Use the safer CUDA provider
settings:

```powershell
$env:SPLATSHOP_INFINIDEPTH_CUDNN_CONV_ALGO_SEARCH = "HEURISTIC"
$env:SPLATSHOP_INFINIDEPTH_CUDNN_CONV_USE_MAX_WORKSPACE = "0"
$env:SPLATSHOP_INFINIDEPTH_CACHE_SESSIONS = "0"
```

If the same error persists, try disabling TF32 and reducing the dense input
resolution:

```powershell
$env:SPLATSHOP_INFINIDEPTH_CUDA_USE_TF32 = "0"
$env:SPLATSHOP_INFINIDEPTH_INPUT_HEIGHT = "576"
$env:SPLATSHOP_INFINIDEPTH_INPUT_WIDTH = "768"
```

### Wrong ONNX Runtime DLL

The target delay-loads `onnxruntime.dll` and calls `Ort::InitApi()` after runtime
DLL directories are configured. If a wrong DLL still loads, check copied DLLs in
the executable directory and the leading entries in `PATH`.
