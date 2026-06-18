
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <cstdlib>

#include "ImageLoader.h"
#include "inpaint/LamaInpaint.h"
// Temporarily disabled while maintaining only the Lama inpaint module.
// #include "moge/MogeGsPredictor.h"
#include "moge_gs_export.h"
#include "json/json.hpp"

static nlohmann::json dump_render_target_matrix_json(const glm::mat4& matrix){
	nlohmann::json rows = nlohmann::json::array();
	for(int row = 0; row < 4; row++){
		nlohmann::json values = nlohmann::json::array();
		for(int col = 0; col < 4; col++){
			values.push_back(matrix[col][row]);
		}
		rows.push_back(values);
	}
	return rows;
}

static void dump_render_target_framebuffer(RenderTarget& target, string outputDirectory = "./debug"){

	int width = target.width;
	int height = target.height;
	int64_t numPixels = int64_t(width) * int64_t(height);

	if(width <= 0 || height <= 0 || target.framebuffer == nullptr){
		println("Framebuffer dump skipped: invalid target.");
		return;
	}

	std::filesystem::create_directories(outputDirectory);

	vector<uint64_t> pixels(numPixels);
	vector<uint8_t> rgba(numPixels * 4);
	vector<uint8_t> opacity(numPixels);
	vector<float> depths(numPixels);
	vector<float> opacities(numPixels);

	CURuntime::check(cuCtxSynchronize());
	CURuntime::check(cuMemcpyDtoH(
		pixels.data(),
		(CUdeviceptr)target.framebuffer,
		numPixels * sizeof(uint64_t)
	));

	float minDepth = Infinity;
	float maxDepth = -Infinity;
	float minOpacity = Infinity;
	float maxOpacity = -Infinity;
	uint64_t finiteDepthCount = 0;

	for(int y = 0; y < height; y++)
	for(int x = 0; x < width; x++)
	{
		int srcPixelID = x + y * width;
		int dstPixelID = x + (height - 1 - y) * width;

		uint64_t pixel = pixels[srcPixelID];
		uint32_t color = uint32_t(pixel & 0xffffffffull);
		uint32_t udepth = uint32_t(pixel >> 32);

		float depth;
		memcpy(&depth, &udepth, sizeof(depth));

		uint8_t accumulatedOpacity = uint8_t((color >> 24) & 0xff);
		uint8_t transparencyMask = 255 - accumulatedOpacity;

		rgba[4 * dstPixelID + 0] = uint8_t((color >>  0) & 0xff);
		rgba[4 * dstPixelID + 1] = uint8_t((color >>  8) & 0xff);
		rgba[4 * dstPixelID + 2] = uint8_t((color >> 16) & 0xff);
		rgba[4 * dstPixelID + 3] = accumulatedOpacity;
		opacity[dstPixelID] = transparencyMask;
		depths[dstPixelID] = depth;
		opacities[dstPixelID] = float(transparencyMask) / 255.0f;

		minOpacity = min(minOpacity, opacities[dstPixelID]);
		maxOpacity = max(maxOpacity, opacities[dstPixelID]);

		if(std::isfinite(depth)){
			minDepth = min(minDepth, depth);
			maxDepth = max(maxDepth, depth);
			finiteDepthCount++;
		}
	}

	static uint64_t dumpID = 0;
	string basename = format("{}/framebuffer_{:04}", outputDirectory, dumpID++);
	string colorPath = basename + "_color.png";
	string opacityPath = basename + "_transparent_mask.png";
	string opacityFloatPath = basename + "_transparent_mask.pfm";
	string depthPath = basename + "_depth.pfm";
	string inpaintedPath = basename + "_inpainted.png";
	string mogePlyPath = basename + "_moge_gs.ply";
	string infoPath = basename + "_info.txt";
	string cameraPath = basename + "_camera.json";

	stbi_write_png(colorPath.c_str(), width, height, 4, rgba.data(), width * 4);
	stbi_write_png(opacityPath.c_str(), width, height, 1, opacity.data(), width);

	{
		std::ofstream file(opacityFloatPath, std::ios::binary);
		file << "Pf\n" << width << " " << height << "\n-1.0\n";
		file.write((char*)opacities.data(), opacities.size() * sizeof(float));
	}

	{
		std::ofstream file(depthPath, std::ios::binary);
		file << "Pf\n" << width << " " << height << "\n-1.0\n";
		file.write((char*)depths.data(), depths.size() * sizeof(float));
	}

	{
		glm::mat4 cameraWorld = glm::inverse(target.view);
		glm::vec4 cameraPosition = cameraWorld * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
		float proj00 = target.proj[0][0];
		float proj11 = target.proj[1][1];
		float aspect = proj11 / proj00;
		float fovyRadians = 2.0f * atan(1.0f / proj11);

		nlohmann::json camera = {
			{"width", width},
			{"height", height},
			{"coordinateSystem", "OpenGL camera, view space looks along -Z; depth is positive linear -view_z"},
			{"viewMatrix", dump_render_target_matrix_json(target.view)},
			{"projectionMatrix", dump_render_target_matrix_json(target.proj)},
			{"viewProjectionViewportMatrix", dump_render_target_matrix_json(target.VP)},
			{"cameraWorldMatrix", dump_render_target_matrix_json(cameraWorld)},
			{"position", {cameraPosition.x, cameraPosition.y, cameraPosition.z}},
			{"projection", {
				{"proj00", proj00},
				{"proj11", proj11},
				{"aspect", aspect},
				{"fovYRadians", fovyRadians},
				{"fovYDegrees", fovyRadians * 180.0f / glm::pi<float>()},
				{"fxPixels", 0.5f * float(width) * proj00},
				{"fyPixels", 0.5f * float(height) * proj11},
				{"cxPixels", 0.5f * float(width)},
				{"cyPixels", 0.5f * float(height)}
			}},
			{"files", {
				{"color", colorPath},
				{"transparentMask", opacityPath},
				{"transparentMaskFloat", opacityFloatPath},
				{"depth", depthPath}
			}}
		};

		writeFile(cameraPath.c_str(), camera.dump(4));
	}

	CUcontext splatshopCudaContext = nullptr;
	CURuntime::check(cuCtxGetCurrent(&splatshopCudaContext));

	LamaInpaintTimings inpaintTimings;
	int inpaintExitCode = LamaInpaint::inpaintFramebuffer(colorPath, opacityPath, inpaintedPath, &inpaintTimings);
	moge_gs::ExportResult mogeResult;
	int mogeExitCode = -2;
	string mogeInputPath = inpaintedPath;

	if (inpaintExitCode == 0 && std::filesystem::exists(inpaintedPath)) {
		CUcontext beforeMogeCudaContext = nullptr;
		CURuntime::check(cuCtxGetCurrent(&beforeMogeCudaContext));
		println("MoGe CUDA context before export: {}", uint64_t(beforeMogeCudaContext));

		moge_gs::ExportConfig mogeConfig;
		mogeConfig.mogeModelPath = std::getenv("SPLATSHOP_MOGE_CORE_MODEL")
			? std::getenv("SPLATSHOP_MOGE_CORE_MODEL")
			: R"(E:\projects\MoGe\workspace\moge_v1_forward_houseindoor_768_with_features_dynamic_query.onnx)";
		mogeConfig.gsModelPath = std::getenv("SPLATSHOP_MOGE_GS_MODEL")
			? std::getenv("SPLATSHOP_MOGE_GS_MODEL")
			: R"(E:\projects\MoGe\workspace\gs_predictor_houseindoor_768_tokens1734.onnx)";
		mogeConfig.imagePath = mogeInputPath;
		mogeConfig.outputPlyPath = mogePlyPath;
		mogeConfig.cudaDeviceId = std::getenv("SPLATSHOP_MOGE_CUDA_DEVICE")
			? std::stoi(std::getenv("SPLATSHOP_MOGE_CUDA_DEVICE"))
			: 0;
		mogeConfig.resizeTo = std::getenv("SPLATSHOP_MOGE_RESIZE_TO")
			? std::stoi(std::getenv("SPLATSHOP_MOGE_RESIZE_TO"))
			: 768;
		mogeConfig.warmup = 0;
		mogeConfig.repeat = 1;

		println("MoGe config:");
		println("  mogeModelPath: {}", mogeConfig.mogeModelPath);
		println("  gsModelPath: {}", mogeConfig.gsModelPath);
		println("  imagePath: {}", mogeConfig.imagePath);
		println("  outputPlyPath: {}", mogeConfig.outputPlyPath);
		println("  resizeTo: {}", mogeConfig.resizeTo);
		println("  cudaDeviceId: {}", mogeConfig.cudaDeviceId);

		mogeResult = moge_gs::export_to_ply(mogeConfig);
		mogeExitCode = mogeResult.exitCode;
		CUcontext afterMogeCudaContext = nullptr;
		CURuntime::check(cuCtxGetCurrent(&afterMogeCudaContext));
		println("MoGe CUDA context after export: {}", uint64_t(afterMogeCudaContext));

		if (mogeExitCode == 0) {
			println("MoGe GS prediction saved: {}", mogePlyPath);
		} else {
			println("MoGe GS prediction failed: {}", mogeResult.message);
		}
	} else {
		println("MoGe GS prediction skipped: inpainted image unavailable.");
	}
	// Temporarily disabled while maintaining only the Lama inpaint module.
	// MogeGsPredictorTimings mogeTimings;
	// int mogeExitCode = -2;
	// string mogeInputPath = inpaintedPath;
	// if(inpaintExitCode == 0 && std::filesystem::exists(mogeInputPath)){
	// 	CUcontext splatshopCudaContext = nullptr;
	// 	CURuntime::check(cuCtxGetCurrent(&splatshopCudaContext));
	// 	mogeExitCode = MogeGsPredictor::predictToPly(mogeInputPath, mogePlyPath, &mogeTimings);
	// 	if(splatshopCudaContext != nullptr){
	// 		CURuntime::check(cuCtxSetCurrent(splatshopCudaContext));
	// 		CURuntime::check(cuCtxSynchronize());
	// 	}
	// }else{
	// 	println("MoGe GS prediction skipped: inpainted image unavailable.");
	// }

	if (splatshopCudaContext != nullptr) {
		CURuntime::check(cuCtxSetCurrent(splatshopCudaContext));
		CURuntime::check(cuCtxSynchronize());
	}

	CUcontext restoredCudaContext = nullptr;
	CURuntime::check(cuCtxGetCurrent(&restoredCudaContext));
	println("Framebuffer dump CUDA context before LaMa/MoGe: {}", uint64_t(splatshopCudaContext));
	println("Framebuffer dump CUDA context after restore: {}", uint64_t(restoredCudaContext));

	string info = format(
		"width: {}\nheight: {}\nsource: virt_framebuffer->cptr\nlayout: uint64 color_low32 depth_high32\nalpha: accumulated opacity\ntransparentMask: 1.0 - accumulated opacity\ndepthConvention: positive linear -view_z in the exported camera coordinate system\nfiniteDepthCount: {}\nminDepth: {}\nmaxDepth: {}\nminTransparentMask: {}\nmaxTransparentMask: {}\ncameraPath: {}\ninpaintedPath: {}\ninpaintExitCode: {}\nlamaTotalMs: {}\nlamaModelLoadMs: {}\nlamaModelLoadedThisCall: {}\nlamaInputLoadAndPreprocessMs: {}\nlamaInferenceMs: {}\nlamaResultSaveMs: {}\n"
		"mogeInputPath: {}\nmogePlyPath: {}\nmogeExitCode: {}\nmogeTotalMs: {}\nmogeGaussianCount: {}",
		width, height, finiteDepthCount, minDepth, maxDepth, minOpacity, maxOpacity,
		cameraPath, inpaintedPath, inpaintExitCode, inpaintTimings.totalMs, inpaintTimings.modelLoadMs,
		inpaintTimings.modelLoadedThisCall, inpaintTimings.inputLoadAndPreprocessMs,
		inpaintTimings.inferenceMs, inpaintTimings.resultSaveMs,
		mogeInputPath,
		mogePlyPath,
		mogeExitCode,
		mogeResult.timings.total_ms,
		mogeResult.timings.gaussianCount
	);
	writeFile(infoPath.c_str(), info);

	println("Framebuffer dump saved: {}, {}, {}, {}, {}, {}, {}", colorPath, opacityPath, opacityFloatPath, depthPath, cameraPath, inpaintedPath, infoPath);
}

void SplatEditor::render(){

	cuStreamSynchronize(0);
	cuStreamSynchronize(mainstream);

	if(GLRenderer::width * GLRenderer::height == 0){
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		return;
	}

	{ // adjust framebuffer size
		GLRenderer::view.framebuffer->setSize(GLRenderer::width, GLRenderer::height);

		uint64_t requiredBytes = GLRenderer::width * GLRenderer::height * 8;
		virt_framebuffer->commit(requiredBytes);
	}

	glBindFramebuffer(GL_FRAMEBUFFER, GLRenderer::view.framebuffer->handle);

	static CUevent event_render_start;
	static CUevent event_render_end;
	static bool initialized = false;
	if(!initialized){
		cuEventCreate(&event_render_start, CU_EVENT_DEFAULT);
		cuEventCreate(&event_render_end, CU_EVENT_DEFAULT);
		initialized = true;
	}
	
	if(ovr->isActive())
	{ // RENDER VR

		if(Runtime::measureTimings){
			cuEventRecord(event_render_start, 0);
		}

		auto poseHMD = ovr->getHmdPose();
		auto poseLeft = ovr->getEyePose(vr::Hmd_Eye::Eye_Left);
		auto poseRight = ovr->getEyePose(vr::Hmd_Eye::Eye_Right);

		auto size = ovr->getRecommmendedRenderTargetSize();	
		int width = size[0];
		int height = size[1];

		

		Runtime::debugValues["vr resolution"] = format("{} x {}", width, height);

		if(!settings.enableOverlapped)
		{ // LEFT

			virt_framebuffer->commit(width * height * 8);

			RenderTarget target;
			target.width = width;
			target.height = height;
			target.framebuffer = (uint64_t*)virt_framebuffer->cptr;
			target.indexbuffer = nullptr;
			target.view = mat4(viewLeft.view);// * scene.transform;
			target.proj = viewLeft.proj;
			target.VP = viewLeft.VP;

			vector<RenderTarget> targets = { target };
			draw(&scene, targets);

			Rectangle targetViewport;
			targetViewport.x = 0;
			targetViewport.y = 0;
			targetViewport.width = width;
			targetViewport.height = height;

			auto glMapping = mapCudaGl(viewLeft.framebuffer->colorAttachments[0]);
			prog_gaussians_rendering->launch("kernel_blit_opengl", 
				{&launchArgs, &target, &glMapping.surface, &targetViewport}, 
				targetViewport.width * targetViewport.height);
			glMapping.unmap();
		}

		if(!settings.enableOverlapped)
		{ // RIGHT
			virt_framebuffer->commit(width * height * 8);

			RenderTarget target;
			target.width = width;
			target.height = height;
			target.framebuffer = (uint64_t*)virt_framebuffer->cptr;
			target.indexbuffer = nullptr;
			target.view = mat4(viewRight.view); // * scene.transform;
			target.proj = viewRight.proj;
			target.VP = viewRight.VP;

			vector<RenderTarget> targets = { target };
			draw(&scene, targets);

			Rectangle targetViewport;
			targetViewport.x = 0;
			targetViewport.y = 0;
			targetViewport.width = width;
			targetViewport.height = height;

			auto glMapping = mapCudaGl(viewRight.framebuffer->colorAttachments[0]);
			prog_gaussians_rendering->launch("kernel_blit_opengl", 
				{&launchArgs, &target, &glMapping.surface, &targetViewport}, 
				targetViewport.width * targetViewport.height);
			glMapping.unmap();
		}

		if(settings.enableOverlapped)
		{ // LEFT & RIGHT
			shared_ptr<CudaVirtualMemory> virt_framebuffer_left = virt_framebuffer;
			static shared_ptr<CudaVirtualMemory> virt_framebuffer_right = CURuntime::allocVirtual("framebuffer_right");

			virt_framebuffer_left->commit(width * height * 8);
			virt_framebuffer_right->commit(width * height * 8);

			RenderTarget target_left;
			target_left.width = width;
			target_left.height = height;
			target_left.framebuffer = (uint64_t*)virt_framebuffer_left->cptr;
			target_left.indexbuffer = nullptr;
			target_left.view = mat4(viewLeft.view); 
			target_left.proj = viewLeft.proj;
			target_left.VP = viewLeft.VP;
			target_left.isLeft = true;
			target_left.isRight = false;

			RenderTarget target_right;
			target_right.width = width;
			target_right.height = height;
			target_right.framebuffer = (uint64_t*)virt_framebuffer_right->cptr;
			target_right.indexbuffer = nullptr;
			target_right.view = mat4(viewRight.view); 
			target_right.proj = viewRight.proj;
			target_right.VP = viewRight.VP;
			target_right.isLeft = false;
			target_right.isRight = true;

			vector<RenderTarget> targets = { target_left, target_right };
			draw(&scene, targets);

			Rectangle targetViewport;
			targetViewport.x = 0;
			targetViewport.y = 0;
			targetViewport.width = width;
			targetViewport.height = height;

			auto glMapping_left = mapCudaGl(viewLeft.framebuffer->colorAttachments[0]);
			auto glMapping_right = mapCudaGl(viewRight.framebuffer->colorAttachments[0]);

			prog_gaussians_rendering->launch("kernel_blit_opengl", 
				{&launchArgs, &target_left, &glMapping_left.surface, &targetViewport}, 
				targetViewport.width * targetViewport.height);
			prog_gaussians_rendering->launch("kernel_blit_opengl", 
				{&launchArgs, &target_right, &glMapping_right.surface, &targetViewport}, 
				targetViewport.width * targetViewport.height);

			glMapping_left.unmap();
			glMapping_right.unmap();

		}

		// blit vr framebuffers to desktop framebuffer
		glBlitNamedFramebuffer(viewLeft.framebuffer->handle, GLRenderer::view.framebuffer->handle, 
			0, 0, width, height,
			0, 0, GLRenderer::view.framebuffer->width / 2, GLRenderer::view.framebuffer->height,
			GL_COLOR_BUFFER_BIT, GL_LINEAR
		);
		
		glBlitNamedFramebuffer(viewRight.framebuffer->handle, GLRenderer::view.framebuffer->handle, 
			0, 0, width, height,
			GLRenderer::view.framebuffer->width / 2, 0, GLRenderer::view.framebuffer->width, GLRenderer::view.framebuffer->height,
			GL_COLOR_BUFFER_BIT, GL_LINEAR
		);

		if(Runtime::measureTimings){
			cuEventRecord(event_render_end, 0);

			cuCtxSynchronize();

			float duration;
			cuEventElapsedTime(&duration, event_render_start, event_render_end);

			Runtime::timings.add("[render - VR]", duration);
		}

		// DEBUG - INSET
		// int insetSize = 64;
		// glBlitNamedFramebuffer(viewRight.framebuffer->handle, GLRenderer::view.framebuffer->handle, 
		// 	Runtime::mousePosition.x - insetSize / 2, Runtime::mousePosition.y - insetSize / 2, 
		// 	Runtime::mousePosition.x + insetSize / 2, Runtime::mousePosition.y + insetSize / 2,
		// 	500, 0, 500 + 1024, 0 + 1024,
		// 	GL_COLOR_BUFFER_BIT, GL_NEAREST
		// );

	}else{

		// RENDER DESKTOP

		if(Runtime::measureTimings){
			cuEventRecord(event_render_start, 0);
		}

		cuStreamSynchronize(0);

		// { // MAIN
		// 	RenderTarget target;
		// 	target.width = GLRenderer::width;
		// 	target.height = GLRenderer::height;
		// 	target.framebuffer = (uint64_t*)virt_framebuffer->cptr;
		// 	target.indexbuffer = nullptr;
		// 	target.view = mat4(GLRenderer::camera->view);
		// 	target.proj = GLRenderer::camera->proj;
			
		// 	draw(&scene, GLRenderer::view, target, mainstream, sidestream);

		// 	prog_gaussians_editing->launch("kernel_computeHoveredObject", {&launchArgs, &target}, 256, mainstream);

		// 	auto glMapping = mapCudaGl(GLRenderer::view.framebuffer->colorAttachments[0]);
		// 	prog_gaussians_rendering->launch("kernel_toOpenGL", {&launchArgs, &target, &glMapping.surface}, GLRenderer::width * GLRenderer::height, mainstream);
		// 	glMapping.unmap();
		// }

		if(settings.enableStereoFramebufferTest && settings.enableOverlapped){ 
			// render to multiple targets concurrently
			
			static shared_ptr<CudaVirtualMemory> virt_framebuffer_secondary = CURuntime::allocVirtual("secondary framebuffer (stereo concurrency test)");
			virt_framebuffer_secondary->commit(virt_framebuffer->comitted);

			RenderTarget target;
			target.width = GLRenderer::width;
			target.height = GLRenderer::height;
			target.framebuffer = (uint64_t*)virt_framebuffer->cptr;
			target.indexbuffer = nullptr;
			target.view = mat4(GLRenderer::camera->view);
			target.proj = GLRenderer::camera->proj;
			target.VP = GLRenderer::camera->VP;

			RenderTarget secondary = target;
			secondary.framebuffer = (uint64_t*)virt_framebuffer_secondary->cptr;
			
			vector<RenderTarget> targets = { target, secondary};
			draw(&scene, targets);

			Rectangle targetViewport_left;
			targetViewport_left.x = 0;
			targetViewport_left.y = 0;
			targetViewport_left.width = GLRenderer::width / 2;
			targetViewport_left.height = GLRenderer::height / 2;

			Rectangle targetViewport_right;
			targetViewport_right.x = GLRenderer::width / 2;
			targetViewport_right.y = 0;
			targetViewport_right.width = GLRenderer::width / 2;
			targetViewport_right.height = GLRenderer::height / 2;

			auto glMapping = mapCudaGl(GLRenderer::view.framebuffer->colorAttachments[0]);

			prog_gaussians_rendering->launch("kernel_blit_opengl", 
				{&launchArgs, &target, &glMapping.surface, &targetViewport_left}, 
				targetViewport_left.width * targetViewport_left.height);

			prog_gaussians_rendering->launch("kernel_blit_opengl", 
				{&launchArgs, &secondary, &glMapping.surface, &targetViewport_right}, 
				targetViewport_right.width * targetViewport_right.height);

			glMapping.unmap();
		}else if(settings.enableStereoFramebufferTest && !settings.enableOverlapped){ 

			{ // LEFT
				RenderTarget target;
				target.width = GLRenderer::width;
				target.height = GLRenderer::height;
				target.framebuffer = (uint64_t*)virt_framebuffer->cptr;
				target.indexbuffer = nullptr;
				target.view = mat4(GLRenderer::camera->view);
				target.proj = GLRenderer::camera->proj;
				target.VP = GLRenderer::camera->VP;

				draw(&scene, {target});

				Rectangle viewport;
				viewport.x = 0;
				viewport.y = 500;
				viewport.width = GLRenderer::width / 2;
				viewport.height = GLRenderer::height / 2;

				auto glMapping = mapCudaGl(GLRenderer::view.framebuffer->colorAttachments[0]);

				prog_gaussians_rendering->launch("kernel_blit_opengl", 
					{&launchArgs, &target, &glMapping.surface, &viewport}, 
					viewport.width * viewport.height);

				glMapping.unmap();
			}

			{ // RIGHT
				RenderTarget target;
				target.width = GLRenderer::width;
				target.height = GLRenderer::height;
				target.framebuffer = (uint64_t*)virt_framebuffer->cptr;
				target.indexbuffer = nullptr;
				target.view = mat4(GLRenderer::camera->view);
				target.proj = GLRenderer::camera->proj;
				target.VP = GLRenderer::camera->VP;

				draw(&scene, {target});

				Rectangle viewport;
				viewport.x = GLRenderer::width / 2;
				viewport.y = 500;
				viewport.width = GLRenderer::width / 2;
				viewport.height = GLRenderer::height / 2;

				auto glMapping = mapCudaGl(GLRenderer::view.framebuffer->colorAttachments[0]);

				prog_gaussians_rendering->launch("kernel_blit_opengl", 
					{&launchArgs, &target, &glMapping.surface, &viewport}, 
					viewport.width * viewport.height);

				glMapping.unmap();
			}



		}else{
			// standard single-target-rendering
			RenderTarget target;
			target.width = GLRenderer::width;
			target.height = GLRenderer::height;
			target.framebuffer = (uint64_t*)virt_framebuffer->cptr;
			target.indexbuffer = nullptr;
			target.view = mat4(GLRenderer::camera->view);
			target.proj = GLRenderer::camera->proj;
			target.VP = GLRenderer::camera->VP;
			
			vector<RenderTarget> targets = { target };
			draw(&scene, targets);

			prog_gaussians_editing->launch("kernel_computeHoveredObject", {&launchArgs, &target}, 256, mainstream);

			auto glMapping = mapCudaGl(GLRenderer::view.framebuffer->colorAttachments[0]);
			prog_gaussians_rendering->launch("kernel_toOpenGL", {&launchArgs, &target, &glMapping.surface}, GLRenderer::width * GLRenderer::height, mainstream);
			glMapping.unmap();

			if(settings.requestFramebufferDump){
				dump_render_target_framebuffer(target);
				settings.requestFramebufferDump = false;
			}
		}




		if(Runtime::measureTimings){
			cuEventRecord(event_render_end, 0);

			cuCtxSynchronize();

			float duration;
			cuEventElapsedTime(&duration, event_render_start, event_render_end);

			Runtime::timings.add("[render - desktop]", duration);
		}

		if(settings.showInset)
		{ // BLIT AN INSET
			cuCtxSynchronize();
			
			int insetSize = 32;
			int factor = 16;
			ivec2 center = {16 * 60 + 8, 16 * 50 + 8};
			// int start = {16 * 60 - 8, 16 * 50 - 8};
			// int end =   {16 * 60 + 8, 16 * 50 + 8};

			// first copy source area to bottom-left
			// ivec2 center = ivec2{GLRenderer::width / 2, GLRenderer::height / 2};
			glBlitNamedFramebuffer(GLRenderer::view.framebuffer->handle, GLRenderer::view.framebuffer->handle, 
				center.x - insetSize / 2, center.y - insetSize / 2, center.x + insetSize / 2, center.y + insetSize / 2,
				0, 0, insetSize, insetSize,
				GL_COLOR_BUFFER_BIT, GL_NEAREST
			);

			// then copy and rescale to target area
			glBlitNamedFramebuffer(GLRenderer::view.framebuffer->handle, GLRenderer::view.framebuffer->handle, 
				0, 0, insetSize, insetSize,
				center.x - factor * insetSize / 2, 
				10, //center.y - factor * insetSize / 2, 
				center.x + factor * insetSize / 2, 
				10 + factor * insetSize, // center.y + factor * insetSize / 2,
				GL_COLOR_BUFFER_BIT, GL_NEAREST
			);

			// make border in source area
			// left
			glBlitNamedFramebuffer(GLRenderer::view.framebuffer->handle, GLRenderer::view.framebuffer->handle, 
				200, 200, 1, 1,
				center.x - insetSize / 2, center.y - insetSize / 2, center.x - insetSize / 2 + 1, center.y + insetSize / 2,
				GL_COLOR_BUFFER_BIT, GL_NEAREST
			);
			// right
			glBlitNamedFramebuffer(GLRenderer::view.framebuffer->handle, GLRenderer::view.framebuffer->handle, 
				200, 200, 1, 1,
				center.x + insetSize / 2 - 1, center.y - insetSize / 2, center.x + insetSize / 2, center.y + insetSize / 2,
				GL_COLOR_BUFFER_BIT, GL_NEAREST
			);
			// bottom
			glBlitNamedFramebuffer(GLRenderer::view.framebuffer->handle, GLRenderer::view.framebuffer->handle, 
				200, 200, 1, 1,
				center.x - insetSize / 2, center.y - insetSize / 2, center.x + insetSize / 2, center.y - insetSize / 2 + 1,
				GL_COLOR_BUFFER_BIT, GL_NEAREST
			);
			// top
			glBlitNamedFramebuffer(GLRenderer::view.framebuffer->handle, GLRenderer::view.framebuffer->handle, 
				200, 200, 1, 1,
				center.x - insetSize / 2, center.y + insetSize / 2 - 1, center.x + insetSize / 2, center.y + insetSize / 2,
				GL_COLOR_BUFFER_BIT, GL_NEAREST
			);
		}
	}

	cuCtxSynchronize();
	CudaModularProgram::resolveTimings();
	
	// DRAW GUI
	if(!ovr->isActive()){

		ImGui::SetCurrentContext(imguicontext_desktop);

		drawGUI();

		// { // show vr gui in desktop mode
		// 	imguiStyleVR();
		// 	makePaintingVR(imn_painting->page);
		// 	ImGui::StyleColorsDark();
		// }

		Runtime::totalTileFragmentCount = 0;

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}else{

		{ // render desktop gui while in VR
			ImGui::SetCurrentContext(imguicontext_desktop);
			ImGuiIO& io = ImGui::GetIO();

			auto fbo = GLRenderer::view.framebuffer;
			glBindFramebuffer(GL_FRAMEBUFFER, fbo->handle);
			glViewport(0, 0, fbo->width, fbo->height);

			drawGUI();
			ImGui::Render();
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		}

		Runtime::totalTileFragmentCount = 0;
	}

	cuStreamSynchronize(0);
	cuStreamSynchronize(mainstream);
	cuCtxSynchronize();
	
	mouse_prev = Runtime::mouseEvents;

	if(viewmode == VIEWMODE_IMMERSIVE_VR && ovr->isActive()){

		auto size = ovr->getRecommmendedRenderTargetSize();	
		int width = size[0];
		int height = size[1];	

		vr::VRTextureBounds_t bounds = {
			0.0f, 
			1.0f - float(height) / float(viewLeft.framebuffer->height),
			float(width) / float(viewLeft.framebuffer->width), 
			1.0f,
		};

		ovr->submit(viewLeft.framebuffer->colorAttachments[0]->handle, vr::EVREye::Eye_Left, bounds);
		ovr->submit(viewRight.framebuffer->colorAttachments[0]->handle, vr::EVREye::Eye_Right, bounds);

		ovr->postPresentHandoff();
	}

	cuMemcpyDtoHAsync(h_state_pinned, cptr_state, sizeof(DeviceState), mainstream);
	memcpy(&deviceState, h_state_pinned, sizeof(DeviceState));

	Runtime::numSelectedSplats = deviceState.numSelectedSplats;

	settings.shortcutsDisabledForXFrames = max(settings.shortcutsDisabledForXFrames - 1, 0);

	Runtime::mouseEvents.clear();

	if(Runtime::measureTimings){
		CudaModularProgram::clearTimings();
	}

	ImGui::SetCurrentContext(imguicontext_desktop);

	postRenderStuff();
}
