#include "MogeGsPredictor.h"

#ifndef SPLATSHOP_MOGE_GS_PIPELINE_SOURCE
#define SPLATSHOP_MOGE_GS_PIPELINE_SOURCE "E:/projects/MoGe/cpp/moge_gs_pipeline.cpp"
#endif

#define main SplatshopMogeGsPipelineCliMain
#include SPLATSHOP_MOGE_GS_PIPELINE_SOURCE
#undef main

int run_moge_gs_pipeline_integrated(
	const std::string& mogeModelPath,
	const std::string& gsModelPath,
	const std::string& imagePath,
	int64_t numTokens,
	const std::string& outputPlyPath,
	int cudaDeviceId,
	int resizeTo,
	MogeGsPredictorTimings* timings
){
	MogeGsPredictorTimings localTimings;
	MogeGsPredictorTimings& t = timings ? *timings : localTimings;
	t = MogeGsPredictorTimings{};

	try{
		const auto totalBegin = Clock::now();

		Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "splatshop_moge_gs_pipeline");

		const auto modelLoadBegin = Clock::now();
		OnnxSession moge(env, mogeModelPath, cudaDeviceId);
		OnnxSession gs(env, gsModelPath, cudaDeviceId);
		t.modelLoadMs = elapsed_ms(modelLoadBegin, Clock::now());

		const auto preprocessBegin = Clock::now();
		ImageInputs image = preprocess_image(imagePath, numTokens, resizeTo);
		const TensorF32 firstQuery = make_query_tensor(make_query_coordinates_grid(image.original_height, image.original_width));
		t.preprocessMs = elapsed_ms(preprocessBegin, Clock::now());

		std::cout << "Image " << image.original_width << "x" << image.original_height
			<< ", MoGe resized " << image.resized_width << "x" << image.resized_height
			<< ", image_14 " << image.image14_width << "x" << image.image14_height << "\n";
		std::cout << "Preprocess: " << t.preprocessMs << " ms\n";

		const auto moge1Begin = Clock::now();
		auto mogeFirst = moge.run({
			{"image_14", &image.image_14},
			{"resized_image", &image.resized_image},
			{"query_coordinates", &firstQuery},
			{"uv0", &image.uv0},
			{"uv1", &image.uv1},
			{"uv2", &image.uv2},
		});
		t.mogeImageGridOnnxMs = elapsed_ms(moge1Begin, Clock::now());
		print_shape("moge.points", mogeFirst.at("points"));
		print_shape("moge.mask", mogeFirst.at("mask"));
		print_shape("moge.dino_tokens", mogeFirst.at("dino_tokens"));

		const auto postBegin = Clock::now();
		MoGePost post = postprocess_moge(mogeFirst.at("points"), mogeFirst.at("mask"), image.original_height, image.original_width);
		std::vector<float> sparseCoords = make_3d_uniform_coord_triangle_cpp(
			post.depth, image.original_height, image.original_width, post.pixel_k, kSparseSampleCount);
		TensorF32 sparseQuery = make_query_tensor(sparseCoords);
		t.mogePostprocessAndSparseCoordsMs = elapsed_ms(postBegin, Clock::now());

		const auto moge2Begin = Clock::now();
		auto mogeSecond = moge.run({
			{"image_14", &image.image_14},
			{"resized_image", &image.resized_image},
			{"query_coordinates", &sparseQuery},
			{"uv0", &image.uv0},
			{"uv1", &image.uv1},
			{"uv2", &image.uv2},
		});
		t.mogeSparseOnnxMs = elapsed_ms(moge2Begin, Clock::now());

		TensorF32 predDepth3d;
		predDepth3d.shape = {1, static_cast<int64_t>(sparseCoords.size() / 2), 1};
		predDepth3d.data.resize(sparseCoords.size() / 2);
		const TensorF32& predPoints3d = mogeSecond.at("points");
		for(size_t i = 0; i < predDepth3d.data.size(); ++i){
			predDepth3d.data[i] = predPoints3d.data[i * 3 + 2] + post.shift;
		}

		const int patchH = image.original_height / 16;
		const int patchW = image.original_width / 16;
		const int64_t requiredDinoTokens = static_cast<int64_t>(patchH) * patchW;
		const int64_t availableDinoTokens = mogeFirst.at("dino_tokens").shape.size() >= 2 ? mogeFirst.at("dino_tokens").shape[1] : 0;
		if(availableDinoTokens < requiredDinoTokens){
			throw std::runtime_error(
				"GS head needs at least " + std::to_string(requiredDinoTokens) +
				" DINO tokens for image size " + std::to_string(image.original_width) + "x" + std::to_string(image.original_height) +
				", but MoGe exported " + std::to_string(availableDinoTokens) +
				". Use a smaller --resize_to value or a larger num_tokens.");
		}

		TensorF32 depthmap = post.depth.empty() ? TensorF32{} : TensorF32{post.depth, {1, 1, image.original_height, image.original_width}};
		const auto gsBegin = Clock::now();
		auto gsRaw = gs.run({
			{"image", &image.image_rgb},
			{"depthmap", &depthmap},
			{"dino_tokens", &mogeFirst.at("dino_tokens")},
		});
		t.gsRawOnnxMs = elapsed_ms(gsBegin, Clock::now());
		print_shape("gs.opacities", gsRaw.at("opacities"));
		print_shape("gs.offset_xy", gsRaw.at("offset_xy"));
		print_shape("gs.raw_gaussians", gsRaw.at("raw_gaussians"));

		const auto adapterBegin = Clock::now();
		GaussianSet dense = build_dense_gaussians(
			image.image_rgb,
			gsRaw.at("opacities"),
			gsRaw.at("raw_gaussians"),
			image.original_height,
			image.original_width);
		GaussianSet sparse = build_sparse_gaussians(
			dense,
			sparseCoords,
			predDepth3d,
			post.pixel_k,
			image.original_height,
			image.original_width);
		GaussianSet filtered = filter_gaussians_by_scale_percentile(sparse);
		t.adapterAndFilterMs = elapsed_ms(adapterBegin, Clock::now());

		const auto saveBegin = Clock::now();
		write_ply(outputPlyPath, filtered, post.pixel_k, image.original_height, image.original_width);
		t.resultSaveMs = elapsed_ms(saveBegin, Clock::now());
		t.gaussianCount = filtered.n;
		t.totalMs = elapsed_ms(totalBegin, Clock::now());

		std::cout << "MoGe model load: " << t.modelLoadMs << " ms\n";
		std::cout << "MoGe image-grid ONNX: " << t.mogeImageGridOnnxMs << " ms\n";
		std::cout << "MoGe postprocess + sparse coords: " << t.mogePostprocessAndSparseCoordsMs << " ms\n";
		std::cout << "MoGe sparse ONNX: " << t.mogeSparseOnnxMs << " ms\n";
		std::cout << "GS raw ONNX: " << t.gsRawOnnxMs << " ms\n";
		std::cout << "GS adapter/filter: " << t.adapterAndFilterMs << " ms\n";
		std::cout << "MoGe PLY save: " << t.resultSaveMs << " ms\n";
		std::cout << "Wrote " << outputPlyPath << " (" << filtered.n << " gaussians)\n";
		return 0;
	}catch(const std::exception& e){
		std::cerr << "Error: " << e.what() << "\n";
		return 1;
	}
}
