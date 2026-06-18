#pragma once

#include <cstddef>
#include <string>

struct MogeGsPredictorTimings{
	double totalMs = 0.0;
	double modelLoadMs = 0.0;
	double preprocessMs = 0.0;
	double mogeImageGridOnnxMs = 0.0;
	double mogePostprocessAndSparseCoordsMs = 0.0;
	double mogeSparseOnnxMs = 0.0;
	double gsRawOnnxMs = 0.0;
	double adapterAndFilterMs = 0.0;
	double resultSaveMs = 0.0;
	size_t gaussianCount = 0;
};

struct MogeGsPredictor{
	static int predictToPly(
		const std::string& imagePath,
		const std::string& outputPlyPath,
		MogeGsPredictorTimings* timings = nullptr
	);
};
