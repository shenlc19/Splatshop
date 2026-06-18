#pragma once

#include <string>

struct LamaInpaintTimings{
	double totalMs = 0.0;
	double modelLoadMs = 0.0;
	double inputLoadAndPreprocessMs = 0.0;
	double inferenceMs = 0.0;
	double resultSaveMs = 0.0;
	bool modelLoadedThisCall = false;
};

struct LamaInpaint{
	static int inpaintFramebuffer(
		const std::string& imagePath,
		const std::string& maskPath,
		const std::string& outputPath,
		LamaInpaintTimings* timings = nullptr
	);
};
