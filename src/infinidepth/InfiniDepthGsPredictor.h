#pragma once

#include <cstdint>
#include <string>

struct InfiniDepthGsTimings{
	double totalMs = 0.0;
	double modelLoadMs = 0.0;
	double inputProcessingMs = 0.0;
	double depthInferenceMs = 0.0;
	double gsInferenceAndExportMs = 0.0;
	bool modelsLoadedThisCall = false;
	bool cacheSessions = true;
	int64_t gaussianCount = 0;
	int64_t removedCount = 0;
};

struct InfiniDepthGsPredictor{
	static int predictToPly(
		const std::string& imagePath,
		const std::string& depthPath,
		const std::string& cameraJsonPath,
		const std::string& outputPlyPath,
		InfiniDepthGsTimings* timings = nullptr
	);

	static void clearCache();
};
