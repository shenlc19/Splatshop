#include "RuntimeDllSearchPath.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace {

std::string env_or_default(const char* name, std::string fallback){
	const char* value = std::getenv(name);
	if(value && value[0] != '\0'){
		return value;
	}
	return fallback;
}

#ifdef _WIN32
std::wstring widen_utf8(const std::string& text){
	if(text.empty()){
		return {};
	}
	int size = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, nullptr, 0);
	if(size <= 0){
		return std::wstring(text.begin(), text.end());
	}
	std::wstring wide(static_cast<size_t>(size), L'\0');
	MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, wide.data(), size);
	if(!wide.empty() && wide.back() == L'\0'){
		wide.pop_back();
	}
	return wide;
}

void add_dll_directory(const std::string& directory){
	if(directory.empty() || !std::filesystem::exists(directory)){
		std::cout << "Runtime DLL directory skipped: " << directory << "\n";
		return;
	}

	std::wstring directoryW = widen_utf8(directory);
	AddDllDirectory(directoryW.c_str());

	DWORD pathChars = GetEnvironmentVariableW(L"PATH", nullptr, 0);
	std::wstring oldPath;
	if(pathChars > 0){
		oldPath.resize(pathChars - 1);
		GetEnvironmentVariableW(L"PATH", oldPath.data(), pathChars);
	}

	std::wstring newPath = directoryW;
	if(!oldPath.empty()){
		newPath += L";";
		newPath += oldPath;
	}
	SetEnvironmentVariableW(L"PATH", newPath.c_str());
	std::cout << "Runtime DLL directory added: " << directory << "\n";
}
#endif

} // namespace

void configureRuntimeDllSearchPath(){
#ifdef _WIN32
	SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
	add_dll_directory(env_or_default("INFINIDEPTH_ONNXRUNTIME_DLL_DIR", R"(E:\libs\onnxruntime-win-x64-gpu-1.23.2\lib)"));
	add_dll_directory(env_or_default("INFINIDEPTH_CUDA_DLL_DIR", R"(C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin)"));
	add_dll_directory(env_or_default("INFINIDEPTH_TORCH_DLL_DIR", R"(D:\miniconda\envs\moge\Lib\site-packages\torch\lib)"));
#endif
}
