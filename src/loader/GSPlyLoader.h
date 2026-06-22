#pragma once


#include <cmath>
#include <iostream>
#include <filesystem>
#include <print>
#include <format>
#include <memory>
#include <string>
#include <thread>

#include "unsuck.hpp"

#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/matrix_decompose.hpp>


#include "Splats.h"

using namespace std;

namespace fs = std::filesystem;


struct GSPlyHeader{
	int numSplats;
	int bytesPerSplat;
	int headerSize;
	int numSHCoefficients;
	int shDegree;

	int64_t OFFSETS_POSITION   = 0;
	int64_t OFFSETS_SCALE      = 0;
	int64_t OFFSETS_ROTATION   = 0;
	int64_t OFFSETS_OPACITY    = 0;
	int64_t OFFSETS_DC         = 0;
	int64_t OFFSETS_SHs        = 0;
};

struct GSPlyLoader{

	static int shDegreeToNumCoefficients(int degree){

		if(degree == 0) return 1;
		if(degree == 1) return 4;
		if(degree == 2) return 9;
		if(degree == 3) return 16;

		return 0;
	}

	static int numCoefficientsToShDegree(int numCoefficients){
		if(numCoefficients == 1) return 0;
		if(numCoefficients == 4) return 1;
		if(numCoefficients == 9) return 2;
		if(numCoefficients == 16) return 3;

		return 0;
	}

	static int plyScalarTypeSize(const string& type){
		if(type == "char" || type == "uchar" || type == "int8" || type == "uint8") return 1;
		if(type == "short" || type == "ushort" || type == "int16" || type == "uint16") return 2;
		if(type == "int" || type == "uint" || type == "int32" || type == "uint32") return 4;
		if(type == "float" || type == "float32") return 4;
		if(type == "double" || type == "float64") return 8;

		cout << format("type not implemented: {} \n", type);
		exit(123);
	}

	static GSPlyHeader readHeader(string path){

		int64_t numSplats = 0;

		auto potentialHeaderData = readBinaryFile(path, 0, 10'000);
		std::string strPotentialHeader(potentialHeaderData->data_char, potentialHeaderData->size);

		uint64_t posStartEndHeader = strPotentialHeader.find("end_header");
		uint64_t posEndHeader = strPotentialHeader.find('\n', posStartEndHeader);
		
		auto headerData = readBinaryFile(path, 0, posEndHeader);
		string strHeader(headerData->data_char, headerData->size);

		vector<string> lines = split(strHeader, '\n');
		int64_t numAttributesProcessed = 0;
		int64_t bytesPerSplat          = 0;
		int64_t numSHCoefficients      = 0;
		int64_t byteOffset         = 0;
		int64_t OFFSETS_POSITION   = 0;
		int64_t OFFSETS_SCALE      = 0;
		int64_t OFFSETS_ROTATION   = 0;
		int64_t OFFSETS_OPACITY    = 0;
		int64_t OFFSETS_DC         = 0;
		int64_t OFFSETS_SHs        = 0;
		bool parsingVertexElement  = false;

		for(string line : lines){

			// remove potentially present carriage-return characters
			// line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());

			vector<string> tokens = split(line, ' ');

			if(tokens.size() >= 3 && tokens[0] == "element"){
				parsingVertexElement = tokens[1] == "vertex";
				if(parsingVertexElement){
					numSplats = std::stoi(tokens[2]);
				}
			}else if(tokens.size() >= 3 && tokens[0] == "property" && parsingVertexElement){
				string type = tokens[1];
				string name = tokens[2];

				if(name == "x"){
					// attributes.x = numAttributesProcessed;
					OFFSETS_POSITION = byteOffset;
				}else if(name == "y"){
					// attributes.y = numAttributesProcessed;
				}else if(name == "z"){
					// attributes.z = numAttributesProcessed;
				}else if(name == "f_dc_0"){
					OFFSETS_DC = byteOffset;
				}else if(name == "f_rest_0"){
					OFFSETS_SHs = byteOffset;
				}else if(name == "opacity"){
					OFFSETS_OPACITY = byteOffset;
				}else if(name == "scale_0"){
					OFFSETS_SCALE = byteOffset;
				}else if(name == "rot_0"){
					OFFSETS_ROTATION = byteOffset;
				}

				if(name.starts_with("f_rest"s)){
					numSHCoefficients++;
				}

				byteOffset += plyScalarTypeSize(type);
				numAttributesProcessed++;
			}
		}

		bytesPerSplat = byteOffset;

		int64_t shDegree = 0;
		int64_t SHs = (numSHCoefficients + 3) / 3;
		if(SHs == 1){
			shDegree = 0;
		}else if(SHs == 4){
			shDegree = 1;
		}else if(SHs == 9){
			shDegree = 2;
		}else if(SHs == 16){
			shDegree = 3;
		}else{
			cout << format("unkown amount of SH components: {} \n", SHs);
			exit(123);
		}

		GSPlyHeader header;
		header.numSplats         = numSplats;
		header.bytesPerSplat     = bytesPerSplat;
		header.headerSize        = posEndHeader;
		header.numSHCoefficients = numSHCoefficients;
		header.shDegree          = shDegree;
		header.OFFSETS_POSITION  = OFFSETS_POSITION;
		header.OFFSETS_SCALE     = OFFSETS_SCALE;
		header.OFFSETS_ROTATION  = OFFSETS_ROTATION;
		header.OFFSETS_OPACITY   = OFFSETS_OPACITY;
		header.OFFSETS_DC        = OFFSETS_DC;
		header.OFFSETS_SHs       = OFFSETS_SHs;

		return header;
	}


	static shared_ptr<Splats> load(string path){

		auto header = readHeader(path);

		// header.numSplats = min(header.numSplats, 40'000'000);

		println("numSplats:         {}", header.numSplats);
		println("bytes per splat:   {}", header.bytesPerSplat);
		println("shDegree:          {}", header.shDegree);
		println("numSHCoefficients: {}", header.numSHCoefficients);

		// deactivate SHs
		// header.numSHCoefficients = 0;
		// header.shDegree = 0;

		int64_t numSHBytes = header.numSHCoefficients * sizeof(float);

		shared_ptr<Splats> splats = make_shared<Splats>();

		splats->name = fs::path(path).filename().string();

		splats->numSplatsLoaded   = 0;
		splats->numSplats         = header.numSplats;
		splats->bytesPerSplat     = header.bytesPerSplat;
		splats->headerSize        = header.headerSize;
		splats->numSHCoefficients = header.numSHCoefficients;
		splats->shDegree          = header.shDegree;

		splats->position = make_shared<Buffer>(header.numSplats * sizeof(vec3));
		splats->scale    = make_shared<Buffer>(header.numSplats * sizeof(vec3));
		splats->rotation = make_shared<Buffer>(header.numSplats * sizeof(vec4));
		splats->color    = make_shared<Buffer>(header.numSplats *  8llu);
		splats->SHs      = make_shared<Buffer>(header.numSplats *  numSHBytes);

		
		auto t = jthread([header, path, splats, numSHBytes](){

			double t_start = now();

			int64_t batchSize = 100'000;
			Buffer buffer(batchSize * header.bytesPerSplat);

			for(int64_t i = 0; i < header.numSplats; i++){

				bool dbgSplat = splats->numSplatsLoaded > 33'000'000 && splats->numSplatsLoaded < 33'000'010;

				if(i % batchSize == 0){
					uint64_t start = header.headerSize + 1 + i * header.bytesPerSplat;
					uint64_t remainingBytes = (header.numSplats - i) * header.bytesPerSplat;
					uint64_t batchByteSize = std::min(remainingBytes, uint64_t(batchSize * header.bytesPerSplat));
					readBinaryFile(path, start, batchByteSize, buffer.data);
				}

				uint64_t srcOffset = (i % batchSize) * header.bytesPerSplat;

				{ // POSITION
					auto pos = glm::vec4(
						buffer.get<float>(srcOffset + header.OFFSETS_POSITION + 0llu),
						buffer.get<float>(srcOffset + header.OFFSETS_POSITION + 4llu),
						buffer.get<float>(srcOffset + header.OFFSETS_POSITION + 8llu),
						1.0
					);

					splats->min.x = min(splats->min.x, pos.x);
					splats->min.y = min(splats->min.y, pos.y);
					splats->min.z = min(splats->min.z, pos.z);
					splats->max.x = max(splats->max.x, pos.x);
					splats->max.y = max(splats->max.y, pos.y);
					splats->max.z = max(splats->max.z, pos.z);
					
					splats->position->set(pos.x, 12llu * i + 0llu);
					splats->position->set(pos.y, 12llu * i + 4llu);
					splats->position->set(pos.z, 12llu * i + 8llu);

					if(dbgSplat){
						println("{}, {}, {}", pos.x, pos.y, pos.z);
					}
				}

				{ // SCALE
					float sx = buffer.get<float>(srcOffset + header.OFFSETS_SCALE + 0llu);
					float sy = buffer.get<float>(srcOffset + header.OFFSETS_SCALE + 4llu);
					float sz = buffer.get<float>(srcOffset + header.OFFSETS_SCALE + 8llu);

					sx = exp(sx);
					sy = exp(sy);
					sz = exp(sz);

					splats->scale->set(sx, 12 * i + 0llu);
					splats->scale->set(sy, 12 * i + 4llu);
					splats->scale->set(sz, 12 * i + 8llu);
				}

				{ // ROTATION
					float w = buffer.get<float>(srcOffset + header.OFFSETS_ROTATION +  0llu);
					float x = buffer.get<float>(srcOffset + header.OFFSETS_ROTATION +  4llu);
					float y = buffer.get<float>(srcOffset + header.OFFSETS_ROTATION +  8llu);
					float z = buffer.get<float>(srcOffset + header.OFFSETS_ROTATION + 12llu);

					float length = sqrt(x * x + y * y + z * z + w * w);

					auto q = glm::quat(
						w / length, 
						x / length, 
						y / length,
						z / length 
					);

					splats->rotation->set(q.w, 16 * i +  0llu);
					splats->rotation->set(q.x, 16 * i +  4llu);
					splats->rotation->set(q.y, 16 * i +  8llu);
					splats->rotation->set(q.z, 16 * i + 12llu);
				}

				{ // COLOR
					float r = buffer.get<float>(srcOffset + header.OFFSETS_DC + 0llu);
					float g = buffer.get<float>(srcOffset + header.OFFSETS_DC + 4llu);
					float b = buffer.get<float>(srcOffset + header.OFFSETS_DC + 8llu);

					float CO = 0.28209479177387814; // spherical harmonics coefficient
					r = clamp(0.5 + CO * r, 0.0, 1.0);
					g = clamp(0.5 + CO * g, 0.0, 1.0);
					b = clamp(0.5 + CO * b, 0.0, 1.0);

					splats->color->set<uint16_t>(clamp(65536.0f * r, 0.0f, 65535.0f), 8llu * i + 0llu);
					splats->color->set<uint16_t>(clamp(65536.0f * g, 0.0f, 65535.0f), 8llu * i + 2llu);
					splats->color->set<uint16_t>(clamp(65536.0f * b, 0.0f, 65535.0f), 8llu * i + 4llu);
				}

				if(splats->SHs){
					// SHs are stored per splat but in [rrr..][ggg..][bbb..] fashion. 
					// We need to rearange to [rgb][rgb]...

					for(int j = 0; j < header.numSHCoefficients / 3; j++){

						int64_t numVec3s = header.numSHCoefficients / 3;

						float a = buffer.get<float>(srcOffset + header.OFFSETS_SHs + 4 * (0 * numVec3s + j));
						float b = buffer.get<float>(srcOffset + header.OFFSETS_SHs + 4 * (1 * numVec3s + j));
						float c = buffer.get<float>(srcOffset + header.OFFSETS_SHs + 4 * (2 * numVec3s + j));

						splats->SHs->set<float>(a, numSHBytes * i + 12 * j + 0);
						splats->SHs->set<float>(b, numSHBytes * i + 12 * j + 4);
						splats->SHs->set<float>(c, numSHBytes * i + 12 * j + 8);
					}

					// memcpy(
					// 	splats->SHs->ptr + numSHBytes * i, 
					// 	buffer.ptr + srcOffset + header.OFFSETS_SHs, 
					// 	numSHBytes
					// );
				}

				float opacity = buffer.get<float>(srcOffset + header.OFFSETS_OPACITY);
				opacity = (1.0 / (1.0 + std::exp(-opacity)));
				// splats->color->set<float>(opacity, 16 * i + 12);
				// splats->color->set<uint8_t>(255.0f * opacity, 4 * i + 3);
				splats->color->set<uint16_t>(clamp(65536.0f * opacity, 0.0f, 65535.0f), 8llu * i + 6llu);

				splats->numSplatsLoaded++;
			}

			double t_end = now();
			double seconds = t_end - t_start;

			println("loaded {:L} splats in {:.3f} seconds.", splats->numSplatsLoaded, seconds);
		});

		// t.join();
		t.detach();

		return splats;
	}


};
