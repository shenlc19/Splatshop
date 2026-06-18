
#include "GLRenderer.h"
#include "common.h"

#include "loader/GSPlyLoader.h"
#include "loader/SplatsyLoader.h"
#include "loader/GLBLoader.h"
#include "loader/LASLoader.h"

#include "Runtime.h"
#include "CURuntime.h"
#include "ImageLoader.h"
#include "SplatEditor.h"

using namespace std;

SplatEditor* editor = nullptr;

void initCuda(){
	cuInit(0);

	CUcontext context;
	cuDeviceGet(&CURuntime::device, 0);
	cuCtxCreate(&context, 0, CURuntime::device);

	// cuCtxSetLimit(CU_LIMIT_MALLOC_HEAP_SIZE, 10'000'000'000);

	CUmemAllocationProp prop = {};
	prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
	prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	prop.location.id   = CURuntime::device;

	size_t granularity_minimum;
	size_t granularity_recommended;
	cuMemGetAllocationGranularity(&granularity_minimum, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
	cuMemGetAllocationGranularity(&granularity_recommended, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);

	println("granularity_minimum:     {:L}", granularity_minimum);
	println("granularity_recommended: {:L}", granularity_recommended);
}

void initScene(string path) {

	Runtime::controls->yaw    = -1.325;
	Runtime::controls->pitch  = -0.330;
	Runtime::controls->radius = 4.691;
	Runtime::controls->target = { -0.028, -0.100, 2.301, };

	// SplatsyFilesLoader::load("./savefile_ref/scene.json", editor->scene, *Runtime::controls);
	// SplatsyFilesLoader::load("./splatmodels_lantern/scene.json", editor->scene, *Runtime::controls);

	// string path = "./gaussians_w_pca.ply";
	// string path = "./splatmodels_benchmark_garden_far/scene.json";
	// string path = "E:/resources/gaussian_splats/garden.ply";
	// string path = "E:/resources/splats/gardentable.ply";
	// string path = "./splatmodels_3dgs_and_perspcorrect/scene.json";
	// string path = "/home/hahlbohm/code/nerficg_public/nerficg/output/HTGS/htgs_garden_2025-03-06-20-25-48/gaussians.ply";
	// string path = "F:/SplatEditor/city_gaussians_mc_aerial_c36.ply";
	// string path = "F:/SplatEditor/splatmodels.json";

	if(fs::exists(path)){
		if(iEndsWith(path, ".json")){
			SplatsyFilesLoader::load(path, editor->scene, *Runtime::controls);
		}else if(iEndsWith(path, ".ply")){
			auto splats = GSPlyLoader::load(path);
			shared_ptr<SNSplats> node = make_shared<SNSplats>(splats->name, splats);
			editor->scene.world->children.push_back(node);
		}

		// Runtime::controls->yaw    = 3.155;
		// Runtime::controls->pitch  = -0.220;
		// Runtime::controls->radius = 3.877;
		// Runtime::controls->target = { 0.353, 0.518, 1.240, };
	}else{
		println("Could not find file {}", path);
	}

	
	// {// Lot's of gardens
	// 	string path = "./garden.ply";

	// 	for(int i = 0; i < 5; i++)
	// 	for(int j = 0; j < 5; j++)
	// 	{
	// 		auto splats = GSPlyLoader::load(path);
	// 		shared_ptr<SNSplats> node = make_shared<SNSplats>(splats->name, splats);

	// 		node->transform = translate(vec3{float(30.0f * i), float(25.0f * j), 0.0f});

	// 		editor->scene.world->children.push_back(node);
	// 	}

	// 	Runtime::controls->yaw    = 3.155;
	// 	Runtime::controls->pitch  = -0.220;
	// 	Runtime::controls->radius = 3.877;
	// 	Runtime::controls->target = { 0.353, 0.518, 1.240, };
	// }


	


	// { // CITY GAUSSIANS
	// 	// string path = "F:/CityGaussians/mc_aerial_c36/point_cloud/iteration_30000/point_cloud.ply";
	// 	string path = "E:/resources/splats/city_gaussians_mc_aerial_c36.ply";
	// 	auto splats = GSPlyLoader::load(path);

	// 	shared_ptr<SNSplats> node = make_shared<SNSplats>(splats->name, splats);

	// 	mat4 world = {
	// 		-0.117,     -0.009,      0.000,      0.000,
	// 		 0.009,     -0.117,      0.000,      0.000,
	// 		 0.000,      0.000,      0.118,      0.000,
	// 		-0.234,     -0.243,      0.166,      1.000
	// 	};

	// 	node->transform = world * mat4{
	// 		1.000,      0.000,      0.000,      0.000,
	// 		0.000,      1.000,      0.000,      0.000,
	// 		0.000,      0.000,      1.000,      0.000,
	// 		5.871,     -7.894,      0.517,      1.000
	// 	};

	// 	editor->scene.world->children.push_back(node);

		

	// 	// position: -3.8790769445903313, 12.585053772306223, -8.25656827876122 
	// 	Runtime::controls->yaw    = -2.925;
	// 	Runtime::controls->pitch  = -0.583;
	// 	Runtime::controls->radius = 15.792;
	// 	Runtime::controls->target = { -0.485, -0.294, 0.228, };

	// }


	
	// { // CAMPUS
	// 	string path = "E:/resources/splats/campus.ply";

	// 	if(fs::exists(path)){
	// 		shared_ptr<Splats> splats = GSPlyLoader::load(path);

	// 		shared_ptr<SNSplats> node = make_shared<SNSplats>(splats->name, splats);
	// 		node->transform = mat4(
	// 			1.0f, 0.0f, 0.0f, 0.0f,
	// 			0.0f, 0.0f, -1.0f, 0.0f,
	// 			0.0f, 1.0f, 0.0f, 0.0f,
	// 			0.0f, 0.0f, 0.0f, 1.0f
	// 		);

	// 		editor->scene.world->children.push_back(node);

	// 		// position: -343.2103134600034, 113.59831636346553, -425.2667003230538 
	// 		Runtime::controls->yaw    = -3.010;
	// 		Runtime::controls->pitch  = -0.545;
	// 		Runtime::controls->radius = 518.481;
	// 		Runtime::controls->target = { -275.179, -325.936, -158.800, };
	// 	}
	// }

	// { // CAMPUS2
	// 	string path = "F:/Campus/point_cloud.ply";

	// 	if(fs::exists(path)){
	// 		shared_ptr<Splats> splats = GSPlyLoader::load(path);

	// 		shared_ptr<SNSplats> node = make_shared<SNSplats>(splats->name, splats);
	// 		node->transform = mat4(
	// 			1.0f, 0.0f, 0.0f, 0.0f,
	// 			0.0f, 0.0f, -1.0f, 0.0f,
	// 			0.0f, 1.0f, 0.0f, 0.0f,
	// 			1880.0f, 51.0f, 0.0f, 1.0f
	// 		);

	// 		editor->scene.world->children.push_back(node);

	// 		// position: -343.2103134600034, 113.59831636346553, -425.2667003230538 
	// 		Runtime::controls->yaw    = -3.010;
	// 		Runtime::controls->pitch  = -0.545;
	// 		Runtime::controls->radius = 518.481;
	// 		Runtime::controls->target = { -275.179, -325.936, -158.800, };
	// 	}
	// }

	// { // CAMPUS2
	// 	string path = "F:/Campus/point_cloud.ply";

	// 	if(fs::exists(path)){
	// 		shared_ptr<Splats> splats = GSPlyLoader::load(path);

	// 		shared_ptr<SNSplats> node = make_shared<SNSplats>(splats->name, splats);
	// 		node->transform = mat4(
	// 			1.0f, 0.0f, 0.0f, 0.0f,
	// 			0.0f, 0.0f, -1.0f, 0.0f,
	// 			0.0f, 1.0f, 0.0f, 0.0f,
	// 			0.0f, -2000.0f, 0.0f, 1.0f
	// 		);

	// 		editor->scene.world->children.push_back(node);

	// 		// position: -343.2103134600034, 113.59831636346553, -425.2667003230538 
	// 		// position: -1591.913119322585, 1651.705754893393, -1863.6676586831395 
	// 		Runtime::controls->yaw    = -2.490;
	// 		Runtime::controls->pitch  = -0.767;
	// 		Runtime::controls->radius = 2882.711;
	// 		Runtime::controls->target = { 156.317, 2.200, -272.176, };

	// 	}
	// }

	// { // CAMPUS3
	// 	string path = "F:/Campus/point_cloud.ply";

	// 	if(fs::exists(path)){
	// 		shared_ptr<Splats> splats = GSPlyLoader::load(path);

	// 		shared_ptr<SNSplats> node = make_shared<SNSplats>(splats->name, splats);
	// 		node->transform = mat4(
	// 			1.0f, 0.0f, 0.0f, 0.0f,
	// 			0.0f, 0.0f, -1.0f, 0.0f,
	// 			0.0f, 1.0f, 0.0f, 0.0f,
	// 			1880.0f, -2000.0f, 0.0f, 1.0f
	// 		);

	// 		editor->scene.world->children.push_back(node);

	// 		// position: -343.2103134600034, 113.59831636346553, -425.2667003230538 
	// 		// position: -1591.913119322585, 1651.705754893393, -1863.6676586831395 
	// 		Runtime::controls->yaw    = -2.490;
	// 		Runtime::controls->pitch  = -0.767;
	// 		Runtime::controls->radius = 2882.711;
	// 		Runtime::controls->target = { 156.317, 2.200, -272.176, };

	// 	}
	// }

}

int main(int argc, char** argv){

	initCuda();

	GLRenderer::init();

	SplatEditor::setup();
	editor = SplatEditor::instance;

	{ // load some textures
		int n;
		string imgPath = "./resources/images/symbols.png";
		uint8_t* data = stbi_load(imgPath.c_str(), &Runtime::gltex_symbols_width, &Runtime::gltex_symbols_height, &n, 4);

		int numPixels = Runtime::gltex_symbols_width * Runtime::gltex_symbols_height;
		for(int i = 0; i < numPixels; i++){
			data[4 * i + 0] = 255 - data[4 * i + 0];
			data[4 * i + 1] = 255 - data[4 * i + 1];
			data[4 * i + 2] = 255 - data[4 * i + 2];
		}

		glGenTextures(1, &Runtime::gltex_symbols);
		glBindTexture(GL_TEXTURE_2D, Runtime::gltex_symbols);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, Runtime::gltex_symbols_width, Runtime::gltex_symbols_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
		glGenerateMipmap(GL_TEXTURE_2D);

		stbi_image_free(data);
	}

	{ // load some textures
		int n;
		string imgPath = "./resources/images/symbols_32x32.png";
		uint8_t* data = stbi_load(imgPath.c_str(), &Runtime::gltex_symbols_32x32_width, &Runtime::gltex_symbols_32x32_height, &n, 4);

		int numPixels = Runtime::gltex_symbols_32x32_width * Runtime::gltex_symbols_32x32_height;
		for(int i = 0; i < numPixels; i++){
			data[4 * i + 0] = data[4 * i + 0];
			data[4 * i + 1] = data[4 * i + 1];
			data[4 * i + 2] = data[4 * i + 2];
		}

		glGenTextures(1, &Runtime::gltex_symbols_32x32);
		glBindTexture(GL_TEXTURE_2D, Runtime::gltex_symbols_32x32);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, Runtime::gltex_symbols_32x32_width, Runtime::gltex_symbols_32x32_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
		glGenerateMipmap(GL_TEXTURE_2D);

		stbi_image_free(data);
	}

	// GPUPrefixSums::init();
	// GPUSorting::initSorting();

	string scenePath = "models/counter/point_cloud/iteration_30000/point_cloud.ply";
	if(argc >= 2){
		scenePath = argv[1];
	}

	initScene(scenePath);

	glfwSetDropCallback(GLRenderer::window, [](GLFWwindow* window, int count, const char** paths){
		
		for(int i = 0; i < count; i++){
			string path = paths[i];

			if(fs::exists(path)){
				if(iEndsWith(path, ".json")){
					SplatsyFilesLoader::load(path, editor->scene, *Runtime::controls);
				}else if(iEndsWith(path, ".ply")){
					auto splats = GSPlyLoader::load(path);
					shared_ptr<SNSplats> node = make_shared<SNSplats>(splats->name, splats);
					editor->scene.world->children.push_back(node);
					editor->setSelectedNode(node.get());
				}else if(fs::is_directory(path)){

					bool hasSceneJson = fs::exists(path + "/scene.json");
					bool hasAssetDir = fs::exists(path + "/assets");
					bool hasSplatsDir = fs::exists(path + "/splats");

					if(hasSceneJson && hasAssetDir && hasSplatsDir){
						SplatsyFilesLoader::load(path + "/scene.json", editor->scene, *Runtime::controls);
					}
				}else if(iEndsWith(path, ".glb")){
					// auto glb = GLBLoader::load(path);

					// shared_ptr<SNTriangles> node = make_shared<SNTriangles>(path);
					// node->set(glb.positions, glb.uvs);
					// node->setTexture(glb.textureSize, glb.texture->data);

					// editor->scene.world->children.push_back(node);

					// Runtime::controls->focus(node->aabb.min, node->aabb.max, 1.0f);


					GLBLoader::load(path, [&](GLB glb){
						shared_ptr<SNTriangles> node = make_shared<SNTriangles>(path);
						node->set(glb.positions, glb.uvs);
						node->setTexture(glb.textureSize, glb.texture->data);

						editor->scene.world->children.push_back(node);

						Runtime::controls->focus(node->aabb.min, node->aabb.max, 1.0f);
					});
				}

				
			}else{
				println("Could not find file {}", path);
			}
		}
	});

	// installDragEnterHandler(GLRenderer::window, [&](string file){
	// 	if(iEndsWith(file, ".ply")){
	// 		editor->loadTempSplats(file);
	// 	}else if(iEndsWith(file, ".las")){

	// 	}
	// });

	// installDragDropHandler(GLRenderer::window, [&](string file){
	// 	Scene& scene = editor->scene;
	// 	auto tempSplats = editor->tempSplats;

	// 	if(tempSplats){
	// 		tempSplats->dmng.data.writeDepth = true;
	// 		tempSplats->hidden = false;

	// 		scene.deselectAllNodes();
	// 		// editor->sortSplats(tempSplats.get());
	// 		editor->sortSplatsDevice(tempSplats.get());

	// 		tempSplats->selected = true;
	// 		editor->tempSplats = nullptr;
	// 	}

	// 	if(iEndsWith(file, ".las")){
	// 		auto points = LasLoader::load(file);

	// 		shared_ptr<SNPoints> node = make_shared<SNPoints>(points->name, points);
	// 		scene.world->children.push_back(node);
	// 	}else if(iEndsWith(file, ".glb")){
	// 		auto glb = GLBLoader::load(file);

	// 		shared_ptr<SNTriangles> node = make_shared<SNTriangles>(file);
	// 		node->set(glb.positions, glb.uvs);
	// 		node->setTexture(glb.textureSize, glb.texture->data);

	// 		scene.world->children.push_back(node);

	// 		Runtime::controls->focus(node->aabb.min, node->aabb.max, 1.0f);
	// 	}
	// });

	// installDragOverHandler(GLRenderer::window, [&](int32_t x, int32_t y){
	// 	// - During WinAPI drag, the regular glfw mouse callbacks are supressed.
	// 	//   So instead, listen to the WinAPI for mouse changes.
	// 	// - DragOver returns screen coordinates. We need to transform them to window coordinates. 
	// 	int window_x, window_y;
	// 	glfwGetWindowPos(GLRenderer::window, &window_x, &window_y);

	// 	Runtime::mousePosition.x = x - window_x;
	// 	Runtime::mousePosition.y = GLRenderer::height - (y - window_y);

	// 	Runtime::mouseEvents.pos_x = x - window_x;
	// 	Runtime::mouseEvents.pos_y = GLRenderer::height - (y - window_y);
	// });

	GLRenderer::loop(
		[&]() {editor->update();},
		[&]() {editor->render();}
	);

	return 0;
}
