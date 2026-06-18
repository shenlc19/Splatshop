#pragma once

#include "glm/gtc/matrix_access.hpp"

#ifdef NO_OPENVR
	#include "NOpenVRHelper.h"
#else
	#include "OpenVRHelper.h"
#endif
#include "CudaVirtualMemory.h"

#include "Mesh.h"
#include "./scene/SceneNode.h"
#include "./scene/Scene.h"
#include "./scene/SNSplats.h"
#include "./scene/ImguiNode.h"
#include "./scene/SNPoints.h"
#include "./scene/SNTriangles.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "CudaModularProgram.h"

#include "OrbitControls.h"
#include "DesktopVRControls.h"
#include "./actions/InputAction.h"

#include "common.h"
#include "AssetLibrary.h"

#include "loader/GSPlyLoader.h"
#include "loader/SplatsyFilesLoader.h"
#include "loader/SplatsyLoader.h"
#include "loader/PLYHeightmapLoader.h"
#include "loader/GLBLoader.h"
#include "loader/LASLoader.h"
#include "writer/GSPlyWriter.h"
#include "writer/SplatsyWriter.h"
#include "writer/SplatsyFilesWriter.h"
// #include "writer/SplatsyPlyWriter.h"
#include "Runtime.h"
#include "tween.h"

#include "common.h"
#include "utils.h"

struct Action;
struct InputAction;

using glm::transpose;
using glm::vec2;
using glm::quat;
using glm::vec3;

struct TriangleQueueItem{
	TriangleData geometry;
	TriangleMaterial material;
};

struct GuiVrPage{
	string label;
	int width;
	int height;
};

struct _DrawSphereArgs{
	vec3 pos   = {0.0f, 0.0f, 0.0f};
	vec3 scale = {1.0f, 1.0f, 1.0f};
	vec4 color = {0.0f, 1.0f, 0.0f, 1.0f};
};

struct CudaGlMappings{
	std::vector<CUgraphicsResource> resources;
	CUgraphicsResource resource;
	CUsurfObject surface;
	CUtexObject texture;
	
	
	bool isMapped = true;

	~CudaGlMappings(){
		if(isMapped){
			unmap();
		}
	}

	void unmap(){
		cuSurfObjectDestroy(surface);
		cuTexObjectDestroy (texture);
		cuGraphicsUnmapResources(resources.size(), resources.data(), ((CUstream)CU_STREAM_DEFAULT));
		// cuGraphicsUnregisterResource(resource);

		isMapped = false;
	};
};

struct SplatEditor{
	
	inline static SplatEditor* instance;

	Scene scene;
	shared_ptr<SNSplats> tempSplats = nullptr;
	shared_ptr<SNTriangles> sn_vr_editing = nullptr;
	shared_ptr<SNTriangles> sn_dbgsphere = nullptr;
	shared_ptr<SNSplats> sn_brushsphere = nullptr;

	vector<SceneNode*> scheduledForRemoval;

	CudaModularProgram* prog_gaussians_rendering = nullptr;
	CudaModularProgram* prog_gaussians_editing = nullptr;
	CudaModularProgram* prog_points = nullptr;
	CudaModularProgram* prog_triangles = nullptr;
	CudaModularProgram* prog_lines = nullptr;
	CudaModularProgram* prog_helpers = nullptr;

	OpenVRHelper* ovr = nullptr;
	View viewLeft;
	View viewRight;
	shared_ptr<Framebuffer> fbGuiVr;
	shared_ptr<Framebuffer> fbGuiVr_assets;
	vec2 vrGuiResolution = {1024, 1024};
	MouseEvents mouse_prev;
	shared_ptr<ImguiNode> imn_brushes = nullptr;
	shared_ptr<ImguiNode> imn_assets = nullptr;
	shared_ptr<ImguiNode> imn_layers = nullptr;
	shared_ptr<ImguiNode> imn_painting = nullptr;


	vector<shared_ptr<Action>> history;
	int history_offset = 0;
	
	vector<GuiVrPage> vrPages = {
		{
			.label = "Brushes",
			.width = 440,
			.height = 700,
		},{
			.label = "Assets",
			.width = 440,
			.height = 700,
		},{
			.label = "Layers",
			.width = 440,
			.height = 700,
		}
	};
	int currentVrPage = 0;

	CUstream stream_upload;
	CUstream mainstream;
	CUstream sidestream;

	CUevent event_mainstream;
	CUevent event_edl_applied;
	CUevent ev_reset_stagecounters;

	// CUdeviceptr cptr_framebuffer;
	shared_ptr<CudaVirtualMemory> virt_framebuffer = CURuntime::allocVirtual("framebuffer");
	CUdeviceptr cptr_keys;
	CUdeviceptr cptr_lines = 0;
	CUdeviceptr cptr_numLines = 0;
	CUdeviceptr cptr_uniforms;

	DeviceState deviceState;
	void* h_state_pinned = nullptr;
	void* h_tilecounter = nullptr;
	CUdeviceptr cptr_state;

	Line* h_lines = nullptr;
	uint32_t* h_numLines = nullptr;
	shared_ptr<CudaVirtualMemory> virt_lines_host = CURuntime::allocVirtual("lines_host");
	CUdeviceptr cptr_numLines_host = 0;

	vector<TriangleQueueItem> triangleQueue;

	AssetLibrary assetLibrary;

	// This is prepared at the start of the frame and provides most properties for CUDA kernels
	CommonLaunchArgs launchArgs;

	shared_ptr<SNTriangles> sn_box = nullptr;

	ImGuiContext* imguicontext_desktop = nullptr;
	ImGuiContext* imguicontext_vr = nullptr;

	struct{
		float splatSize                  = 1.0f;
		int splatAppearance              = SPLATAPPEARANCE_GAUSSIAN;
		bool sort                        = true;
		bool disableCUDA                 = false;
		bool showSolid                   = false;
		bool showTiles                   = false;
		bool showRing                    = false;
		bool showHeatmap                 = false;
		bool showAxes                    = true;
		bool showGrid                    = true;
		bool makePoints                  = false;
		bool showBoundingBoxes           = false;
		bool frontToBack                 = true;
		bool enableOpenglRendering       = false;
		bool renderCooperative           = false;
		bool enableEDL                   = true;
		bool showSplatletBoxes           = false;
		float splatletBoxSizeThreshold   = 16.0f;
		bool showDirtySplatletBoxes      = false;
		int rendermode                   = RENDERMODE_COLOR;
		bool enableStereoFramebufferTest = false;
		bool enableSplatCulling          = false;
		bool disableFrustumCulling       = false;
		bool cullSmallSplats             = true;
		bool requestDebugDump            = false;
		bool requestFramebufferDump      = false;
		bool enableOverlapped            = true;
		int splatRenderer                = SPLATRENDERER_3DGS;
		int intersectionMode             = INTERSECTION_APPROXIMATE;
		int brushColorMode               = BRUSHCOLORMODE_NORMAL;

		Brush brush;
		RectSelect rectselect;
		ColorCorrection colorCorrection;
		bool hideGUI                     = false;
		float vr_brushSize               = 0.1f;

		// - We need to be able to disable shortcuts while typing, for example.
		// - Actions at the end of a frame (e.g. processing an opened context menu) may want to disable shortcuts for as long as context menu is open.
		// - Such actions simply disable shortcuts for "two" frames. If the action is processed close to the end of the frame, -1 is subtracted right away, but the other preserves during the next frame. 
		int shortcutsDisabledForXFrames = 0;

		bool renderWarpwise = false;

		bool showDevStuff = false;
		bool showEditing = true;
		bool showKernelInfos = false;
		bool showMemoryInfos = false;
		bool showTimingInfos = false;
		bool showStats = false;
		bool showFileSaveDialog = false;
		bool showGettingStarted = false;
		bool showColorCorrection = false;
		bool showToolbar = true;
		bool openContextMenu = false;

		bool showInset = false;
		float dbg_factor = 1.0f;
		bool renderSoA = false;
		bool renderBandwidth = false;
		bool renderFragIntersections = false;

	} settings;

	struct {
		int hoveredObjectIndex = -1;
		int doubleClickedObjectIndex = -1;
		// int action = ACTION_NONE;
		// shared_ptr<SceneNode> placingItem = nullptr;
		int dbg_method = 0;

		shared_ptr<InputAction> currentAction = nullptr;
	} state;

	struct {
		bool menu_intersects = false;
		vec3 menu_intersection = vec3{0.0f, 0.0f, 0.0f};
	} state_vr; 

	struct ViewmodeDesktop{

	} viewmodeDesktop;

	struct ViewmodeDesktopVR{
		// Controllers should be relative to desktop monitor.
		// This transforms controllers into desktop view space.
		glm::mat4 m_controllers;

		// The right controllers neutral pose.
		// Use to callibrate from physical space to monitor/view space.
		glm::mat4 m_controller_neutral_left = glm::mat4(1.0);
		glm::mat4 m_controller_neutral_right = glm::mat4(1.0);
		glm::mat4 m_controller_neutral = glm::mat4(1.0);
	} viewmodeDesktopVr;

	struct ViewmodeImmersiveVR{
		// - Special world matrix for immersive VR.
		// - This is because in VR, we may want to transform the object
		//   as if it was a miniature toy model.
		// - Composition is: proj * view * world_vr * world
		glm::mat4 world_vr; 
	} viewmodeImmersiveVR;

	ImFont* font_default = nullptr;
	ImFont* font_big = nullptr;
	ImFont* font_vr_title = nullptr;
	ImFont* font_vr_text = nullptr;
	ImFont* font_vr_smalltext = nullptr;

	static void setup();

	void imguiStyleVR();

	CommonLaunchArgs getCommonLaunchArgs();

	// Some of these functions mainly exist because their functionality is used twice (e.g. via GUI & shortcut),
	// and we want to ensure that both approaches end up doing exactly the same thing.
	Box3 getSelectionAABB();
	void updateBoundingBox(SNSplats* node, bool onlySelected = false);
	void updateBoundingBox(SNTriangles* node);
	void updateBoundingBox(PointData& model);
	void insertNodeToNode(shared_ptr<SceneNode> node, shared_ptr<SceneNode> layer, bool onlySelected = false);
	bool merge(shared_ptr<SceneNode> snsource, shared_ptr<SceneNode> sntarget); 
	void applyTransformation(GaussianData& model, mat4 transformation, bool onlySelected = false);
	void apply(GaussianData& model, ColorCorrection value);
	// void setSelected(GaussianData& model);
	void setSelectedNode(SceneNode* node);
	void transformAllSelected(mat4 transform);
	void selectAll();
	void deselectAll();
	void invertSelection();
	void deleteSelection();
	void createOrUpdateThumbnail(SceneNode* node);
	void uploadSplats(SNSplats* node);
	void drawGUI();
	void resetEditor();
	void unloadTempSplats();
	void loadTempSplats(string path);
	void inputHandling();
	void inputHandlingDesktop();
	void inputHandlingVR();
	void setDesktopMode();
	void setDesktopVrMode();
	void setImmersiveVrMode();
	shared_ptr<SNSplats> clone(SNSplats* source);
	void sortSplatsDevice(SNSplats* node, bool putDeletedLast = false);
	void drawSphere(_DrawSphereArgs args);
	void drawLine(vec3 start, vec3 end, uint32_t color = 0xffff00ff);
	void drawBox(Box3 box, uint32_t color = 0xffff00ff);
	Uniforms getUniforms();
	void initCudaProgram();
	shared_ptr<SNTriangles> ovrToNode(string name, RenderModel_t* model, RenderModel_TextureMap_t* texture);
	shared_ptr<SceneNode> getSelectedNode();
	void temporarilyDisableShortcuts(int numFrames = 2);
	bool areShortcutsDisabled();
	void filter(SNSplats* source, SNSplats* target, FilterRules rules);
	shared_ptr<SNSplats> filterToNewLayer(FilterRules rules);
	vector<shared_ptr<SceneNode>> getLayers();
	// Erases given node at the very end of the frame. Was needed because there were issues when
	// directly deleting a node while drawing that node's gui.
	void scheduleRemoval(SceneNode* node);
	void applyDeletion();
	void revertDeletion();
	int32_t getNumSelectedSplats();
	int32_t getNumDeletedSplats();

	struct LambdaAction{
		function<void()> undo;
		function<void()> redo;
	};

	void addAction(LambdaAction action);
	void addAction(shared_ptr<Action> action);
	void undo();
	void redo();
	void clearHistory();

	// undoable functions
	void selectAll_undoable();
	void selectAllInNode_undoable(shared_ptr<SNSplats> node);
	void deselectAll_undoable();
	void invertSelection_undoable();
	void deleteSelection_undoable();
	void deleteNode_undoable(shared_ptr<SceneNode> node);
	shared_ptr<SNSplats> filterToNewLayer_undoable(FilterRules rules);
	shared_ptr<SNSplats> duplicateLayer_undoable(shared_ptr<SNSplats> node);
	// shared_ptr<SNSplats> extractLayer_undoable(shared_ptr<SNSplats> node);
	void merge_undoable(shared_ptr<SceneNode> snsource, shared_ptr<SceneNode> sntarget); 

	// GUI
	void setAction(shared_ptr<InputAction> action);
	void makeSettings();
	void makePerf();
	void makeMenubar();
	void makeLayers();
	void makeStats();
	void makeTODOs();
	void makeToolbar();
	// void makeGuiVr();
	void makeEditingGUI();
	void makeDevGUI();
	void makeDebugInfo();
	void makeAssetGUI();
	void makeColorCorrectionGui();
	void makeSaveFileGUI();
	void makeGettingStarted();

	// MISC
	CudaGlMappings mapCudaGl(shared_ptr<GLTexture> source);
	
	// UPDATE & DRAW 
	void update();
	void render();
	// void draw(Scene* scene, View view, RenderTarget& target);
	void draw(Scene* scene, vector<RenderTarget> targets);
	void postRenderStuff();

};
