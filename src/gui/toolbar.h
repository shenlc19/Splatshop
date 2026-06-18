
#include "actions/BrushSelectAction.h"
#include "actions/SphereSelectAction.h"
#include "actions/SpherePaintAction.h"
#include "actions/RectSelectAction.h"
#include "actions/ThreePointAlignAction.h"
#include "actions/GizmoAction.h"


// see https://github.com/ocornut/imgui/issues/2648
void SplatEditor::makeToolbar(){

	auto editor = SplatEditor::instance;
	auto& settings = editor->settings;
	auto& state = editor->state;
	auto& scene = editor->scene;
	auto ovr = editor->ovr;
	auto drawlist = ImGui::GetForegroundDrawList();

	if(!settings.showToolbar) return;

	auto gltex_symbols_32x32 = Runtime::gltex_symbols_32x32;
	auto gltex_symbols_32x32_width = Runtime::gltex_symbols_32x32_width;
	auto gltex_symbols_32x32_height = Runtime::gltex_symbols_32x32_height;

	ImVec2 toolbar_start = ImVec2(0, 19);
	// ImVec2 toolbar_start = ImVec2(0, 0);
	// ImVec2 toolbar_end;

	ImGui::SetNextWindowPos(toolbar_start);
	ImVec2 requested_size = ImVec2(GLRenderer::width, 0.0f);
	ImGui::SetNextWindowSize(requested_size);

	struct Section{
		float x_start;
		float x_end;
		string label;
	};

	vector<Section> sections;

	auto startSection = [&](string label){
		ImGui::SameLine();
		ImGui::BeginGroup();

		Section section;
		section.x_start = ImGui::GetCursorPosX();
		section.label = label;

		sections.push_back(section);
	};

	auto endSection = [&](){
		ImGui::EndGroup();
		ImGui::SameLine();
		float x = ImGui::GetCursorPosX();
		ImU32 color = IM_COL32(255, 255, 255, 75);
		drawlist->AddLine({x - 4.0f, 51.0f - 32.0f}, {x - 4.0f, 134.0f - 32.0f + 14.0f}, color, 1.0f);

		Section& section = sections[sections.size() - 1];
		section.x_end = x;
	};

	auto startHighlightButtonIf = [&](bool condition){
		ImGuiStyle* style = &ImGui::GetStyle();
		ImVec4* colors = style->Colors;
		ImVec4 color = colors[ImGuiCol_Button];

		if(condition){
			color = ImVec4(0.06f, 0.53f, 0.98f, 1.00f);
		}
		ImGui::PushStyleColor(ImGuiCol_Button, color);
	};

	auto endHighlightButtonIf = [&](){
		ImGui::PopStyleColor(1);
	};
	
	uint32_t flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar;
	ImGui::Begin("Toolbar", nullptr, flags);

	
	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.0f, 0.0f, 0.0f, 0.0f});
	{
		ImTextureID handle = (void*)(intptr_t)gltex_symbols_32x32;
		// ImTextureID my_tex_id = (void*)(intptr_t)gltex_symbols;
		// ImVec2 size = ImVec2(16.0f, 16.0f);
		ImVec2 buttonSize = ImVec2(32.0f, 32.0f);
		float symbolSize = 32.0f;
		ImVec4 bg_col = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
		ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
		int padding = 8;

		struct Box2{
			ImVec2 start;
			ImVec2 end;
		};

		auto getIconBounds = [&](int x, int y) -> Box2 {
			ImVec2 uv_0 = ImVec2(float(x + 0.0f) * symbolSize / gltex_symbols_32x32_width, float(y + 0.0f) * symbolSize / gltex_symbols_32x32_height);
			ImVec2 uv_1 = ImVec2(float(x + 1.0f) * symbolSize / gltex_symbols_32x32_width, float(y + 1.0f) * symbolSize / gltex_symbols_32x32_height);

			Box2 box = {uv_0, uv_1};

			return box;
		};

		{
			startSection("Transform");

			ImVec2 buttonSize = ImVec2(24.0f, 24.0f);
			
			float cx = ImGui::GetCursorPosX();
			float cy = ImGui::GetCursorPosY();
			ImGui::SetCursorPosY(cy - 6);
			cy = ImGui::GetCursorPosY();

			{ // TRANSLATION
				Box2 box = getIconBounds(6, 0);

				if(ImGui::ImageButton("tool_translate", handle, buttonSize, box.start, box.end, bg_col, tint_col)){
					// state.activeGizmo = GIZMO_TRANSLATE;
					shared_ptr<GizmoAction> action = make_shared<GizmoAction>();
					action->mode = GizmoAction::GIZMO_TRANSLATE;
					setAction(action);
				}
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
					ImGui::SetTooltip("Translate selected splats or layer. (t)");
				}
			}

			{ // ROTATION
				Box2 box = getIconBounds(4, 0);

				ImGui::SameLine();
				ImGui::SetCursorPosX(cx + 1 * (buttonSize.x + padding));
				if(ImGui::ImageButton("tool_rotate", handle, buttonSize, box.start, box.end, bg_col, tint_col)){
					// state.activeGizmo = GIZMO_ROTATE;
					shared_ptr<GizmoAction> action = make_shared<GizmoAction>();
					action->mode = GizmoAction::GIZMO_ROTATE;
					setAction(action);
				}
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
					ImGui::SetTooltip("Rotate selected splats or layer. (r)");
				}
			}

			{ // SCALE
				Box2 box = getIconBounds(5, 0);

				ImGui::SameLine();
				ImGui::SetCursorPosX(cx + 2 * (buttonSize.x + padding));
				if(ImGui::ImageButton("tool_scale", handle, buttonSize, box.start, box.end, bg_col, tint_col)){
					// state.activeGizmo = GIZMO_SCALE;
					shared_ptr<GizmoAction> action = make_shared<GizmoAction>();
					action->mode = GizmoAction::GIZMO_SCALE;
					setAction(action);
				}
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
					ImGui::SetTooltip("Scale selected splats or layer. (s)");
				}
			}

			ImGui::SetCursorPosY(cy + buttonSize.y + 4);

			{ // 3-POINT-REALIGN
				Box2 box = getIconBounds(4, 2);

				// ImGui::SameLine();
				ImGui::SetCursorPosX(cx + 0 * (buttonSize.x + padding));
				if(ImGui::ImageButton("tool_3pointalign", handle, buttonSize, box.start, box.end, bg_col, tint_col)){
					shared_ptr<ThreePointAlignAction> action = make_shared<ThreePointAlignAction>();
					setAction(action);
				}
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
					ImGui::SetTooltip("3-Point-Align: \n- Place three points to define the ground plane of the currently selected layer. The layer will then be rotated to face up. \n- Place in counter-clockwise order or it will be upside-down.");
				}
			}

			{ // UPSIDE-DOWN
				Box2 box = getIconBounds(6, 2);

				ImGui::SameLine();
				ImGui::SetCursorPosX(cx + 1 * (buttonSize.x + padding));
				if(ImGui::ImageButton("tool_upsidedown", handle, buttonSize, box.start, box.end, bg_col, tint_col)){

					shared_ptr<SceneNode> node = getSelectedNode();

					if(node && node->visible && !node->locked){
						mat4 transform = mat4{1.0f};
						transform[2][2] = -1;

						mat4 before = node->transform;
						mat4 after = transform * node->transform;

						node->transform = after;

						editor->addAction({
							.undo = [=](){
								node->transform = before;
							},
							.redo = [=](){
								node->transform = after;
							}
						});
					}
				}
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
					ImGui::SetTooltip("Flip the z axis of the selected layer.");
				}
			}

			{ // FLIP YZ
				Box2 box = getIconBounds(7, 2);

				ImGui::SameLine();
				ImGui::SetCursorPosX(cx + 2 * (buttonSize.x + padding));
				if(ImGui::ImageButton("tool_flipyz", handle, buttonSize, box.start, box.end, bg_col, tint_col)){

					shared_ptr<SceneNode> node = getSelectedNode();

					if(node && node->visible && !node->locked){
						mat4 transform = mat4{
							1.0f, 0.0f, 0.0f, 0.0f,
							0.0f, 0.0f, -1.0f, 0.0f,
							0.0f, 1.0f, 0.0f, 0.0f,
							0.0f, 0.0f, 0.0f, 1.0f, 
						};

						mat4 before = node->transform;
						mat4 after = transform * node->transform;

						node->transform = after;

						editor->addAction({
							.undo = [=](){
								node->transform = before;
							},
							.redo = [=](){
								node->transform = after;
							}
						});
					}
				}
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
					ImGui::SetTooltip("Flip the z and y axes of the selected layer.");
				}
			}

			endSection();
		}

		{ // SELECT 
			startSection("Select");

			ImVec2 buttonSize = ImVec2(24.0f, 24.0f);
			int padding = 8;

			float cx = ImGui::GetCursorPosX();
			float cy = ImGui::GetCursorPosY();
			ImGui::SetCursorPosY(cy - 6);
			cy = ImGui::GetCursorPosY();

			{ // SELECT
				Box2 box = getIconBounds(7, 0);

				
				if(ImGui::ImageButton("tool_brush_select", handle, buttonSize, box.start, box.end, bg_col, tint_col)){
					shared_ptr<BrushSelectAction> action = make_shared<BrushSelectAction>();
					setAction(action);
					settings.brush.mode = BRUSHMODE::SELECT;
				}
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
					ImGui::SetTooltip("Selection Tool (Shortcut: 1)");
				}
			}

			{ // DELETE
				Box2 box = getIconBounds(8, 0);

				ImGui::SameLine();
				ImGui::SetCursorPosX(cx + 1 * (buttonSize.x + padding));
				if(ImGui::ImageButton("tool_brush_delete", handle, buttonSize, box.start, box.end, bg_col, tint_col)){
					shared_ptr<BrushSelectAction> action = make_shared<BrushSelectAction>();
					setAction(action);
					settings.brush.mode = BRUSHMODE::ERASE;
				}
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
					ImGui::SetTooltip("Deletion Tool. Flags brushed splats as deleted. Can be reverted with the restoration tool. (Shortcut: 2)");
				}
			}

			{ // SELECT - SPHERE
				Box2 box = getIconBounds(8, 2);

				ImGui::SameLine();
				ImGui::SetCursorPosX(cx + 2 * (buttonSize.x + padding));
				if(ImGui::ImageButton("tool_sphere_select", handle, buttonSize, box.start, box.end, bg_col, tint_col)){
					shared_ptr<SphereSelectAction> action = make_shared<SphereSelectAction>();
					editor->setAction(action);
					settings.brush.mode = BRUSHMODE::SELECT;
				}
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
					ImGui::SetTooltip("Spherical Selection Tool");
				}
			}

			{ // SELECT - RECTANGLE
				Box2 box = getIconBounds(10, 0);

				ImGui::SameLine();
				ImGui::SetCursorPosX(cx + 3 * (buttonSize.x + padding));
				if(ImGui::ImageButton("tool_rect_select", handle, buttonSize, box.start, box.end, bg_col, tint_col)){
					// settings.brush.mode = BRUSHMODE::SELECT;
					settings.rectselect.active = true;
					settings.rectselect.start = {0.0f, 0.0f};
					settings.rectselect.end = {0.0f, 0.0f};
					settings.rectselect.startpos_specified = false;
					//settings.rectselect.mode = 0;

					shared_ptr<RectSelectAction> action = make_shared<RectSelectAction>();
					editor->setAction(action);
				}
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
					ImGui::SetTooltip("Selection Tool (Shortcut: 4)\nAlt: Deselect\nShift: Add to selection.");
				}
			}

			// { // SELECT - MAGIC WAND
			// 	Box2 box = getIconBounds(5, 2);

			// 	ImGui::SameLine();
			// 	ImGui::SetCursorPosX(cx + 4 * (buttonSize.x + padding));
			// 	if(ImGui::ImageButton("tool_magic_wand", handle, buttonSize, box.start, box.end, bg_col, tint_col)){

			// 		shared_ptr<MagicWandAction> action = make_shared<MagicWandAction>();
			// 		editor->setAction(action);
			// 	}
			// 	if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
			// 		ImGui::SetTooltip("Magic Wand: Select splats with similar colors.");
			// 	}
			// }

			ImGui::SetCursorPosY(cy + buttonSize.y + 4);

			{ // SELECT ALL
				// ImGui::SameLine();
				ImGui::SetCursorPosX(cx + 0 * (buttonSize.x + padding));
				Box2 box = getIconBounds(0, 1);
				if (ImGui::ImageButton("Select All#IB", handle, buttonSize, box.start, box.end, bg_col, tint_col)) {
					editor->selectAll_undoable();
				}
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
					ImGui::SetTooltip("Select All Splats");
				}
			}

			{ // DESELECT ALL
				ImGui::SameLine();
				ImGui::SetCursorPosX(cx + 1 * (buttonSize.x + padding));
				Box2 box = getIconBounds(1, 1);
				if (ImGui::ImageButton("Deselect All#IB", handle, buttonSize, box.start, box.end, bg_col, tint_col)) {
					editor->deselectAll_undoable();
				}
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
					ImGui::SetTooltip("Deselect All Splats");
				}
			}

			{ // INVERT SELECTION
				ImGui::SameLine();
				ImGui::SetCursorPosX(cx + 2 * (buttonSize.x + padding));
				Box2 box = getIconBounds(2, 1);
				if (ImGui::ImageButton("Invert Selection#IB", handle, buttonSize, box.start, box.end, bg_col, tint_col)) {
					editor->invertSelection_undoable();
				}
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
					ImGui::SetTooltip("Invert Selection");
				}
			}

			{ // DELETE SELECTION
				ImGui::SameLine();
				ImGui::SetCursorPosX(cx + 3 * (buttonSize.x + padding));
				Box2 box = getIconBounds(4, 1);
				if (ImGui::ImageButton("Delete Selection#IB", handle, buttonSize, box.start, box.end, bg_col, tint_col)) {
					editor->deleteSelection_undoable();
				}
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
					ImGui::SetTooltip("Delete Selection - Flags splats as deleted. (Shortcut: Del)");
				}
			}

			endSection();
		}

		{ // PAINTING
			startSection("Painting");

			// ImVec2 buttonSize = ImVec2(24.0f, 24.0f);
			int padding = 8;

			float cx = ImGui::GetCursorPosX();
			float cy = ImGui::GetCursorPosY();
			ImGui::SetCursorPosY(cy - 6);
			cy = ImGui::GetCursorPosY();

			{ // PAINT - SPHERE
				Box2 box = getIconBounds(8, 2);

				ImGui::SameLine();
				ImGui::SetCursorPosX(cx + 0 * (buttonSize.x + padding));
				if(ImGui::ImageButton("tool_sphere_paint", handle, buttonSize, box.start, box.end, bg_col, tint_col)){
					shared_ptr<SpherePaintAction> action = make_shared<SpherePaintAction>();
					editor->setAction(action);
					settings.brush.mode = BRUSHMODE::SELECT;
				}
				// if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
				// 	ImGui::SetTooltip("Selection Tool (Shortcut: 4)\nAlt: Deselect\nShift: Add to selection.");
				// }
			}
			
			ImGui::SameLine();
			ImGui::Text("       ");

			endSection();
		}

		{
			startSection("Viewpoint");

			float cx = ImGui::GetCursorPosX();
			float cy = ImGui::GetCursorPosY();

			{ // Zoom To
				ImGui::SameLine();
				Box2 box = getIconBounds(3, 2);
				ImGui::SetCursorPosX(cx + 0 * (buttonSize.x + padding));
				if (ImGui::ImageButton("Zoom#IB", handle, buttonSize, box.start, box.end, bg_col, tint_col)) {
					Box3 aabb = editor->getSelectionAABB();

					float length = glm::length(aabb.max - aabb.min);
					vec3 center = (aabb.min + aabb.max) * 0.5f;

					// Runtime::controls->yaw    = 0.0f;
					// Runtime::controls->pitch  = -3.1415f * 0.5f;
					Runtime::controls->radius = 0.7f * length;
					Runtime::controls->target = center;
				}
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
					ImGui::SetTooltip("Zoom to selected splats or layer.");
				}
			}

			{ // TOP
				ImGui::SameLine();
				Box2 box = getIconBounds(0, 2);
				ImGui::SetCursorPosX(cx + 1 * (buttonSize.x + padding));
				if (ImGui::ImageButton("Top#IB", handle, buttonSize, box.start, box.end, bg_col, tint_col)) {
					Box3 aabb = editor->getSelectionAABB();

					float length = glm::length(aabb.max - aabb.min);
					vec3 center = (aabb.min + aabb.max) * 0.5f;

					Runtime::controls->yaw    = 0.0f;
					Runtime::controls->pitch  = -3.1415f * 0.5f;
					Runtime::controls->radius = 0.7f * length;
					Runtime::controls->target = center;
				}
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
					ImGui::SetTooltip("Top View of selected splats or layer.");
				}
			}

			{ // SIDE
				ImGui::SameLine();
				Box2 box = getIconBounds(1, 2);
				ImGui::SetCursorPosX(cx + 2 * (buttonSize.x + padding));
				if (ImGui::ImageButton("Side#IB", handle, buttonSize, box.start, box.end, bg_col, tint_col)) {
					Box3 aabb = editor->getSelectionAABB();

					float length = glm::length(aabb.max - aabb.min);
					vec3 center = (aabb.min + aabb.max) * 0.5f;

					Runtime::controls->yaw    = -3.1415f * 0.5f;
					Runtime::controls->pitch  = 0.0f;
					Runtime::controls->radius = 0.7f * length;
					Runtime::controls->target = center;
				}
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
					ImGui::SetTooltip("Side View of selected splats or layer.");
				}
			}

			{ // FRONT
				ImGui::SameLine();
				Box2 box = getIconBounds(2, 2);
				ImGui::SetCursorPosX(cx + 3 * (buttonSize.x + padding));
				if (ImGui::ImageButton("Front#IB", handle, buttonSize, box.start, box.end, bg_col, tint_col)) {
					Box3 aabb = editor->getSelectionAABB();

					float length = glm::length(aabb.max - aabb.min);
					vec3 center = (aabb.min + aabb.max) * 0.5f;

					Runtime::controls->yaw    = 0.0f;
					Runtime::controls->pitch  = 0.0f;
					Runtime::controls->radius = 0.7f * length;
					Runtime::controls->target = center;
				}
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
					ImGui::SetTooltip("Front View of selected splats or layer.");
				}
			}

			endSection();
		}

		{
			startSection("Window");

			float cx = ImGui::GetCursorPosX();
			float cy = ImGui::GetCursorPosY();

			{ // 720p
				ImGui::SameLine();
				Box2 box = getIconBounds(8, 1);
				ImGui::SetCursorPosX(cx + 0 * (buttonSize.x + padding));
				if (ImGui::ImageButton("720p#IB", handle, buttonSize, box.start, box.end, bg_col, tint_col)) {
					glfwSetWindowSize(GLRenderer::window, 1280, 720);
				}
			}

			{ // 1080p
				ImGui::SameLine();
				Box2 box = getIconBounds(9, 1);
				ImGui::SetCursorPosX(cx + 1 * (buttonSize.x + padding));
				if (ImGui::ImageButton("1080p#IB", handle, buttonSize, box.start, box.end, bg_col, tint_col)) {
					glfwSetWindowSize(GLRenderer::window, 1920, 1080);
				}
			}

			{ // 4k UHD
				ImGui::SameLine();
				Box2 box = getIconBounds(10, 1);
				ImGui::SetCursorPosX(cx + 2 * (buttonSize.x + padding));
				if (ImGui::ImageButton("4kuhd#IB", handle, buttonSize, box.start, box.end, bg_col, tint_col)) {
					glfwSetWindowSize(GLRenderer::window, 3840, 2160);
				}
			}

			endSection();
		}

		{
			startSection("Misc");

			float cx = ImGui::GetCursorPosX();
			float cy = ImGui::GetCursorPosY();

			{ // TOGGLE BOUNDING BOXES
				ImGui::SameLine();
				Box2 box = getIconBounds(13, 0);
				ImGui::SetCursorPosX(cx + 0 * (buttonSize.x + padding));
				if (ImGui::ImageButton("Toggle Bounding Boxes#IB", handle, buttonSize, box.start, box.end, bg_col, tint_col)) {
					settings.showBoundingBoxes = !settings.showBoundingBoxes;
				}
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
					ImGui::SetTooltip("Toggle Bounding Boxes.");
				}
			}

			{ // TOGGLE VR
				ImGui::SameLine();
				Box2 box = getIconBounds(7, 1);
				ImGui::SetCursorPosX(cx + 1 * (buttonSize.x + padding));
				if (ImGui::ImageButton("Toggle VR#IB", handle, buttonSize, box.start, box.end, bg_col, tint_col)) {

					if(ovr->isActive()){
						editor->setDesktopMode();
					}else{
						editor->setImmersiveVrMode();
					}
				}
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
					ImGui::SetTooltip("Toggle VR.");
				}
			}

			{ // Show Axes
				ImGui::SameLine();
				Box2 box = getIconBounds(14, 0);
				ImGui::SetCursorPosX(cx + 2 * (buttonSize.x + padding));
				if (ImGui::ImageButton("ShowAxes#IB", handle, buttonSize, box.start, box.end, bg_col, tint_col)) {
					settings.showAxes = !settings.showAxes;
				}
			}

			{ // Show Grid
				ImGui::SameLine();
				Box2 box = getIconBounds(15, 0);
				ImGui::SetCursorPosX(cx + 3 * (buttonSize.x + padding));
				if (ImGui::ImageButton("ShowGrid#IB", handle, buttonSize, box.start, box.end, bg_col, tint_col)) {
					settings.showGrid = !settings.showGrid;
				}
			}

			{ // Undo
				ImGui::SameLine();
				Box2 box = getIconBounds(9, 2);
				ImGui::SetCursorPosX(cx + 4 * (buttonSize.x + padding));
				if (ImGui::ImageButton("Undo#IB", handle, buttonSize, box.start, box.end, bg_col, tint_col)) {
					editor->undo();
				}
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
					ImGui::SetTooltip("Undo last action.");
				}
			}
			
			{ // Redo
				ImGui::SameLine();
				Box2 box = getIconBounds(10, 2);
				ImGui::SetCursorPosX(cx + 5 * (buttonSize.x + padding));
				if (ImGui::ImageButton("Redo#IB", handle, buttonSize, box.start, box.end, bg_col, tint_col)) {
					editor->redo();
				}
				if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
					ImGui::SetTooltip("Redo");
				}
			}

			endSection();
		}

		{ // APPEARANCE
			startSection("Appearance");

			float cx = ImGui::GetCursorPosX();
			float cy = ImGui::GetCursorPosY();

			ImGui::Checkbox("Solid", &editor->settings.showSolid);

			ImGui::SameLine();
			ImGui::Checkbox("Ring", &settings.showRing);

			ImGui::SameLine();
			ImGui::RadioButton("Color", &editor->settings.rendermode, RENDERMODE_COLOR); ImGui::SameLine();
			ImGui::RadioButton("Depth", &editor->settings.rendermode, RENDERMODE_DEPTH); ImGui::SameLine();
			ImGui::RadioButton("Heat", &editor->settings.rendermode, RENDERMODE_HEATMAP); ImGui::SameLine();

			ImGui::SetCursorPosX(cx);
			ImGui::SetCursorPosY(cy + 24);
			ImGui::SetNextItemWidth(100.0f);
			ImGui::SliderFloat("splat size", &editor->settings.splatSize, 0.0f, 2.0f);

			// ImGui::SameLine();
			ImGui::Text("Splat Renderer: "); ImGui::SameLine();
			ImGui::RadioButton("3DGS", &editor->settings.splatRenderer, SPLATRENDERER_3DGS); ImGui::SameLine(); 
			ImGui::RadioButton("Persp.Correct", &editor->settings.splatRenderer, SPLATRENDERER_PERSPECTIVE_CORRECT); ImGui::SameLine();
			if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
				string str = "Render splats using 'Efficient Perspective-Correct 3D Gaussian Splatting Using Hybrid Transparency'\n";
				str += "To be used with correspondingly trained splat models, not suitable with standard 3DGS-trained models. ";
				ImGui::SetTooltip(str.c_str());
			}
			

			endSection();
		}

		{ // DEV
			startSection("Dev");

			float cx = ImGui::GetCursorPosX();
			float cy = ImGui::GetCursorPosY();

			// ImGui::SameLine();
			// if(ImGui::Button("downsample")){
			// 	editor->downsample();
			// }

			// ImGui::SameLine();
			// if(ImGui::Button("toggle")){
			// 	editor->scene.process<SceneNode>([](SceneNode* node){
			// 		if(node->hidden) return;

			// 		node->visible = !node->visible;
			// 	});
			// }

			// ImGui::RadioButton("Original", &state.dbg_method, 0); ImGui::SameLine();
			// ImGui::RadioButton("TileSubsets", &state.dbg_method, 1); ImGui::SameLine();

			// ImGui::SameLine(); ImGui::Text(" | ");
			// bool frustumCulling = !settings.disableFrustumCulling;
			// ImGui::SameLine();
			// ImGui::Checkbox("Frustum Culling", &frustumCulling);
			// settings.disableFrustumCulling = !frustumCulling;

			ImGui::SameLine();
			ImGui::Checkbox("Cull Small Splats", &settings.cullSmallSplats);

			// ImGui::SameLine(); ImGui::Text("          ");

			ImGui::SetCursorPosX(cx);
			ImGui::SetCursorPosY(cy + 24);

			// ImGui::SameLine();
			ImGui::Checkbox("Stereo Test", &editor->settings.enableStereoFramebufferTest);
			ImGui::SameLine();
			// ImGui::Checkbox("Splat Culling", &editor->settings.enableSplatCulling);

			ImGui::SameLine();
			ImGui::Checkbox("Overlapped", &editor->settings.enableOverlapped);
			ImGui::SameLine();


			ImGui::SameLine();
			if(ImGui::Button("Dump")){
				editor->settings.requestDebugDump = true;
			}

			ImGui::SameLine();
			if(ImGui::Button("Dump FB")){
				editor->settings.requestFramebufferDump = true;
				GLRenderer::requestFramebufferDump = true;
			}

			ImGui::SameLine();
			ImGui::Checkbox("SoA", &editor->settings.renderSoA);
			ImGui::SameLine();
			ImGui::Checkbox("Bandwidth", &editor->settings.renderBandwidth);


			// ImGui::SameLine();
			// ImGui::Checkbox("FragIntersections", &editor->settings.renderFragIntersections);

			ImGui::SameLine();
			ImGui::Text("Intersection: ");
			ImGui::RadioButton("Approx.##intersection", &editor->settings.intersectionMode, INTERSECTION_APPROXIMATE); ImGui::SameLine();
			ImGui::RadioButton("3DGS##intersection", &editor->settings.intersectionMode, INTERSECTION_3DGS); ImGui::SameLine();
			ImGui::RadioButton("TightBB##intersection", &editor->settings.intersectionMode, INTERSECTION_TIGHTBB); ImGui::SameLine();

			endSection();
		}

		{ // STATS
			startSection("Stats");

			size_t availableMem = 0;
			size_t totalMem = 0;
			cuMemGetInfo(&availableMem, &totalMem);
			float usedMem = totalMem - availableMem;
			float available = double(availableMem) / 1'000'000'000.0;
			float used = double(usedMem) / 1'000'000'000.0;
			float total = double(totalMem) / 1'000'000'000.0;
			float progress = used / total;

			auto cursorPos = ImGui::GetCursorPos();

			static float value = 16.3f;
			float max = 24.0f;
			// ImGui::VSliderFloat("##v", ImVec2(18, 50), &value, 0.0f, max, "%.1f", ImGuiSliderFlags_ReadOnly);

			const char* label = "GPU";
			ImGuiWindow* window = ImGui::GetCurrentWindow();
			ImGuiContext& g = *GImGui;
			const ImGuiStyle& style = g.Style;
			
			ImVec2 widgetSize = { 24, 50 };
			ImVec4 c_frameBg = style.Colors[ImGuiCol_FrameBg];
			ImU32 c_background = IM_COL32(255.0f * c_frameBg.x, 255.0f * c_frameBg.y, 255.0f * c_frameBg.z, 255.0f * c_frameBg.w);
			ImU32 c_foreground = IM_COL32(200, 100, 0, 255);

			ImU32 c_green = IM_COL32(102, 194, 165, 255);
			ImU32 c_orange = IM_COL32(253, 174, 97, 255);
			ImU32 c_red = IM_COL32(213,62,79, 255);

			if(progress < 0.5){
				c_foreground = c_green;
			}else if(progress < 0.75){
				c_foreground = c_orange;
			}else{
				c_foreground = c_red;
			}

			const ImVec2 label_size = ImGui::CalcTextSize(label, NULL, true);
			const ImRect frame_bb(window->DC.CursorPos, window->DC.CursorPos + widgetSize);
			const ImRect bb(frame_bb.Min, frame_bb.Max + ImVec2(label_size.x > 0.0f ? style.ItemInnerSpacing.x + label_size.x : 0.0f, 0.0f));
			ImGui::ItemSize(bb, style.FramePadding.y);

			const ImGuiID id = window->GetID("GPU_memor_usage_widget");
			ImGui::ItemAdd(frame_bb, id);

			float use_y = progress * float(frame_bb.Min.y) + (1.0 - progress) * float(frame_bb.Max.y);

			drawlist->AddRectFilled(frame_bb.Min, frame_bb.Max, c_background);
			drawlist->AddRectFilled({frame_bb.Min.x, use_y}, frame_bb.Max, c_foreground);

			if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){
				float usedGB = double(usedMem) / 1'000'000'000;
				float totalGB = double(totalMem) / 1'000'000'000;
				string msg = format(getSaneLocale(), "GPU Memory | Used: {:.1f} GB, Total: {:.1f} GB", usedGB, totalGB);
				ImGui::SetTooltip(msg.c_str());
			}

			ImGui::SameLine();
			if(ImGui::Button("Clear History & Deleted Splats")){
				editor->clearHistory();
				editor->applyDeletion();
			}
			if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)){

				int64_t historySize = editor->history.size();

				int totalSize = 0;
				for(auto action : editor->history){
					totalSize += action->byteSize();
				} 

				float MB = totalSize / 1'000'000.0f;
				int32_t numDeleted = editor->getNumDeletedSplats();

				auto l = getSaneLocale();
				string msg;
				msg += format(l, "- {} entries comprising at least {:.1f} MB in undo/redo history will be cleared. \n", historySize, MB);
				msg += format(l, "- Also removes {:L} 'deleted' splats for real.", numDeleted);

				ImGui::SetTooltip(msg.c_str());
			}

			endSection();
		}


		ImGui::Text(" ");

	}

	ImGui::PopStyleColor(1);

	ImVec2 wpos = ImGui::GetWindowPos();
	ImVec2 toolbar_end = ImVec2{wpos.x + ImGui::GetWindowWidth(), wpos.y + ImGui::GetWindowHeight()};
	// ImGui::GetForegroundDrawList()->AddRect( start, end, IM_COL32( 255, 255, 0, 255 ) );


	ImGui::End();

	{ // TOOLBAR SECTION LABELS 
		uint32_t flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar;
		ImGui::Begin("Toolbar", nullptr, flags);

		ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{1.0f, 1.0f, 1.0f, 0.5f});



		for(Section section : sections){

			float x_center = (section.x_end + section.x_start) / 2.0f;
			float width = ImGui::CalcTextSize(section.label.c_str()).x;
			float x = x_center - width / 2.0f;

			ImGui::SetCursorPosX(x);
			ImGui::Text(section.label.c_str());
			ImGui::SameLine();

		}

		ImGui::Text(" ");

		ImGui::PopStyleColor(1);

		ImGui::End();
	}

	if(editor->state.currentAction)
	{
		
		ImGui::SetNextWindowPos(ImVec2(toolbar_start.x, toolbar_end.y - 1.0f));
		ImGui::SetNextWindowSize(ImVec2(GLRenderer::width, 0.0f));
		
		uint32_t flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar;
		ImGui::Begin("Toolbar123", nullptr, flags);

		// ImGui::Text("Test");
		
		editor->state.currentAction->makeToolbarSettings();

		ImGui::End();
	}




}
