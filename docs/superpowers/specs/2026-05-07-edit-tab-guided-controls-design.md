# Edit Tab Guided Controls Design

## Summary

Simplify the Edit tab control experience without changing the basic three-panel page layout. The Edit tab will keep its current left control panel, center 2D plan editor, and right 3D result panel. The redesign changes only the UI inside those panels.

The left panel becomes a target-first guided workflow: users first choose what they are editing, then choose the task for that target, then see only the relevant method and value controls. Display-only settings move out of the left panel and into a compact control in the 2D plan editor header. Pending edit actions remain pinned at the bottom of the left panel so the buffered edit workflow stays visible.

## Goals

- Preserve the current three-panel Edit tab layout: left controls, center 2D plan editor, right 3D result.
- Reduce the number of controls visible at one time in the left panel.
- Separate Building Add, Height, and Remove workflows so they no longer appear as one long stacked form.
- Apply the same target-first guided pattern to Building, Tree, and Land cover editing.
- Move basemap, overlay, and visibility controls out of the main editing workflow.
- Keep the existing buffered edit model: edits preview in 2D, can be undone or cleared, and are committed through **Update 3D model**.
- Keep map drawing and selection behavior stable wherever possible.

## Non-Goals

- Do not change the page-level three-panel layout or panel ordering.
- Do not replace the Leaflet 2D plan editor or the 3D result panel.
- Do not redesign other app tabs.
- Do not add a new backend editing subsystem.
- Do not make model regeneration automatic after every edit.
- Do not add a separate pending-edit history table in this redesign.

## Recommended Approach

Use a target-first guided control panel inside the existing Edit tab layout.

The user first chooses the edit target:

- **Building**
- **Tree**
- **Land cover**

The selected target determines the task choices:

- Building: **Add**, **Height**, **Remove**
- Tree: **Add**, **Remove**
- Land cover: **Paint**

The selected task determines the method and fields shown below it. This keeps repeated editing fast while reducing the visual load of the current left panel.

This approach is preferable to a strict wizard because spatial editing often requires switching tools quickly. It is also preferable to simple collapsible sections because it changes the mental model instead of only hiding existing complexity.

## Layout Boundary

The current Edit tab layout is a hard constraint. The page remains a three-column workspace:

- Left panel: edit controls
- Center panel: 2D plan editor
- Right panel: 3D result

Only the UI inside those panels changes. The redesign should not add a new full-width toolbar, move the 2D map, move the 3D result, or change the overall panel proportions beyond normal responsive behavior already present in the app.

## Left Control Panel Design

The left panel keeps the **Edit Model** heading and becomes a guided control surface with three layers.

### 1. Target

A target selector lets users choose **Building**, **Tree**, or **Land cover**. The selector replaces the current mode tabs conceptually, but keeps the same target choices.

Switching target resets the task and method to valid defaults for that target. It also clears transient task UI such as selected building ids for height editing.

Default target state:

- Building defaults to Add / Rectangle.
- Tree defaults to Add / Click.
- Land cover defaults to Paint / Click.

### 2. Task

The task selector shows only the tasks for the selected target.

Building tasks:

- **Add**: create new building footprints.
- **Height**: select buildings and buffer height edits.
- **Remove**: delete buildings by click or area.

Tree tasks:

- **Add**: add trees by click or area.
- **Remove**: remove canopy by click or area.

Land cover tasks:

- **Paint**: paint one cell or an area.

### 3. Tool Details

The tool details area shows only the selected task's method and inputs.

Building Add:

- Method: Rectangle or Polygon.
- Inputs: height and min/base height.

Building Height:

- Method: Click or Area.
- State: selected building count and selected chips or summary when useful.
- Inputs: top height and optional min/base height.
- Action: buffer/apply the height edit to the current selection.

Building Remove:

- Method: Click or Area.
- Short contextual hint for the active method.

Tree Add:

- Method: Click or Area.
- Inputs: top height, trunk/bottom height, diameter, fixed proportion.

Tree Remove:

- Method: Click or Area.

Land Cover Paint:

- Method: Click or Area.
- Inputs: editable land-cover class swatches and selected class name.

Controls for inactive tasks are not visible. This is the main simplification.

## Pending Edit Footer

The pending edit controls stay pinned at the bottom of the left panel:

- Pending edit count.
- **Undo last**.
- **Clear edits**.
- **Update 3D model**.

The footer remains visible even when the tool details area scrolls. Button enablement follows the current behavior:

- Undo and Clear are disabled when there are no pending edits or a commit is in progress.
- Update 3D model is disabled when there are no pending edits, the map is loading, or a commit is in progress.
- During commit, the update button shows the existing spinner/loading state.

Global commit errors appear near this footer because commit applies to the whole pending edit buffer.

## 2D Plan Editor Panel Design

The center panel remains the **2D plan editor**. The Leaflet map, drawing interactions, overlays, and pending edit previews remain the core of this panel.

Display-only settings move into the 2D plan editor panel header through a compact **Display** control. This control manages:

- Basemap: CartoDB Positron, Google Satellite, OpenStreetMap.
- Overlay/backdrop: Buildings, Canopy, Land cover, None.
- Mode-specific visibility settings, such as showing building height labels when the Buildings overlay is active.

The header should also summarize the active overlay, for example **Buildings overlay** or **Canopy overlay**, so users can understand the map state without opening the Display control.

The selected edit target may still update the default overlay when the target changes, matching the current behavior:

- Building target defaults to Buildings overlay.
- Tree target defaults to Canopy overlay.
- Land cover target defaults to Land cover overlay.

The user can override the overlay from the Display control.

## 3D Result Panel Design

The right panel remains the **3D result** panel. Its purpose and placement do not change.

The empty state can keep the current guidance that users must apply an edit and click **Update 3D model** before a 3D result is rendered. If the left footer already shows a clear update action, the empty-state text can be slightly shorter, but no behavior change is required.

## State Model

`EditTab` remains the owner of edit policy and transient UI state. The current flat action model should be refactored conceptually into three pieces of state:

```ts
target: 'building' | 'tree' | 'land_cover'
task: target-specific task
method: task-specific method
```

These values resolve to the existing action and map interaction concepts before reaching `PlanMapEditor`. For example:

- Building / Add / Rectangle resolves to `add_rect` and `draw_rect_3pt`.
- Building / Height / Click resolves to `set_height_click` and `click_feature`.
- Tree / Remove / Area resolves to `remove_area` and `draw_polygon` in the tree target context.
- Land cover / Paint / Click resolves to `paint_click` and `click_point`.

This keeps `PlanMapEditor` focused on map rendering and input events. It should not need to know about the user's target/task workflow beyond the resolved interaction props it already receives.

## Component Boundaries

### `app/frontend/src/tabs/EditTab.tsx`

Owns:

- Target, task, and method state.
- Default task and method selection for each target.
- Validation for task-specific inputs.
- Pending edits and conversion to API DTOs.
- Error and info messages.
- Display control state for basemap, overlay, and visibility options.

The implementation may introduce small helper functions or a local configuration table to describe valid targets, tasks, methods, labels, defaults, and mappings to existing interactions.

### `app/frontend/src/components/PlanMapEditor.tsx`

Continues to own:

- Leaflet map lifecycle.
- Basemap rendering.
- Backdrop/overlay rendering.
- Pending edit overlay rendering.
- Drawing and feature-picking interactions.
- Building height labels and selected-building visual state.

It should receive resolved props from `EditTab`, not target/task workflow objects.

### CSS

The existing Edit tab CSS can be extended with focused classes for:

- Target selector.
- Task selector.
- Tool details area.
- Sticky pending edit footer.
- 2D plan editor header display control.

The design should avoid deeply nested cards. The left panel is already a panel; controls inside it should be compact groups, segmented controls, menus, or form sections rather than card-within-card structures.

## Error Handling And Feedback

Task-specific validation errors appear near the relevant controls in the left panel. Examples:

- Invalid height or min/base height appears in Building Height or Building Add details.
- No buildings inside a selection polygon appears in Building Height or Building Remove details.
- Polygon covers no cells appears in the active Add/Paint/Remove task details.

Global commit errors from **Update 3D model** appear near the pinned pending edit footer.

Success and info messages should be shorter and contextual. Examples:

- `Buffered 1 tree edit.`
- `Selected 4 buildings.`
- `Buffered height edit for 4 buildings.`
- `Committed 3 edits.`

Changing target or task clears stale task-specific errors and transient selections, matching the current intent of clearing errors when mode/action changes.

## Testing

Automated frontend tests should focus on pure helpers and state mapping where practical:

- Mapping target/task/method to the existing `ModeAction` and `MapInteraction` values.
- Choosing valid default task and method values when targets change.
- Preventing invalid stale methods when switching between Building, Tree, and Land cover.
- Keeping existing building-height edit helper tests passing.
- Keeping existing grid and geometry tests passing.

Manual verification should cover:

- The Edit tab still uses the same three panels.
- The left panel shows only the selected target and task controls.
- Building Add, Height, and Remove workflows are separated.
- Tree Add and Remove workflows are separated.
- Land cover Paint remains available and compact.
- Display settings are available from the 2D plan editor header.
- Target changes set sensible default overlays, while the Display control can override them.
- Pending footer remains visible and button states update correctly.
- Add, remove, height, tree, and land-cover edits still buffer and commit through **Update 3D model**.

Build verification should run:

```bash
npm run build
```

from `app/frontend`.

## Risks And Mitigations

- **Risk: Hiding inactive controls makes features less discoverable.**  
  Mitigation: task labels remain visible for the selected target, and target labels summarize available tasks.

- **Risk: Refactoring action state could break existing map interactions.**  
  Mitigation: keep the mapping to existing `ModeAction` and `MapInteraction` values explicit and covered by tests.

- **Risk: Moving display settings could make basemap/overlay harder to find.**  
  Mitigation: place the Display control in the 2D plan editor header, next to an active overlay summary.

- **Risk: A pinned footer may reduce vertical space in the narrow left panel.**  
  Mitigation: keep the footer compact and allow only the tool details area to scroll.

- **Risk: Tree and Land cover may not need as much structure as Building.**  
  Mitigation: use the same pattern but keep their task lists short; do not add artificial steps.
