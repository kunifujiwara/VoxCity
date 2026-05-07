# Non-Edit Tabs Guided UX Design

## Summary

Redesign the non-Edit VoxCity frontend tabs around a shared guided control-panel pattern. The redesign covers Target Area, Generation, Zoning, Solar, View, Landmark, and Export. The Edit tab is not redesigned in this pass; its recently simplified target/task/method workflow is the reference for the new interaction style.

The app keeps its current tab order and broad page layouts. Target Area, Generation, Solar, View, Landmark, and Export remain two-column or single-panel where they already are. Zoning remains a three-column workspace. Within those layouts, each tab should make the primary action obvious, reduce visible control clutter, clarify prerequisites, and keep advanced controls available without putting them in the main path.

## Goals

- Optimize all non-Edit tabs in one cohesive UX pass.
- Support both first-time users and experienced VoxCity users.
- Keep the existing tab order and broad panel structure.
- Make the primary action clear on every tab.
- Reduce visible controls at first glance while preserving advanced settings.
- Improve prerequisite and empty-state guidance between tabs.
- Reuse a small set of shared guided UI patterns instead of solving each tab separately.
- Preserve existing API behavior, state ownership, and core map/viewer components.

## Non-Goals

- Do not redesign the Edit tab in this pass.
- Do not convert the whole app into a wizard.
- Do not change backend APIs or simulation semantics.
- Do not remove advanced settings.
- Do not replace Leaflet, Three.js, `SceneViewer`, `PlanMapEditor`, or existing simulation components.
- Do not introduce broad navigation changes, new routes, or a new app-level workflow engine.
- Do not add heavy onboarding text or in-app tutorials.

## Recommended Approach

Use a shared guided control-panel pattern across the non-Edit tabs.

Each control panel should follow the same broad rhythm:

1. Primary choice or mode.
2. Essential inputs for the selected mode.
3. Secondary, display, or advanced settings in compact sections.
4. Primary action and feedback near the bottom of the panel.

This approach gives the app a coherent guided feel without slowing down repeated analytical work. It is preferable to a strict step-by-step wizard because VoxCity users often need to tweak settings and rerun analyses. It is preferable to a minimal polish pass because it creates a consistent UX language across all major workflows.

## Shared UI Patterns

The implementation should introduce shared presentational primitives where practical. These components should not own domain state or call APIs directly; tabs keep owning their state and pass props/children into the shared structure.

### Guided Panel Shell

A reusable control-panel shell with:

- Scrollable body.
- Optional pinned footer.
- Optional status/error slot near the footer.
- Compact spacing consistent with the Edit tab.

Tabs that run long actions should use the footer for their primary action:

- Generation: **Generate VoxCity Model**.
- Solar/View/Landmark: **Run Simulation**.
- Export: format-specific export action.

### Choice Groups

Use compact segmented buttons or target-style choice buttons for primary choices:

- Target Area: draw vs coordinates, then drawing mode.
- Generation: Normal vs PLATEAU.
- Zoning: 2D area vs building surfaces, then shape when relevant.
- Solar: instantaneous vs cumulative, ground vs building.
- View: green, sky, custom; ground vs building.
- Landmark: ground vs building.
- Export: CityLES vs OBJ.

Choice groups should be direct controls, not nested cards.

### Guided Sections

Use compact section labels and grouped controls for:

- Required setup.
- Optional details.
- Advanced settings.
- Display settings.
- Zone/simulation result summaries.

The visual style should reuse or extend the Edit tab's `guided-section`, `guided-section-label`, and segmented control language where it fits.

### Advanced And Display Sections

Advanced controls should remain available but not dominate the initial view. Existing components such as `ColorSettings`, `SamplingSettings`, `VoxelClassVisibility`, and source selectors can remain as collapsible/compact sections.

Display-only controls should move away from core workflow controls when possible. For map-based tabs, basemap/backdrop display controls should sit in the map/editor header, matching the Edit tab pattern.

### Empty And Readiness States

Prerequisite states should be short, contextual, and action-oriented:

- Generation needs a target area.
- Zoning, Solar, View, Landmark, and Export need a generated model.
- Sim tabs can mention zones only when zones exist or when zone stats are relevant.

These states should not be long tutorials. They should make the next required tab/action clear.

## Layout Boundaries

The broad app structure stays stable:

- Keep the current tab order: Target Area, Generation, Edit, Zoning, Solar, View, Landmark, Export.
- Keep the existing top tab bar.
- Keep Target Area as controls plus map.
- Keep Generation as controls plus 3D preview.
- Keep Zoning as controls, 2D editor, and 3D preview.
- Keep Solar/View/Landmark as controls plus 3D result viewer.
- Keep Export as a compact export panel.

Within those boundaries, individual control areas can be redesigned where useful. The design should prioritize scanability, primary action placement, and reduced control clutter.

## Target Area Tab

Target Area becomes a guided area setup panel.

The first choice is how to define the area:

- Draw on map.
- Enter coordinates.

When drawing on the map, show city search and drawing mode together. Drawing mode options remain:

- Free hand.
- Rotated free hand.
- Set dimensions.

Dimension inputs appear only for the dimension-based mode. The main action is **Load Map**. The map panel keeps the current map picker but should show a compact selected-area status when a rectangle exists.

Coordinate entry remains available, but its fields should be visually grouped as an alternate input mode rather than competing with map drawing. The **Set Rectangle** action should be the primary action for coordinate mode.

## Generation Tab

Generation keeps the two-column control/preview layout.

The left panel should show:

1. Generation mode: Normal or PLATEAU.
2. Essential generation settings, especially mesh size.
3. Source strategy.
4. Advanced parameters.
5. Pinned **Generate VoxCity Model** action.

Normal mode should keep automatic source selection as the recommended default. Auto-detected source information should be compact and readable. Manual source selection should be available in a collapsible source configuration section.

PLATEAU-specific toggles such as CityGML cache and nDSM canopy stay visible only when PLATEAU mode is active, preferably under advanced/settings grouping unless they are essential to the selected mode.

The preview panel remains the 3D preview. It should keep its current behavior and show generated results when available.

## Zoning Tab

Zoning keeps its three-column workspace:

- Left: guided zone builder and zone list.
- Center: 2D zone editor.
- Right: 3D preview and building-surface picking.

The left panel should start with zone type:

- 2D area.
- Building surfaces.

Shape selection appears only for 2D area zones. Zone creation and zone list actions remain in the left panel. The zone list should remain efficient for repeat edits, but controls should be grouped so users can understand the active zone type, active group, and available actions.

Basemap and backdrop controls should move from the left zone builder into a compact **Display** control in the 2D editor header, matching the Edit tab's display menu pattern. This separates map display state from zone-building state.

The 3D panel stays focused on preview and building-surface selection. Building-surface refinement controls remain tied to the selected building/zone state.

## Solar Tab

Solar uses the shared simulation control pattern.

The visible core setup should include:

- Calculation type: instantaneous or cumulative.
- Analysis target: ground or building surfaces.
- Required time/date fields for the selected calculation type.

Color settings, voxel visibility, and zone stats remain available but should not visually compete with the primary setup. Zone stats should stay visible when useful, especially after a run or when zones exist, but should be separated from required simulation inputs.

The **Run Simulation** action should be pinned in the control panel footer. Loading, error, and run status appear near that action.

The viewer panel remains the simulation preview and continues to use `SceneViewer`.

## View Tab

View uses the shared simulation control pattern.

The visible core setup should include:

- View type: green, sky, or custom.
- Analysis target: ground or building surfaces.
- View point height.

Custom class selection appears only for custom view type. Inclusion/exclusion mode and class toggles should be grouped inside that custom section.

Sampling settings, color settings, voxel visibility, and zone stats move into secondary sections. The current `SamplingSettings`, `ColorSettings`, and `VoxelClassVisibility` components should remain intact where possible.

The **Run Simulation** action should be pinned, with loading and API errors nearby.

## Landmark Tab

Landmark follows the shared simulation pattern but starts with landmark selection.

The control panel should group selection state into one coherent section:

- Analysis target.
- Select landmark buildings by clicking in the 3D viewer when selection mode is active.
- Selected building chips/count.
- Manual landmark ID entry.
- Clear selection.
- Back to selection when viewing a simulation result.

Simulation settings follow below selection. Sampling, color, voxel visibility, and zone stats remain secondary sections.

The **Run Simulation** action should be pinned. Errors and run status appear near it.

The 3D viewer keeps the current selection behavior: before simulation results, clicks can select landmark buildings; after results, the viewer shows the simulation result with highlights.

## Export Tab

Export becomes a compact guided export panel.

The first choice is export format:

- CityLES.
- OBJ.

Only fields relevant to the selected format are visible.

CityLES fields:

- Building material.
- Tree type.
- Trunk height ratio.

OBJ fields:

- Output filename.
- Optional NetCDF export.

The primary action is pinned and named for the selected format, for example **Export CityLES** or **Export OBJ**. Success and error feedback appear near the action.

## State And Data Flow

Existing state ownership should remain stable.

`App.tsx` continues to own cross-tab state:

- Target rectangle.
- Model readiness.
- Geometry invalidation token.
- Zone collection.
- Simulation run nonces.
- Cached figure JSON where retained for compatibility.

Each tab continues to own local state:

- Form choices.
- Advanced setting values.
- Loading state.
- Error state.
- Success/run state.
- Tab-specific selection state.

Shared guided UI components should be presentational. They can receive disabled states, labels, descriptions, children, and action handlers, but they should not know about VoxCity APIs or domain-specific state transitions.

## Error Handling And Feedback

Feedback should be closer to the relevant action.

- Prerequisite errors appear as guided empty states before a tab's main UI.
- API errors from generation, simulation, or export appear near the pinned primary action.
- Success messages appear near the same action and should be concise.
- Loading state should disable the primary action and show the existing spinner pattern.
- Advanced settings should retain values while collapsed.
- Existing simulation and zoning state should survive tab switches as it does today.

Example message style:

- `Target area is ready.`
- `Model generated. Grid: 240 x 180 x 42.`
- `Simulation complete.`
- `Export complete.`

## Testing

Automated tests should focus on behavior and helper logic rather than visual snapshots.

Recommended coverage:

- Shared guided components render footer actions, disabled states, and status slots correctly.
- Prerequisite empty states appear when target area or model state is missing.
- Primary actions are disabled while loading or when prerequisites are missing.
- Existing helper tests continue to pass for edit workflow, zones, geometry, and API client behavior.
- Any new pure helper for tab readiness or labels is covered by focused tests.

Manual verification should cover:

- Target Area: draw mode, dimension mode, coordinate mode, map load, rectangle status.
- Generation: normal auto sources, manual sources, PLATEAU settings, advanced settings, model generation.
- Zoning: 2D zones, building-surface zones, display menu, zone list actions, 2D/3D preview.
- Solar: instantaneous and cumulative runs, ground/building targets, zones, color/visibility settings.
- View: green, sky, and custom classes, sampling settings, color/visibility settings.
- Landmark: click selection, manual IDs, clear selection, run, back to selection.
- Export: CityLES and OBJ paths, success/error feedback.
- Responsive behavior for two-column and three-column layouts.

Build verification should run from `app/frontend`:

```bash
npm run build
```

## Risks And Mitigations

- **Risk: Shared components become too generic or too domain-aware.**
  Mitigation: keep them presentational and let each tab own domain state.

- **Risk: Pinned footers reduce vertical space.**
  Mitigation: use a scrollable panel body and compact footer actions, as in the Edit tab.

- **Risk: Advanced settings become harder to discover.**
  Mitigation: use clear section labels and keep common advanced sections in consistent positions.

- **Risk: Simulation tabs feel too similar despite different tasks.**
  Mitigation: share structure, but keep each tab's first section specific to its domain.

- **Risk: Moving display controls in Zoning may surprise users.**
  Mitigation: mirror the Edit tab's 2D editor header display menu so the pattern is consistent.

- **Risk: UX-only changes accidentally alter API behavior.**
  Mitigation: preserve existing API calls and state ownership, and verify primary workflows manually.