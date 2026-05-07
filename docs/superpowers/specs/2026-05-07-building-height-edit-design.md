# Building Height Editing Design

## Summary

Add editable building height values to the web app Edit tab's Building mode. Users can display height labels on building footprints, toggle those labels off when the map is crowded, select individual buildings or groups of buildings, buffer height changes, and commit them through the existing **Update 3D model** flow.

This design extends the current edit pipeline instead of introducing a separate editing subsystem.

## Goals

- Display current building height values for building footprints in Building mode.
- Provide a user toggle to show or hide height labels.
- Support individual building height edits by selecting one footprint and editing its value in the side panel.
- Support group height edits by multi-click selection and by polygon selection.
- For polygon group selection, include only footprints fully inside the drawn polygon.
- Keep top height as the primary value and expose min/base height as an advanced field.
- Buffer height edits and apply them only when the user clicks **Update 3D model**.
- Keep the existing Add, Remove, Tree, and Land cover editing flows intact.

## Non-Goals

- Add a building-height audit table or spreadsheet-style editor.
- Add immediate per-click model regeneration.
- Add a new backend endpoint dedicated only to height edits.
- Refactor the whole edit tab or replace the existing Leaflet map editor.

## Recommended Approach

Extend the existing Edit tab buffered-edit flow. The frontend adds height-edit controls and map-label rendering, and the backend adds a new `set_building_height` edit kind inside `/api/model/apply_edits`.

This approach matches the current user workflow: edits are previewed on the 2D map, can be undone or cleared, and are voxelized once when the user commits the batch.

## UI And Interaction

### Building Mode Controls

The Building mode panel will include three sections:

- **Add**: existing rectangle and polygon building creation controls.
- **Height**: new individual and group height editing controls.
- **Remove**: existing click and area deletion controls.

The Height section has an **Individual / Group** segmented control.

### Height Label Display

Building height labels are shown by default when the Edit tab is in Building mode and the Buildings overlay is visible. A `Show height values` toggle lets the user hide labels without leaving Building mode.

Labels read from `geo.building_geojson.features[*].properties.height`. If a building has a buffered pending height edit, its label shows the pending value so the map reflects the staged state. Undoing or clearing pending edits returns the label to the committed GeoJSON value.

The advanced min/base field reads from `geo.building_geojson.features[*].properties.min_height`. The model-geo response should include this property when `building_gdf.min_height` is present and finite, and should return `0` otherwise. If the user does not change or enable the advanced field, the frontend omits `min_height_m` from the pending edit so the backend preserves existing segment bases.

### Individual Height Editing

In Individual mode, the user clicks a footprint on the 2D map. The selected footprint is highlighted, and the side panel shows:

- Building id.
- `Height (m)` as the main input.
- An advanced `Min height / base (m)` input.
- A button to buffer a pending height edit for that building.

The input fields edit local draft values first. Pressing the buffer button appends one pending `set_building_height` edit for the selected building. If the user buffers another height edit for the same building, the latest pending operation controls the displayed label after pending edits are replayed in order. The 3D model is not regenerated until **Update 3D model** is clicked.

### Group Height Editing

In Group mode, the user can build a selected set in either of two ways:

- Click footprints to toggle buildings in or out of the selected set.
- Draw a polygon to select buildings whose footprints are fully inside the polygon.

The panel shows the selected count, a shared `Height (m)` input, an advanced `Min height / base (m)` input, and a button to buffer the pending group height edit. The selected buildings are highlighted immediately, and their labels show the pending group value after it is buffered.

Fully-inside polygon selection will be implemented as a distinct helper, separate from the existing intersection-based `buildingsInPolygon` helper. A footprint counts as fully inside when every exterior coordinate of every selected polygon ring lies inside or on the boundary of the drawn polygon, and no footprint edge crosses outside the drawn polygon. The helper returns frontend building ids from `properties.idx`.

### Pending Edit Behavior

Height edits participate in the same pending edit list as add, remove, tree, and land-cover edits:

- **Undo last** removes the most recent buffered edit.
- **Clear edits** removes all buffered edits.
- **Update 3D model** sends all pending edits to the backend in order.

Height edits are appended to the pending list in the order the user buffers them. Label overrides are computed by replaying pending edits in order, so the latest pending height edit for a building controls the displayed value. **Undo last** removes only the last pending operation, which makes repeated edits predictable. **Clear edits** removes every pending operation and returns all labels to committed values.

Buildings already pending deletion are not selectable for height editing. If the user buffers a height edit and then buffers a delete for the same building, the pending delete hides the footprint and the backend applies the operations in list order; the delete wins in the final model if it comes after the height edit.

## Frontend Data Flow

`EditTab` remains the owner of edit policy and pending state. It will add state for:

- Height label visibility.
- Height edit mode: individual or group.
- Selected individual building id.
- Selected group building ids.
- Draft height and optional draft min/base height.
- Pending height edits.

`PlanMapEditor` remains the reusable 2D map component. It gains props for:

- `showBuildingHeightLabels`.
- Selected or highlighted building ids.
- Building click callbacks that can be used for height selection as well as existing deletion selection.
- Optional pending height values for label overrides.

The frontend will add a pure selection helper, likely in `app/frontend/src/lib/grid.ts` or a small sibling module, for fully-contained building selection. This keeps group-height semantics testable and avoids changing the existing intersection-based delete-area behavior.

`PlanMapEditor` renders height labels from `ModelGeoResult.building_geojson` but does not decide whether an edit is valid, pending, or committed.

The frontend API type `PendingEditDto` gains:

```ts
{
  kind: 'set_building_height';
  building_ids: number[];
  height_m: number;
  min_height_m?: number;
}
```

The ids are frontend building ids from `properties.idx`, matching the existing delete-building behavior.

Individual edit example:

```json
{
  "kind": "set_building_height",
  "building_ids": [42],
  "height_m": 18.5
}
```

Group edit with advanced min/base value example:

```json
{
  "kind": "set_building_height",
  "building_ids": [42, 77, 81],
  "height_m": 24,
  "min_height_m": 3
}
```

## Backend Data Flow

`/api/model/apply_edits` will accept the new `set_building_height` edit kind.

The backend helper will:

1. Validate `building_ids` as integers.
2. Validate `height_m > 0`.
3. Validate `0 <= min_height_m < height_m` when `min_height_m` is provided.
4. Translate frontend GeoJSON ids to source ids using the same `building_gdf` mapping approach as `_apply_delete_buildings`.
5. Update `vc.buildings.heights` for all grid cells whose `vc.buildings.ids` match the translated source ids.
6. Update `vc.buildings.min_heights` for those same owned grid cells without discarding unrelated segment pairs. Each cell stores a list of `[min_height, max_height]` pairs. The helper must capture the cell's previous committed height before writing `vc.buildings.heights`. It should update segment pairs whose `max_height` matches that previous committed height; when `min_height_m` is omitted it preserves each matched segment's existing `min_height`, and when `min_height_m` is provided it sets the matched segment to `[min_height_m, height_m]`. If a selected cell has no matching segment pair, it should write a single `[existing_min_height_or_0, height_m]` pair for that owned cell.
7. Update matching `building_gdf.height` and, when provided, `building_gdf.min_height`.
8. Return the changed-cell count and affected building count.

After all pending edits are applied, the existing endpoint flow remains unchanged: call `regenerate_voxels(vc, inplace=True)`, refresh the raw cache, render the edit preview, and return the updated figure JSON.

## Component Boundaries

### `app/frontend/src/tabs/EditTab.tsx`

Owns height editing state, validation, pending edit management, and conversion to API DTOs. It should preserve the existing Building Add and Remove controls.

### `app/frontend/src/components/PlanMapEditor.tsx`

Renders building height labels and selected-building visual state. Reports clicked building ids to the parent. It should not own the rules for individual vs group editing or commit behavior.

### `app/frontend/src/api.ts`

Defines the new pending edit DTO variant.

### `app/backend/main.py`

Adds a focused helper for applying building height edits and wires it into the existing edit endpoint.

## Validation And Error Handling

Frontend validation prevents buffering invalid height values. The buffer button is disabled until:

- Height is finite and positive.
- Min/base height, when supplied, is finite, non-negative, and less than height.
- Individual mode has one selected building, or group mode has at least one selected building.

Backend validation mirrors those checks and returns `400` responses for malformed edit payloads.

If a valid building id no longer maps to grid cells, the backend reports zero changed cells for that edit rather than failing the full batch. This matches the current delete-building behavior and avoids surprising failures from stale client state.

The frontend prevents buffering new height edits for footprints already hidden by pending deletion. The backend still treats stale height edits as normal ordered operations so a batch remains deterministic even if a client sends height and delete edits for the same building.

## Testing

Backend tests should cover:

- Successful `set_building_height` update for one building.
- Successful group height update for multiple buildings.
- Optional min/base height update.
- Invalid height and invalid min/base validation.
- Translation from frontend GeoJSON ids through `building_gdf` to `vc.buildings.ids`.

Frontend tests should cover pure helper behavior where practical:

- Extracting height values from building GeoJSON features.
- Overlaying pending height values onto label display data.
- Converting pending frontend height edits into API DTOs.
- Selecting only fully-contained buildings for polygon group selection.
- Treating footprint vertices on the drawn polygon boundary as inside for full-containment selection.

Manual verification should cover:

- Generate or load a small model.
- Toggle height labels on and off in Building mode.
- Buffer one individual height edit and confirm the 2D label reflects the pending value.
- Undo that buffered individual edit and confirm the label returns to the committed value.
- Clear buffered height edits and confirm selected highlights and pending labels reset.
- Buffer one group height edit using multi-click selection.
- Buffer one group height edit using polygon selection with fully-contained footprints only.
- Click **Update 3D model** and confirm labels and the 3D preview update.

## Risks And Mitigations

- Dense building areas may make labels crowded. The show/hide toggle mitigates this.
- `properties.idx` represents GeoDataFrame index values, not always source ids. The backend must reuse the explicit mapping pattern already used by building deletion.
- Repeated draft edits can create multiple pending height operations for the same building. Replaying pending edits in order keeps labels, undo, and backend application deterministic.
- Applying min/base height to groups can be destructive if used casually. Keeping it in an advanced field reduces accidental changes.