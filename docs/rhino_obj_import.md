# Importing Rhino Models (OBJ) into VoxCity

`voxcity.importer.add_buildings_from_obj` adds buildings authored in Rhino to an
existing VoxCity model. Buildings are voxelized directly from their 3D mesh form;
terrain, land cover, and trees come from the base model.

## Preparing the model in Rhino

1. **Model buildings as closed solids.** Watertight geometry voxelizes most
   reliably (non-watertight meshes are repaired/filled with a warning).
2. **One building per object/layer.** Each OBJ object/group becomes one building
   (its own ID + name).
3. **Choose an anchor.** Pick one identifiable point in the model and record:
   - its model coordinates -> `anchor_model_point` (default `(0,0,0)`),
   - the real-world `(lon, lat)` of that point -> `anchor_lonlat`,
   - its real-world elevation in meters -> `anchor_elevation`.
4. **Rotation.** `rotation=0` means model **+Y points true north** and **+X
   points east**. Otherwise pass the angle (degrees).
5. **Units.** Check Rhino `Units` and pass `units` ("m", "cm", "mm", "ft", "in").
6. **Export OBJ.** `File > Export Selected > .obj`, keep **Z up**, export with
   object names/groups. If your export is Y-up, pass `z_up=False`.

## Windows / glazing

Model opaque mass as **closed solids**; model windows as **open planar surfaces**
flush with the wall (within ~1 `meshsize`). Window groups are surface-voxelized
and the building facade voxels they touch are recolored to the glass code
(`-16`); the wall behind stays solid. Windows only reclassify existing building
cells, so there must be a solid wall behind each pane (do **not** cut window
holes in the building solid).

**Auto-detection.** A group is treated as a window when its object/layer name
**or** its assigned OBJ material name contains `window`, `glass`, or `glazing`
(case-insensitive). Both `o <name>` (object) and `g <name>` (group) directives
are honored as names, so naming a group `window` works on its own — no `.mtl`
needed. Override per group with `roles`, e.g.
`roles={"Facade_North": "window"}` or force a glass-named group back to building
with `roles={"Glass_Wall": "building"}`. Customize the keywords with
`window_keywords=(...)` (e.g. add Japanese terms). Disable with
`auto_window=False`.

**Material-only exports.** If your OBJ has no named objects/layers but separates
geometry by material (a common default Rhino export, e.g. `usemtl Glass` for the
panes), the loader splits it into one group per material and uses the material
name as the group name — so a `Glass` material is auto-detected as a window with
no extra setup. This requires the companion `.mtl` to be alongside the `.obj`
(it carries the material names); without it the geometry still splits by
material but the groups get generic names and won't auto-detect.

```python
vc = add_buildings_from_obj(
    vc, "design.obj",
    anchor_lonlat=(139.7536, 35.6841), anchor_elevation=12.0,
    rotation=0.0, units="m",
    # windows auto-detected by name/material; or be explicit:
    roles={"Windows_South": "window"},
)
```

**Web app note:** in the import UI's file picker, select **both** the `.obj`
and its `.mtl` (the picker accepts multiple files) so material-name window
detection works just like the Python API — the `.mtl` is saved next to the
`.obj` server-side. Name-based detection works with the `.obj` alone. Either
way, the per-group role dropdown lets you override to **building / window /
skip**.

For procedural (non-geometry) windows, the `set_building_material_by_id`
material utilities still apply to imported buildings.

## Example

```python
from voxcity.generator import get_voxcity
from voxcity.importer import add_buildings_from_obj
from voxcity.exporter.obj import export_obj

vc = get_voxcity(rectangle_vertices, meshsize=2.0)
vc = add_buildings_from_obj(
    vc, "design.obj",
    anchor_lonlat=(139.7536, 35.6841),
    anchor_elevation=12.0,
    anchor_model_point=(0.0, 0.0, 0.0),
    rotation=0.0, move=(0.0, 0.0, 0.0), units="m",
)
export_obj(vc, "output", "with_imported_building")
```

Iterate on `rotation`/`move` and re-export to verify placement visually.
