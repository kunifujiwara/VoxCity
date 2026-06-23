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

## Windows / glazing (current behavior)

Model opaque mass as **solids**; model windows as **planar surfaces (not solids)**
on a layer such as `Window`. In this version, non-building layers are detected and
**skipped** — pass `roles={"Window": "window"}` to mark them. For windows today,
use the procedural material utilities on the imported buildings:

```python
from voxcity.utils.material import set_building_material_by_id, get_material_dict
mat = get_material_dict()
set_building_material_by_id(vc.voxels.classes, vc.buildings.ids, ids=[1, 2],
                           mark=mat["concrete"], window_ratio=0.4, glass_id=mat["glass"])
```

Geometry-driven windows (mapping window surfaces directly to glass voxels) are
planned (see the design spec, Path B).

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
