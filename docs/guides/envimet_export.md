# Exporting to ENVI-met

`voxcity.exporter.envimet.export_inx` writes a VoxCity model as an ENVI-met
`.INX` project file. Buildings, land cover, terrain, and trees are all
converted, with trees placed as `<3Dplants>` entries referencing plant IDs of
the form `HxxW01` (e.g. `H05W01` for a 5 m tall tree).

## Plant database registration

Those `HxxW01` IDs only resolve inside ENVI-met if a matching plant database
defines them. By default, `export_inx` now also writes a
`projectdatabase.edb` file next to the `.INX`, containing `PLANT3D`
definitions for `H01W01`..`H50W01`:

```python
from voxcity.exporter.envimet import export_inx

export_inx(vc, "output", file_basename="voxcity")
# -> output/voxcity.INX
# -> output/projectdatabase.edb
```

**In ENVI-met Spaces**, register `projectdatabase.edb` as the project's plant
database (Project Settings -> Databases) *before* opening `voxcity.INX`.
Without this step, ENVI-met reports a "tree identification error" for any
`HxxW01` ID it cannot find.

## Opting out

If you already maintain your own plant database (or are exporting a model
with no trees), skip the `.edb` write with:

```python
export_inx(vc, "output", file_basename="voxcity", generate_plant_db=False)
```

You can also call `generate_edb_file(output_dir=..., lad=..., trunk_height_ratio=...)`
directly to (re)generate the database on its own, e.g. with a custom leaf
area density (`lad`) or trunk-to-total-height ratio.
