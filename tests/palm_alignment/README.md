# PALM alignment harness (terrain + buildings)

Manually-run integration scripts, **not pytest targets**. There is
deliberately no `__init__.py` here and nothing is named `test_*`: these need a
working PALM install in WSL and take minutes per run, so pytest must never
collect them.

## What this proves

`voxcity.exporter.palm` derives **both** terrain and buildings from the
**voxel grid**:

```
zt      = (_ground_level(voxel_classes) - min) * meshsize
heights = the column's contiguous -3 runs, relative to _ground_level
```

Every exported height is therefore an exact multiple of `meshsize`, and on
exact multiples of `dz` PALM's own re-discretization (`ceil`, `floor` and
`nint` all agree) becomes the identity. The claim is that PALM's simulated
world then sits cell-for-cell on the voxcity voxel model. This harness checks
that claim against a **real PALM run**, not against a mock.

The building half of the claim needs a fixture whose 2-D
`buildings.heights` **disagree** with its voxel columns — a fixture built
from tidy `n * meshsize` heights agrees with the voxel grid no matter what
the exporter does, and would have passed straight through the
double-rounding defect. `stage_fixture.py` asserts the disagreement (and
the exporter's resolution of it) in its self-check before staging.

## The loop

### 1. Stage the fixture

```powershell
& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity `
    python tests/palm_alignment/stage_fixture.py align1 1
```

Arguments are `[JOB] [CORES]`, default `align1 1`. The script

* builds a 20x20 x 40 fixture at `meshsize 2.0` with three terraces, a water
  strip with stepped banks, a 1-voxel hut, a mid-rise, a tower, an arcade
  with a 2-cell gap in its voxel column (which also puts the export on the
  LOD2 `buildings_3d` path), and two tree patches with elevated crowns (see
  its module docstring);
* gives every building a **continuous** height that disagrees with its voxel
  column (`3.2 / 6.8 / 21.7 / 12.4 m` against `1 / 4 / 12 / 7` cells);
* self-checks the fixture against `_ground_level`, `_build_zt` and
  `_buildings_from_voxels` **before** staging — including that every
  building's continuous height really does disagree — so a fixture that
  stopped encoding reality fails here rather than silently weakening the run;
* stages it through the app's real exporter path
  (`backend.palm_driver.stage_inputs`, with `PALM_SURFACE_NETCDF=1` set before
  `backend.config` is imported);
* saves `alignment_ref.npz` (`voxel_classes`, `meshsize`, `ground_level`,
  `meta_json`) into the job directory for the comparator;
* **applies the time-shrink itself** — spinup 300 s, `end_time` 240 s,
  averaging/dosurf windows 120 s — so a run finishes in a few minutes instead
  of a day. Nothing else needs editing.

Job directory: `C:\Users\kunih\AppData\Local\Temp\voxcity_output\palm_jobs\<JOB>\`
(`%base_data` in `.palm.config.default` is the WSL view of that path).

### 2. Run PALM in WSL

Verbatim, from Windows, keeping the `wsl.exe` process open (a detached
`nohup &` dies with the WSL session):

```
wsl.exe -e bash -lc 'cd ~/palm/pnc-work && export PATH=$HOME/palm/v25.10.1/install/bin:$PATH && palmrun -r align1 -c default -a "d3#" -X 1 -v -z'
```

`-X` must equal the `CORES` the job was staged with, or PALM aborts PAC0233
(`npex*npey != numprocs`). Run it as a background task and wait; a small job
takes a few minutes. When it is done, `MONITORING/<JOB>_stdout.000` ends with
`-finished-   time-stepping` and `OUTPUT/` holds `<JOB>_topo_surf.000.nc`.

### 3. Compare

```powershell
& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity `
    python tests/palm_alignment/compare_topo.py `
    "C:\Users\kunih\AppData\Local\Temp\voxcity_output\palm_jobs\align1"
```

Exit 0 = aligned (any mismatch was explained by PALM's own topography
filter), exit 1 = hard mismatch or an untrustworthy input file.

Buildings are counted from `topo_all == 1` (absolute z), never from the
file's own `buildings_3d`: PALM writes that one terrain-relative and copies
its second slot into its first (`data_output_topo_and_surface_setup_mod.f90`,
"Set lowest grid point in output"), so every ground-mounted column reads one
cell taller than it is. Confirmed on this fixture's 1-voxel hut, whose PALM
`buildings_3d` column is `[1, 1, 0, ...]` while `topo_all` correctly has a
single building cell.

Last run (2026-08-25, PALM v25.10.1, 1 rank, LOD2 fixture):
`solid 0 / building 0 / canopy 0` mismatching columns of 400,
`PALM-FILTER RESIDUE: 0  HARD: 0`, verdict **ALIGNED**.

## Run it on ONE rank

Stage with `CORES = 1` and run with `-X 1`. PALM's own source warns that
"in case of `DATA_TOPO_SURF` DOM occasionally reports the error 'Parallel
operation on file opened for non-parallel access', although output seems to be
fine" (`data_output_topo_and_surface_setup_mod.f90`). It is not fine: a 4-rank
run of this fixture wrote a `topo_surf` whose 2-D `zt` disagreed with the
static driver in 181 of 400 cells (`buildings_2d` in 19), and an older 4-rank
job dropped cells from the 3-D `topo_all` instead. The single-rank run
reproduces the static driver exactly, in every field.

`compare_topo.py` guards against this on its own: it recomputes `zt` and
`buildings_2d` from `topo_all` and refuses to call a file evidence when the
two disagree. The physics being tested — the exporter's quantization making
PALM's rounding the identity — does not depend on the domain decomposition,
so one rank costs nothing scientifically.

## Mutation-checking the comparator

A comparator that cannot fail is not a gate. To re-verify it:

1. load `alignment_ref.npz`, set one solid voxel of a mid-terrace column to
   `0` (air), re-save;
2. re-run `compare_topo.py` — it must exit 1 with a `[HARD]` row at that
   `(y, x)`;
3. do the same for a column next to a >= 2-cell terrace step — it must exit 0
   with a `[PALM-FILTER RESIDUE]` row, proving the classifier is not simply
   always-HARD;
4. restore by re-running `stage_fixture.py` with the same arguments (it
   rewrites the `.npz`).

Both branches were exercised this way when the harness was written.

## Housekeeping

**Ports 3000 and 8000 belong to a different project. Never kill them.**
Nothing in this harness listens on a port; if something is holding one, it is
not this.
