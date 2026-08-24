# PALM terrain-alignment harness

Manually-run integration scripts, **not pytest targets**. There is
deliberately no `__init__.py` here and nothing is named `test_*`: these need a
working PALM install in WSL and take minutes per run, so pytest must never
collect them.

## What this proves

`voxcity.exporter.palm._build_zt` derives terrain from the **voxel grid**:

```
zt = (_ground_level(voxel_classes) - min) * meshsize
```

Every terrain height is therefore an exact multiple of `meshsize`, and on
exact multiples of `dz` PALM's own re-discretization (`ceil`, `floor` and
`nint` all agree) becomes the identity. The claim is that PALM's simulated
world then sits cell-for-cell on the voxcity voxel model. This harness checks
that claim against a **real PALM run**, not against a mock.

## The loop

### 1. Stage the fixture

```powershell
& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity `
    python tests/palm_alignment/stage_fixture.py align1 1
```

Arguments are `[JOB] [CORES]`, default `align1 1`. The script

* builds a 20x20 x 40 fixture at `meshsize 2.0` with three terraces, a water
  strip with stepped banks, a 1-voxel hut, a mid-rise, a tower, and two tree
  patches with elevated crowns (see its module docstring);
* self-checks the fixture against `_ground_level` and `_build_zt` **before**
  staging, so a fixture that stopped encoding reality fails here rather than
  silently weakening the run;
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
