# Task 3 Report

## What I implemented

### `tests/test_downloader_gsi_integration.py`

- Appended `test_partial_5m_area_comes_back_uniformly_10m` exactly as specified in the brief.
- The test exercises the Boso peninsula ROI that exposed the bug and asserts:
  - the written GeoTIFF has z14 pixel size,
  - valid coverage exceeds 99%, and
  - the elevation range is greater than 50 m.

### `src/voxcity/generator/api.py`

- Updated the Japan auto-source docstring line to record the whole-area 10 m switch behavior:
  - 5 m when the area is fully covered,
  - otherwise the whole area at 10 m.

### `docs/superpowers/specs/2026-06-26-gsi-dem-downloader-design.md`

- Replaced the product table so it records both the GSI data ID and the actual URL tile-set path.
- Appended the 2026-10-05 amendment to Design Decision 2, documenting:
  - the `dem10b` URL tile-set name `dem`, and
  - the whole-ROI rewrite to z14 instead of per-pixel patching.
- This edit was made locally as required by the brief.
- It is intentionally absent from the commit staging list.

### `CHANGELOG.md`

- Added the `### Fixed` section under `## 1.7.0 (unreleased)` with the exact GSI DEM fallback / uniform-10 m entry from the brief.

## What I tested

### Live network integration test

Command:

```bash
VOXCITY_LIVE_GSI=1 venv/bin/python -m pytest tests/test_downloader_gsi_integration.py -q
```

Actual output:

```text
tests/test_downloader_gsi_integration.py ..                              [100%]

============================== 2 passed in 23.64s ==============================
```

### Existing GSI downloader regression file

Command:

```bash
venv/bin/python -m pytest tests/test_downloader_gsi.py -q
```

Actual output:

```text
tests/test_downloader_gsi.py .......................................     [100%]

============================== 39 passed in 3.29s ==============================
```

### Required Ruff check before commit

Command:

```bash
venv/bin/python -m ruff check src/voxcity
```

Actual output:

```text
All checks passed!
```

### Extra targeted lint check on the touched test file

Command:

```bash
venv/bin/python -m ruff check tests/test_downloader_gsi_integration.py
```

Actual output:

```text
All checks passed!
```

## Files changed

- `src/voxcity/generator/api.py`
- `CHANGELOG.md`
- `tests/test_downloader_gsi_integration.py`
- `docs/superpowers/specs/2026-06-26-gsi-dem-downloader-design.md` (local spec amendment only; not staged for commit)

## Self-review findings

- The live test is meaningful rather than vacuous:
  - it checks z14 pixel size directly from the written transform,
  - it checks coverage against `GSI_NODATA`, and
  - it checks real relief (`max - min > 50`) so a flattened or mostly-empty raster would fail.
- The changelog entry was inserted under `## 1.7.0 (unreleased)`, immediately after the existing `### Added` section.
- The spec amendment was made even though it lives under `docs/superpowers/`; it is kept local and excluded from the commit staging list per the brief.

## Concerns

- `venv/bin/python -m ruff check src/voxcity tests` reports pre-existing Ruff violations in unrelated test files outside this task's scope. The required `src/voxcity` check passed, and the touched integration test file is lint-clean.

## Fix round 1

### What I changed

- Corrected the indentation-sensitive docstring block in `auto_select_data_sources` so the `Land cover`, `Canopy height`, and `DEM` bullets are back at their original top-level indentation.
- Preserved the approved Japan DEM wording change exactly:
  - `Japan -> 'GSI DEM Japan' (bare-earth GSI DEM; 5 m where the area is fully`
  - `covered, otherwise the whole area at 10 m).`
- No other files under the task scope were modified.

### Covering checks I ran

Command:

```bash
venv/bin/python -m ruff check src/voxcity
```

Actual output:

```text
All checks passed!
```

Command:

```bash
venv/bin/python -m pytest tests/test_downloader_gsi.py -q
```

Actual output:

```text
tests/test_downloader_gsi.py .......................................     [100%]

============================== 39 passed in 3.02s ==============================
```