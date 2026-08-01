# -*- coding: utf-8 -*-
"""
Tokyo LAS → DSM/DTM → nDSM → crop by WGS84 rectangle → meter-grid sampler

What’s included
- Downloads & extracts Tokyo GIC LAS ZIPs referenced by vector tiles
- Fast LAS→DSM/DTM (numpy reductions; handles ScaledArrayView, empty LAS)
- Forces a single, correct CRS for central Tokyo (auto-pick EPSG:6677)
- Merge with on-the-fly reprojection to target CRS
- Build nDSM and crop by WGS84 polygon in raster CRS
- Sample on a regular meter grid; rowcol() flatten fix; robust to NoData

Requires: requests, laspy, rasterio, shapely, pyproj, numpy, matplotlib
"""

import os, re, math, warnings, zipfile
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor

import requests
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.windows import Window, from_bounds
from rasterio.transform import from_origin
from rasterio.mask import mask as rio_mask
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
import rasterio.crs

try:
    import laspy
except ImportError:
    laspy = None  # Lazy: only needed for LAS file reading functions
from shapely.geometry import Polygon, mapping
from shapely.ops import transform as shp_transform
from pyproj import Transformer, Geod
import matplotlib.pyplot as plt

# =========================
# Helpers: web tiles & zips
# =========================
def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def extract_zip_urls(pbf_content):
    urls = re.findall(r'https://[^\s"]+\.zip', pbf_content.decode('utf-8', errors='ignore'))
    return list(set(urls))

def download_zip(url, output_dir):
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.basename(urlparse(url).path)
            out = os.path.join(output_dir, filename)
            with open(out, 'wb') as f:
                f.write(r.content)
            print(f"Downloaded: {filename}")
            return True
        else:
            print(f"Failed to download {url}: {r.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def download_pbf_and_get_urls(url):
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            return extract_zip_urls(r.content)
        else:
            print(f"Failed to download PBF {url}: {r.status_code}")
            return []
    except Exception as e:
        print(f"Error downloading PBF {url}: {e}")
        return []

def download_tiles(base_url, bounds, zoom_levels, output_dir, max_workers=10):
    min_lat, min_lon, max_lat, max_lon = bounds
    print("Target area:")
    print(f"  Latitude range: {min_lat} to {max_lat}")
    print(f"  Longitude range: {min_lon} to {max_lon}")
    print(f"  Width (approx): {(max_lon - min_lon) * 111320 * math.cos(math.radians(min_lat)):.2f} meters")
    print(f"  Height (approx): {(max_lat - min_lat) * 111320:.2f} meters")

    pbf_urls = []
    for zoom in zoom_levels:
        min_x, min_y = deg2num(min_lat, min_lon, zoom)
        max_x, max_y = deg2num(max_lat, max_lon, zoom)
        min_x, max_x = sorted([min_x, max_x])
        min_y, max_y = sorted([min_y, max_y])
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                pbf_urls.append(base_url.format(z=zoom, x=x, y=y))
    print(f"Found {len(pbf_urls)} PBF tiles to process")

    zip_urls = set()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for fut in [ex.submit(download_pbf_and_get_urls, u) for u in pbf_urls]:
            zip_urls.update(fut.result())

    print(f"Found {len(zip_urls)} unique ZIP files to download")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = [ex.submit(download_zip, url, output_dir) for url in zip_urls]
        successful = sum(1 for r in results if r.result())
    print(f"Successfully downloaded {successful} of {len(zip_urls)} ZIP files")

def extract_zip(zip_path, extract_base_dir):
    try:
        zip_name = Path(zip_path).stem
        extract_dir = os.path.join(extract_base_dir, zip_name)
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as z:
            for info in z.infolist():
                if '..' in info.filename or info.filename.startswith('/'):
                    print(f"Warning: skipping unsafe path in {zip_path}: {info.filename}")
                    continue
                z.extract(info, extract_dir)
        print(f"Successfully extracted: {zip_path}")
        return True
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return False

def extract_all_zips(zip_dir, extract_base_dir, max_workers=5):
    os.makedirs(extract_base_dir, exist_ok=True)
    zips = [os.path.join(zip_dir, f) for f in os.listdir(zip_dir) if f.lower().endswith('.zip')]
    if not zips:
        print(f"No ZIP files found in {zip_dir}")
        return
    print(f"Found {len(zips)} ZIP files to extract")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = [ex.submit(extract_zip, z, extract_base_dir) for z in zips]
        ok = sum(1 for r in results if r.result())
    print("\nExtraction complete:")
    print(f"Successfully extracted: {ok}/{len(zips)} files")
    print(f"Extracted files can be found in: {extract_base_dir}")

def find_las_files(base_dir):
    outs = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(('.las', '.laz')):
                outs.append(os.path.join(root, f))
    return outs


# =========================
# LAS spatial filtering
# =========================
def read_las_bounds_and_crs(las_path):
    """Return ((xmin, ymin, xmax, ymax), crs) from LAS/LAZ header.
    Falls back gracefully if CRS is missing.
    """
    if laspy is None:
        raise ImportError("laspy is required for LAS file reading")
    try:
        with laspy.open(las_path) as f:
            hdr = f.header
            mins = getattr(hdr, "mins", None)
            maxs = getattr(hdr, "maxs", None)
            if mins is None or maxs is None:
                return None, None
            x_min, y_min = float(mins[0]), float(mins[1])
            x_max, y_max = float(maxs[0]), float(maxs[1])
            try:
                crs = hdr.parse_crs()
            except Exception:
                crs = None
            return (x_min, y_min, x_max, y_max), crs
    except Exception as e:
        print(f"Warning: failed to read LAS header for {las_path}: {e}")
        return None, None

def filter_las_files_by_aoi(las_files, rectangle_vertices, target_crs, pad_m=0.0):
    """Filter LAS/LAZ files to only those intersecting the AOI rectangle.

    rectangle_vertices: WGS84 lon/lat vertices (can be open or closed ring)
    target_crs: CRS string like "EPSG:6677" used to evaluate intersections
    pad_m: optional buffer in meters around AOI in target CRS
    """
    if not las_files:
        return []

    # Close polygon if needed
    if rectangle_vertices[0] != rectangle_vertices[-1]:
        rectangle_vertices = list(rectangle_vertices) + [rectangle_vertices[0]]
    poly_w84 = Polygon(rectangle_vertices)
    tcrs = normalize_crs(target_crs) or target_crs
    to_target = Transformer.from_crs("EPSG:4326", tcrs, always_xy=True).transform
    poly_target = shp_transform(to_target, poly_w84)
    if pad_m:
        poly_target = poly_target.buffer(pad_m)

    selected = []
    for fp in las_files:
        bounds, las_crs = read_las_bounds_and_crs(fp)
        if not bounds:
            continue
        xmin, ymin, xmax, ymax = bounds
        bpoly = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)])
        try:
            if las_crs is not None:
                to_target_from_las = Transformer.from_crs(las_crs, tcrs, always_xy=True).transform
                bpoly_t = shp_transform(to_target_from_las, bpoly)
            else:
                bpoly_t = bpoly
        except Exception as e:
            print(f"Warning: CRS transform failed for {fp}: {e}. Assuming target CRS.")
            bpoly_t = bpoly

        if bpoly_t.intersects(poly_target):
            selected.append(fp)

    print(f"Selected {len(selected)} / {len(las_files)} LAS files intersecting AOI")
    return selected


# =========================
# CRS helpers
# =========================
def choose_jprcs_epsg(lons):
    """Pick JGD2011 Plane Rectangular CRS by AOI lon; Tokyo central → EPSG:6677."""
    lon_c = sum(lons) / len(lons)
    return "EPSG:6677" if lon_c >= 139.0 else "EPSG:6676"  # simple, effective for Kanto


def normalize_crs(crs_like):
    """Return a rasterio CRS object, with safe fallbacks for older GDAL/PROJ.

    - Try rasterio's native parsing first
    - Fall back to PROJ4 strings for JGD2011 PRCS zones 8/9 (EPSG:6676/6677)
    - Return None if input is falsy or cannot be parsed
    """
    if not crs_like:
        return None
    try:
        return rasterio.crs.CRS.from_user_input(crs_like)
    except Exception:
        text = str(crs_like).strip().upper()
        if text.endswith("EPSG:6677") or text == "6677":
            proj4 = (
                "+proj=tmerc +lat_0=36 +lon_0=139.83333333333334 +k=0.9999 "
                "+x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs"
            )
            try:
                return rasterio.crs.CRS.from_string(proj4)
            except Exception:
                return None
        if text.endswith("EPSG:6676") or text == "6676":
            proj4 = (
                "+proj=tmerc +lat_0=36 +lon_0=138.5 +k=0.9999 "
                "+x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs"
            )
            try:
                return rasterio.crs.CRS.from_string(proj4)
            except Exception:
                return None
        return None


# =========================
# nDSM evidence band schema
# =========================
# Bands 2-6 of the nDSM raster, in order. Band 1 is the nDSM height and keeps
# its pre-existing meaning exactly; these five are the per-pixel LiDAR
# statistics that app/backend/ndsm_refine.py's load_ndsm_evidence() pools onto
# the model grid. Tokyo LAS carries no vegetation or building classification --
# roofs and crowns share class 1 -- so the only thing that can separate them
# downstream is physical evidence, and this is where it enters the pipeline.
#
# WHY COUNTS AND SUMS, NEVER RATIOS. Counts and sums are *additive*: adding them
# over whichever 0.5 m pixels happen to fall under a model cell reproduces
# exactly the statistic that would have been computed from the underlying points
# directly, at any cell size and with no resampling error --
#
#     mrf       = sum(n_multi) / sum(n_all)
#     roughness = sqrt(sum(sum_z2)/n - (sum(sum_z)/n)^2),  n = sum(n_nonground)
#
# Ratios stored per pixel would only be *averageable*, which is wrong whenever
# pixel point counts differ, and at the measured 5-8 returns per 0.5 m pixel a
# per-pixel ratio is a handful of Bernoulli trials -- statistically meaningless
# at native resolution even before the pooling error. The whole design rests on
# this property; see pool_evidence() in ndsm_refine.py, which is the consumer.
#
# All six bands share one dtype (float32) and one nodata value. That is not a
# preference: GDAL's GTiff driver cannot create a dataset with per-band data
# types, and rasterio exposes a single ``src.nodata`` -- which the reader uses --
# so uint16 count bands are not representable alongside float32 height bands in
# the single file the reader opens. Counts are exact in float32 well past any
# per-pixel return count (2^24), so nothing is lost.
EVIDENCE_BAND_NAMES = ("n_all", "n_multi", "n_nonground", "sum_z", "sum_z2")

#: Band count of the evidence nDSM. Must equal ndsm_refine.EVIDENCE_BANDS; the
#: reader treats ``src.count < 6`` as degraded mode (height only).
NDSM_BAND_COUNT = 1 + len(EVIDENCE_BAND_NAMES)


def _nearest_fill(arr):
    """Nearest-neighbour fill of NaN gaps in a 2-D array.

    Returns ``None`` when every value is NaN -- there is nothing to fill from,
    and returning an all-NaN array would silently poison every derived height.
    """
    valid = np.isfinite(arr)
    if not valid.any():
        return None
    if valid.all():
        return arr
    from scipy.ndimage import distance_transform_edt

    # return_indices gives, for each cell, the index of the nearest True cell of
    # ``valid`` -- i.e. the nearest pixel that actually has a ground height.
    idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return arr[tuple(idx)]


def _evidence_arrays(idx_flat, height, width, z, cls, n_returns, dtm_arr,
                     dtm_classes, nodata):
    """Per-pixel evidence bands for one tile, in EVIDENCE_BAND_NAMES order.

    Every point is placed by ``np.bincount`` on its pixel index; there is no
    Python loop over points, because tiles run to millions of them.

    The per-point height is ``z - dtm[pixel]`` against the tile's *nearest-
    filled* ground raster. The fill matters: ground returns are sparse under
    canopy and absent under roofs, so the raw DTM has holes exactly where the
    interesting points are, and an unfilled lookup would drop their heights.

    Ground returns are counted in ``n_all`` and ``n_multi`` but excluded from
    ``n_nonground``/``sum_z``/``sum_z2``. That is deliberate and is what makes
    the multi-return fraction physical: a pulse that reaches the ground through
    a crown echoes several times, so its ground echo *is* a multi-return point
    and belongs in both halves of the ratio, while its height above ground
    (~0 m) says nothing about the crown.
    """
    n_pixels = int(height) * int(width)
    bands = np.full((len(EVIDENCE_BAND_NAMES), n_pixels), 0.0, dtype=np.float64)

    n_all = np.bincount(idx_flat, minlength=n_pixels)
    bands[0] = n_all

    multi = np.asarray(n_returns, dtype=np.int64) > 1
    if multi.any():
        bands[1] = np.bincount(idx_flat[multi], minlength=n_pixels)

    nonground = ~np.isin(cls, dtm_classes)
    if nonground.any():
        ng_idx = idx_flat[nonground]
        bands[2] = np.bincount(ng_idx, minlength=n_pixels)

        ground = _nearest_fill(dtm_arr)
        if ground is None:
            # No class-2 point anywhere on the tile, so no height is definable.
            # Leave the sums at zero and let n_nonground stand: the nDSM band is
            # nodata across the whole tile too, so the co-occurrence rule in
            # build_ndsm() drops these pixels before anything can read them.
            print("  Warning: tile has no ground returns; evidence heights are 0")
        else:
            h = z[nonground].astype(np.float64) - ground.ravel()[ng_idx]
            bands[3] = np.bincount(ng_idx, weights=h, minlength=n_pixels)
            bands[4] = np.bincount(ng_idx, weights=h * h, minlength=n_pixels)

    # A pixel with no returns has no evidence -- not zero evidence. Marked
    # nodata across all five bands together, so the bands stay mutually
    # consistent from the very first raster written.
    bands[:, n_all == 0] = float(nodata)
    return bands.reshape(len(EVIDENCE_BAND_NAMES), int(height), int(width)
                         ).astype(np.float32, copy=False)


# =========================
# LAS → DSM / DTM (fast)
# =========================
def process_las_to_raster(las_path, resolution=0.5, dsm_classes=(1, 3), dtm_classes=(2,),
                          with_evidence=False):
    """
    Build DSM (max Z) and DTM (min Z) on a regular grid aligned to LAS extents.
    Returns dicts: {'array','transform','bounds','nodata'}

    With ``with_evidence=True`` a third raster is returned on the same grid,
    holding the five bands of :data:`EVIDENCE_BAND_NAMES` stacked as
    ``(5, height, width)``. The DSM and DTM are computed by the identical code
    either way and are bit-identical to the pre-evidence function -- the flag
    only adds output. The arity of the return value depends on the flag so that
    existing two-value callers keep working untouched.
    """
    if laspy is None:
        raise ImportError("laspy is required for LAS file reading")
    try:
        las = laspy.read(las_path)

        # ScaledArrayView → numpy arrays
        x   = np.asarray(las.x, dtype=np.float64)
        y   = np.asarray(las.y, dtype=np.float64)
        z   = np.asarray(las.z, dtype=np.float32)
        cls = np.asarray(las.classification)

        print(f"LAS file: {las_path}")
        print(f"  Point count: {x.size}")
        if x.size == 0:
            raise ValueError("Empty LAS (0 points).")

        print(f"  X range: {float(np.min(x))} to {float(np.max(x))}")
        print(f"  Y range: {float(np.min(y))} to {float(np.max(y))}")
        print(f"  Classifications: {np.unique(cls)}")

        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.min(y)), float(np.max(y))

        width  = max(1, int(np.ceil((x_max - x_min) / resolution)))
        height = max(1, int(np.ceil((y_max - y_min) / resolution)))
        transform = from_origin(x_min, y_max, resolution, resolution)

        cols = np.clip(((x - x_min) / resolution).astype(np.int64), 0, width  - 1)
        rows = np.clip(((y_max - y)  / resolution).astype(np.int64), 0, height - 1)
        idx_flat = rows * width + cols

        dsm_arr = np.full((height, width), -np.inf, dtype=np.float32)
        dsm_mask = np.isin(cls, dsm_classes)
        if dsm_mask.any():
            np.maximum.at(dsm_arr.ravel(), idx_flat[dsm_mask], z[dsm_mask])
        dsm_arr[np.isneginf(dsm_arr)] = np.nan

        dtm_arr = np.full((height, width),  np.inf, dtype=np.float32)
        dtm_mask = np.isin(cls, dtm_classes)
        if dtm_mask.any():
            np.minimum.at(dtm_arr.ravel(), idx_flat[dtm_mask], z[dtm_mask])
        dtm_arr[np.isposinf(dtm_arr)] = np.nan

        base = {
            'transform': transform,
            'bounds': (x_min, y_min, x_max, y_max),
            'nodata': -9999.0,
        }
        dsm = dict(array=dsm_arr.copy(), **base)
        dtm = dict(array=dtm_arr.copy(), **base)
        if not with_evidence:
            return dsm, dtm

        try:
            n_returns = np.asarray(las.number_of_returns)
        except Exception as exc:
            # Refusing rather than substituting zeros: an all-single-return
            # reading makes the multi-return fraction 0 everywhere, which is
            # the maximally roof-like value, and the classifier would suppress
            # every tree it was built to keep.
            raise ValueError(
                f"{las_path} has no 'number_of_returns' dimension, so the "
                f"multi-return evidence cannot be measured: {exc}"
            )

        evidence = dict(
            array=_evidence_arrays(
                idx_flat, height, width, z, cls, n_returns, dtm_arr,
                dtm_classes, base['nodata'],
            ),
            **base,
        )
        return dsm, dtm, evidence

    except Exception as e:
        print(f"Error processing LAS file {las_path}: {e}")
        return (None, None, None) if with_evidence else (None, None)

def save_raster(raster_data, output_path, crs):
    """Write a raster dict to GeoTIFF.

    A 2-D ``array`` writes one band exactly as before; a 3-D ``(bands, h, w)``
    array writes that many bands. Multi-band is how the evidence tile is
    persisted: all its bands share one dtype and one nodata value, which is what
    a GTiff can express and what the runtime reader assumes.
    """
    array = raster_data['array'].astype(np.float32, copy=False)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    elif array.ndim != 3:
        raise ValueError(f"raster array must be 2-D or 3-D, got {array.ndim}-D")
    transform = raster_data['transform']
    nodata = float(raster_data.get('nodata', -9999.0))

    profile = {
        'driver': 'GTiff',
        'height': array.shape[1],
        'width':  array.shape[2],
        'count':  array.shape[0],
        'dtype':  rasterio.float32,
        'transform': transform,
        'nodata': nodata,
    }
    crs_obj = normalize_crs(crs)
    if crs_obj is not None:
        profile['crs'] = crs_obj

    array = np.where(np.isnan(array), nodata, array).astype(np.float32, copy=False)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(array)
    print(f"Saved raster to {output_path}")
    return True

def process_las_files(las_files, dsm_output_dir, dtm_output_dir, resolution, target_crs,
                      evidence_output_dir=None):
    """Rasterize each LAS to per-tile DSM/DTM GeoTIFFs.

    Pass *evidence_output_dir* to also write the five-band evidence tile beside
    each DSM/DTM; the return value then gains a third list. Omitting it
    reproduces the pre-evidence behaviour byte for byte, which is what the
    comparison runs in Task 7/8 need.
    """
    os.makedirs(dsm_output_dir, exist_ok=True)
    os.makedirs(dtm_output_dir, exist_ok=True)
    with_evidence = evidence_output_dir is not None
    if with_evidence:
        os.makedirs(evidence_output_dir, exist_ok=True)

    dsm_files, dtm_files, ev_files = [], [], []
    for i, las_path in enumerate(las_files):
        print(f"Processing file {i+1}/{len(las_files)}: {os.path.basename(las_path)}")
        result = process_las_to_raster(las_path, resolution=resolution,
                                       with_evidence=with_evidence)
        dsm, dtm = result[0], result[1]
        evidence = result[2] if with_evidence else None
        if dsm is None or dtm is None:
            continue
        base = os.path.splitext(os.path.basename(las_path))[0]
        dsm_path = os.path.join(dsm_output_dir, f"{base}_dsm.tif")
        dtm_path = os.path.join(dtm_output_dir, f"{base}_dtm.tif")
        if save_raster(dsm, dsm_path, target_crs):
            dsm_files.append(dsm_path)
        if save_raster(dtm, dtm_path, target_crs):
            dtm_files.append(dtm_path)
        if evidence is not None:
            ev_path = os.path.join(evidence_output_dir, f"{base}_evidence.tif")
            if save_raster(evidence, ev_path, target_crs):
                ev_files.append(ev_path)
    print(f"Created {len(dsm_files)} DSM GeoTIFFs and {len(dtm_files)} DTM GeoTIFFs")
    if with_evidence:
        print(f"Created {len(ev_files)} evidence GeoTIFFs "
              f"({len(EVIDENCE_BAND_NAMES)} bands each)")
        return dsm_files, dtm_files, ev_files
    return dsm_files, dtm_files


# =========================
# Merge & nDSM
# =========================
#: Fraction of the smaller tile that two footprints must share before
#: :func:`warn_if_inputs_overlap` calls it an overlap.
#:
#: Not zero, and not a bare ``intersects`` test. Tokyo's survey sheets abut, and
#: each tile's raster extent is its point extent rounded *up* to a whole pixel,
#: so neighbouring sheets routinely share a sliver up to one pixel wide -- on a
#: ~700 px tile that is ~0.14%. A check that fired on every adjacent pair would
#: emit thousands of warnings per rebuild, and a warning nobody can read is a
#: warning nobody will act on. One percent is two orders of magnitude above the
#: sliver and two below a genuinely duplicated sheet.
OVERLAP_WARN_FRAC = 0.01


def warn_if_inputs_overlap(input_files, target_crs, min_overlap_frac=OVERLAP_WARN_FRAC,
                           label="tiles"):
    """Check the non-overlap assumption that ``method="first"`` rests on.

    ``merge`` resolves a contested pixel by taking the first dataset that has
    data there. That is only well-defined -- and only band-consistent between
    the DSM, DTM and evidence merges, which are three separate merges over the
    same sheets -- if no two sheets cover the same ground. The survey sheets
    *should* be disjoint, but "should" is exactly the kind of assumption that
    quietly stops holding, so it is measured rather than assumed.

    Warns (never raises): an overlap makes the merge arbitrary, not wrong, and
    aborting a multi-hour rebuild over it would be worse than reporting it.

    Returns the list of ``(file_a, file_b, fraction)`` offenders, empty when the
    assumption holds.
    """
    files = list(input_files)
    if len(files) < 2:
        return []

    from shapely.geometry import box
    from shapely.strtree import STRtree

    tcrs = normalize_crs(target_crs)
    boxes = []
    for fp in files:
        try:
            with rasterio.open(fp) as src:
                b = src.bounds
                geom = box(b.left, b.bottom, b.right, b.top)
                if src.crs is not None and tcrs is not None and src.crs != tcrs:
                    to_target = Transformer.from_crs(src.crs, tcrs, always_xy=True).transform
                    geom = shp_transform(to_target, geom)
        except Exception as exc:
            print(f"Warning: could not read bounds of {fp} for overlap check: {exc}")
            geom = None
        boxes.append(geom)

    indexed = [i for i, g in enumerate(boxes) if g is not None and not g.is_empty]
    if len(indexed) < 2:
        return []
    tree = STRtree([boxes[i] for i in indexed])

    offenders = []
    for position, i in enumerate(indexed):
        gi = boxes[i]
        for hit in tree.query(gi):
            if int(hit) <= position:          # unordered pairs, each seen once
                continue
            j = indexed[int(hit)]
            gj = boxes[j]
            shared = gi.intersection(gj).area
            if shared <= 0.0:
                continue
            smaller = min(gi.area, gj.area)
            if smaller <= 0.0:
                continue
            frac = shared / smaller
            if frac >= float(min_overlap_frac):
                offenders.append((files[i], files[j], frac))

    if offenders:
        offenders.sort(key=lambda t: -t[2])
        worst = offenders[0]
        warnings.warn(
            f"{len(offenders)} pair(s) of {label} overlap by at least "
            f"{100 * float(min_overlap_frac):.1f}% of the smaller footprint; "
            f"merge(method='first') will pick between them arbitrarily. Worst: "
            f"{os.path.basename(worst[0])} / {os.path.basename(worst[1])} "
            f"({100 * worst[2]:.1f}%)",
            UserWarning,
            stacklevel=2,
        )
    return offenders


def merge_geotiffs(input_files, output_path, target_crs, nodata_value=-9999.0, res=None,
                   check_overlap=True):
    """
    Merge multiple GeoTIFFs into one, pre-warping each to target_crs so we
    don’t rely on merge(dst_crs=...), which isn’t available in older rasterio.

    *check_overlap* verifies the disjoint-sheet assumption behind
    ``method="first"``; see :func:`warn_if_inputs_overlap`. Turn it off when the
    inputs are known to interleave, as the intermediate products of
    :func:`merge_geotiffs_batched` do.
    """
    if not input_files:
        print("No GeoTIFF files to merge")
        return None

    if check_overlap:
        warn_if_inputs_overlap(input_files, target_crs)

    # Open and wrap as VRTs in target_crs
    warped = []
    opened = []
    try:
        # Pick a default resolution from the first dataset if not provided
        first_src = rasterio.open(input_files[0])
        opened.append(first_src)
        default_res = first_src.res
        if res is None:
            res = default_res

        tcrs = normalize_crs(target_crs) or target_crs
        for fp in input_files:
            src = rasterio.open(fp)
            opened.append(src)
            if src.crs is None:
                # Assume inputs are already in target CRS; set src_crs and wrap
                vrt = WarpedVRT(
                    src,
                    crs=tcrs,
                    src_crs=tcrs,
                    src_nodata=src.nodata,
                    dst_nodata=nodata_value,
                    resampling=Resampling.nearest,
                    resolution=res,
                )
            else:
                vrt = WarpedVRT(
                    src,
                    crs=tcrs,
                    src_nodata=src.nodata,
                    dst_nodata=nodata_value,
                    resampling=Resampling.nearest,
                    resolution=res,
                )
            warped.append(vrt)

        print(f"Target CRS: {tcrs}")
        print("Merging files...")
        # No dst_crs here — all inputs are already warped VRTs
        merged, transform = merge(warped, nodata=nodata_value, method="first")

        meta = {
            "driver": "GTiff",
            "height": merged.shape[1],
            "width": merged.shape[2],
            "count": merged.shape[0],
            "dtype": merged.dtype,
            "crs": tcrs,
            "transform": transform,
            "nodata": nodata_value
        }
        print(f"Writing merged file to {output_path}")
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(merged)
        print("Merge completed successfully")
        return output_path

    finally:
        for vrt in warped:
            try: vrt.close()
            except: pass
        for src in opened:
            try: src.close()
            except: pass

def merge_geotiffs_batched(
    input_files,
    output_path,
    target_crs,
    nodata_value=-9999.0,
    res=None,
    batch_size=250,
    tmp_dir=None,
    check_overlap=True,
):
    """Merge many GeoTIFFs in batches to reduce peak memory.

    - Chunks inputs into groups of batch_size and merges each group to a temp file
    - Recursively merges the intermediate files until a single output remains
    - Uses the same reprojection logic as merge_geotiffs

    The overlap check runs once over the *whole* input list rather than per
    chunk, and is disabled for the individual merges. Per chunk it would miss
    every pair split across a chunk boundary; on the intermediates it would fire
    on every rebuild, since a chunk is an arbitrary slice of the file list and
    the resulting bounding boxes interleave by construction even though their
    data does not.
    """
    files = list(input_files)
    if not files:
        print("No GeoTIFF files to merge")
        return None

    if check_overlap:
        warn_if_inputs_overlap(files, target_crs)

    if len(files) <= int(batch_size):
        return merge_geotiffs(files, output_path, target_crs, nodata_value=nodata_value,
                              res=res, check_overlap=False)

    base_dir = tmp_dir or os.path.join(os.path.dirname(output_path) or ".", "_merge_tmp")
    os.makedirs(base_dir, exist_ok=True)

    intermediates = []
    try:
        for i in range(0, len(files), int(batch_size)):
            chunk = files[i:i+int(batch_size)]
            tmp_out = os.path.join(base_dir, f"chunk_{i//int(batch_size):04d}.tif")
            print(f"Merging batch {i//int(batch_size)+1}/{int(math.ceil(len(files)/float(batch_size)))} → {tmp_out}")
            merged_path = merge_geotiffs(chunk, tmp_out, target_crs, nodata_value=nodata_value,
                                         res=res, check_overlap=False)
            if merged_path is None:
                continue
            intermediates.append(merged_path)

        if not intermediates:
            print("No intermediates produced; aborting batched merge")
            return None

        # Final merge of intermediates
        print(f"Merging {len(intermediates)} intermediate files into final output ...")
        final_path = merge_geotiffs(intermediates, output_path, target_crs,
                                    nodata_value=nodata_value, res=res, check_overlap=False)
        print("Batched merge completed successfully")
        return final_path
    finally:
        # Best-effort cleanup of intermediate files
        for fp in intermediates:
            try:
                if os.path.exists(fp):
                    os.remove(fp)
            except Exception:
                pass
        # Do not remove the tmp directory automatically; it might be shared

def build_ndsm(dsm_path, dtm_path, out_path, nodata_value=-9999.0, evidence_path=None):
    """Compute nDSM = DSM - DTM in streaming windows to avoid large RAM usage.

    Writes directly to disk using tiled BigTIFF with compression.

    With *evidence_path* -- the merged five-band raster from
    :func:`process_las_files` -- the output carries the full
    :data:`NDSM_BAND_COUNT` band schema: band 1 unchanged, bands 2-6 the
    evidence in :data:`EVIDENCE_BAND_NAMES` order. Without it, one band is
    written exactly as before, which is the comparison baseline Task 7/8 need.

    **Nodata co-occurs across all six bands.** A pixel is either data everywhere
    or nodata everywhere, and this is the only place that can be guaranteed:
    the height's validity comes from the DSM/DTM pair and the evidence's from
    the point counts, and the two are independently sparse. ``load_ndsm_evidence``
    maps band-1 nodata to NaN and evidence nodata to 0.0, which are consistent
    readings of the *same* pixel state only if that state is shared. Measured on
    the inconsistent case: a pixel with no ground reference and 160 returns
    reads back as a NaN height beside ``n_all=160, roughness=1.0`` -- confident
    evidence about a cell that has no height, which is precisely the input that
    makes the classifier assert a verdict it has no basis for.

    The masking runs both ways, but only one way ever bites in practice. A pixel
    with no returns has no DSM and no DTM point either, so evidence-nodata is a
    subset of height-nodata; the traffic is the other direction, dropping
    evidence at pixels the DSM/DTM pair could not give a height. Measured on
    real Chuo-ku sheets that is ~40% of returns -- and it is *not* the returns
    the classifier needs: the multi-return fraction among the dropped points is
    lower (0.31 / 0.12 on two sample tiles) than among the kept ones
    (0.39 / 0.22), because a pixel that holds both a ground echo and a canopy
    echo is by definition a pixel a pulse penetrated. The rule keeps the
    penetrating pixels and discards the opaque ones.
    """
    write_evidence = evidence_path is not None
    with rasterio.open(dsm_path) as dsm, rasterio.open(dtm_path) as dtm:
        if dsm.crs != dtm.crs or dsm.transform != dtm.transform or dsm.shape != dtm.shape:
            raise ValueError("DSM and DTM are not perfectly aligned.")

        evidence = rasterio.open(evidence_path) if write_evidence else None
        try:
            if evidence is not None:
                # Alignment is checked, not assumed: the three merges run
                # independently, and a half-pixel drift would silently pair each
                # cell's height with its neighbour's returns.
                if (evidence.crs != dsm.crs or evidence.transform != dsm.transform
                        or evidence.shape != dsm.shape):
                    raise ValueError("Evidence raster is not aligned with the DSM/DTM.")
                if evidence.count != len(EVIDENCE_BAND_NAMES):
                    raise ValueError(
                        f"Evidence raster has {evidence.count} bands, expected "
                        f"{len(EVIDENCE_BAND_NAMES)} ({', '.join(EVIDENCE_BAND_NAMES)})"
                    )

            meta = dsm.meta.copy()
            meta.update({
                "count": NDSM_BAND_COUNT if write_evidence else 1,
                "dtype": rasterio.float32,
                "nodata": nodata_value,
                # Use tiling and compression to keep IO efficient and file sizes reasonable
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                "compress": "deflate",
                "predictor": 3,
            })

            # Enable BigTIFF to support very large rasters
            with rasterio.open(out_path, 'w', BIGTIFF='YES', **meta) as dst:
                for _, window in dst.block_windows(1):
                    # Read masked arrays so nodata is already masked
                    d = dsm.read(1, window=window, masked=True).astype(np.float32, copy=False)
                    g = dtm.read(1, window=window, masked=True).astype(np.float32, copy=False)

                    # Subtract; mask propagates automatically
                    nd = d - g

                    if evidence is None:
                        out = nd.filled(nodata_value).astype(np.float32, copy=False)
                        dst.write(out, 1, window=window)
                        continue

                    ev = evidence.read(window=window, masked=True).astype(np.float32, copy=False)
                    valid = ~np.ma.getmaskarray(nd)
                    valid &= ~np.ma.getmaskarray(ev).any(axis=0)

                    out = np.full((NDSM_BAND_COUNT,) + nd.shape, nodata_value,
                                  dtype=np.float32)
                    out[0][valid] = np.ma.getdata(nd)[valid]
                    out[1:, valid] = np.ma.getdata(ev)[:, valid]
                    dst.write(out, window=window)
        finally:
            if evidence is not None:
                evidence.close()

    if write_evidence:
        print(f"Created {NDSM_BAND_COUNT}-band nDSM (height + "
              f"{', '.join(EVIDENCE_BAND_NAMES)}): {out_path}")
    else:
        print(f"Created initial normalized DSM (nDSM): {out_path}")
    return out_path


# =========================
# CRS-aware crop
# =========================
def crop_geotiff_by_vertices_exact(input_path, output_path, wgs84_vertices, pad_m=0.0, use_mask=False):
    """
    Crop a GeoTIFF by transforming a WGS84 polygon into the raster CRS.
    pad_m: buffer in meters in raster CRS. use_mask=True for polygon mask.
    """
    if wgs84_vertices[0] != wgs84_vertices[-1]:
        wgs84_vertices = list(wgs84_vertices) + [wgs84_vertices[0]]

    poly_w84 = Polygon(wgs84_vertices)

    with rasterio.open(input_path) as src:
        if src.crs is None:
            raise ValueError("Input GeoTIFF has no CRS; cannot crop by WGS84 vertices.")

        to_src = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform
        poly_src = shp_transform(to_src, poly_w84)
        if poly_src.is_empty:
            raise ValueError("Projected polygon is empty; check CRS/vertices.")

        if pad_m:
            poly_src = poly_src.buffer(pad_m)

        if use_mask:
            out_img, out_transform = rio_mask(src, [mapping(poly_src)], crop=True, nodata=src.nodata)
            meta = src.meta.copy()
            meta.update({"height": out_img.shape[1], "width": out_img.shape[2], "transform": out_transform})
            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(out_img)
        else:
            minx, miny, maxx, maxy = poly_src.bounds
            win = from_bounds(minx, miny, maxx, maxy, src.transform)
            win = win.round_offsets().round_lengths()

            col_off = int(max(0, min(src.width,  win.col_off)))
            row_off = int(max(0, min(src.height, win.row_off)))
            width   = int(max(1, min(src.width  - col_off, win.width)))
            height  = int(max(1, min(src.height - row_off, win.height)))

            window = Window(col_off, row_off, width, height)
            data = src.read(window=window)
            transform = src.window_transform(window)

            meta = src.meta.copy()
            meta.update({"height": height, "width": width, "transform": transform})
            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(data)

    print(f"Cropped GeoTIFF saved to {output_path}")
    return output_path


# =========================
# Rectangle grid sampler
# =========================
def create_height_grid_from_geotiff_rectangle(tiff_path, mesh_size_m, rectangle_vertices):
    """
    Sample raster heights on a regular meter grid inside a WGS84 rectangle.

    Returns
    -------
    grid : np.ndarray, shape (ny, nx), dtype float32
        Heights in meters with NaN for NoData. Orientation is **north-up**
        (row 0 = north). Plot with `imshow(..., origin='upper')`.
    """
    # Close polygon if needed
    if rectangle_vertices[0] != rectangle_vertices[-1]:
        rectangle_vertices = list(rectangle_vertices) + [rectangle_vertices[0]]
    poly_w84 = Polygon(rectangle_vertices)
    if poly_w84.is_empty:
        raise ValueError("Input rectangle is empty/invalid.")

    with rasterio.open(tiff_path) as src:
        if src.crs is None:
            raise ValueError("Raster has no CRS; cannot sample by WGS84 rectangle.")

        # Project WGS84 rectangle into raster CRS
        to_src = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform
        poly_src = shp_transform(to_src, poly_w84)
        if poly_src.is_empty:
            raise ValueError("Projected rectangle is empty; check vertices/CRS.")

        minx, miny, maxx, maxy = poly_src.bounds
        if not (np.isfinite(minx) and np.isfinite(miny) and np.isfinite(maxx) and np.isfinite(maxy)):
            raise ValueError("Non-finite projected bounds.")

        # Decide grid size in meters using geodesic distances on original WGS84 bounds
        l, b, r, t = poly_w84.bounds
        geod = Geod(ellps="WGS84")
        _, _, width_m  = geod.inv(l, b, r, b)
        _, _, height_m = geod.inv(l, b, l, t)

        nx = max(1, int(np.round(width_m  / mesh_size_m)))
        ny = max(1, int(np.round(height_m / mesh_size_m)))

        # Sample **cell centers** in raster CRS to get exactly nx×ny, north->south in Y
        dx = (maxx - minx) / nx
        dy = (maxy - miny) / ny
        xs = minx + (0.5 + np.arange(nx)) * dx              # west -> east
        ys = maxy - (0.5 + np.arange(ny)) * dy              # north -> south (keep north-up)
        XX, YY = np.meshgrid(xs, ys)                        # (ny, nx)

        # ---- FIX: flatten before src.index, then reshape ----
        xq = XX.ravel()
        yq = YY.ravel()
        rows_flat, cols_flat = src.index(xq, yq)            # returns sequences
        rows_flat = np.asarray(rows_flat)
        cols_flat = np.asarray(cols_flat)

        # Valid pixels
        valid = (
            (rows_flat >= 0) & (rows_flat < src.height) &
            (cols_flat >= 0) & (cols_flat < src.width)
        )

        band = src.read(1)
        nodata = src.nodata

        vals = np.full(xq.shape, np.nan, dtype=np.float32)
        if np.any(valid):
            v = band[rows_flat[valid], cols_flat[valid]].astype(np.float32, copy=False)
            if nodata is not None:
                v = np.where(v == nodata, np.nan, v)
            vals[valid] = v

        grid = vals.reshape(ny, nx)  # already north-up due to ys decreasing

        # Debug info
        print("GeoTIFF info:")
        print(f"  NoData value: {nodata}")
        if np.isfinite(grid).any():
            vv = grid[np.isfinite(grid)]
            print(f"  Data range (excl NoData): {float(vv.min()):.3f} to {float(vv.max()):.3f}")
        else:
            print("  Data range: (no valid values in requested rectangle)")

        # Convert to VoxCity 2D grid convention (south-up): row 0 = south
        return np.flipud(grid)

def visualize_height_grid(
    grid,
    title="nDSM Heights",
    cmap="viridis",
    vmin=None,
    vmax=None,
    robust=False,          # if True and vmin/vmax not given, use percentiles
    pr=(2, 98)             # percentiles for robust bounds
):
    masked = np.ma.masked_invalid(grid)

    # Compute robust bounds only if asked and not explicitly provided
    valid = grid[np.isfinite(grid)]
    if robust and valid.size:
        if vmin is None: vmin = np.percentile(valid, pr[0])
        if vmax is None: vmax = np.percentile(valid, pr[1])

    # Safety: ensure vmin <= vmax if both set
    if (vmin is not None) and (vmax is not None) and (vmin > vmax):
        vmin, vmax = vmax, vmin

    plt.figure(figsize=(8, 7))
    img = plt.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)

    # Show arrows on the colorbar if values are clipped beyond vmin/vmax
    extend = 'neither'
    if valid.size and (vmin is not None or vmax is not None):
        below = (vmin is not None) and (valid.min() < vmin)
        above = (vmax is not None) and (valid.max() > vmax)
        extend = 'both' if (below and above) else ('min' if below else ('max' if above else 'neither'))

    plt.colorbar(img, label='Height (m)', extend=extend)
    plt.title(title)
    plt.xlabel('Cell X'); plt.ylabel('Cell Y')

    if valid.size:
        txt = (
            f"Grid: {grid.shape[1]}×{grid.shape[0]}\n"
            f"Valid: {valid.size} / {grid.size} ({100*valid.size/grid.size:.1f}%)\n"
            f"Data range: {valid.min():.2f}–{valid.max():.2f} m\n"
            f"Shown range: "
            f"{(vmin if vmin is not None else 'auto')}–{(vmax if vmax is not None else 'auto')} m\n"
            f"Mean: {valid.mean():.2f} m"
        )
        plt.annotate(txt, xy=(0.02, 0.02), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85), fontsize=9)

    plt.tight_layout()
    plt.show()


# =========================
# Orchestrator
# =========================
def get_ndsm_geotiff_from_tokyo_dsm(rectangle_vertices, las_dir="data/tokyo_las", output_dir='output', geotiff_name='ndsm.tif',
                                    resolution=0.5, crop_pad_m=2.0, use_polygon_mask=False,
                                    with_evidence=True):
    """
    Full pipeline: tiles -> LAS -> DSM/DTM -> merge (to target CRS) -> nDSM -> CRS-aware crop.
    Returns path to cropped nDSM GeoTIFF.

    *with_evidence* (default on) produces the six-band raster the runtime
    classifier reads. ``with_evidence=False`` restores the single-band writer
    verbatim, for comparison runs against the raster currently in production.
    """
    # base_url = "https://gic-tokyo.s3.ap-northeast-1.amazonaws.com/2024/dig/Vectortile/23ku/lp/{z}/{x}/{y}.pbf"

    # Choose target CRS from AOI longitudes (central Tokyo → EPSG:6677)
    lons = [v[0] for v in rectangle_vertices]
    target_crs = choose_jprcs_epsg(lons)

    # IO layout
    os.makedirs(output_dir, exist_ok=True)
    dsm_dir         = f"{output_dir}/dsm_geotiffs"
    dtm_dir         = f"{output_dir}/dtm_geotiffs"
    ev_dir          = f"{output_dir}/evidence_geotiffs"
    merged_dsm_file = f"{output_dir}/merged_dsm.tif"
    merged_dtm_file = f"{output_dir}/merged_dtm.tif"
    merged_ev_file  = f"{output_dir}/merged_evidence.tif"
    merged_ndsm     = f"{output_dir}/merged_ndsm.tif"
    final_ndsm      = f"{output_dir}/{geotiff_name}"
    os.makedirs(dsm_dir, exist_ok=True)
    os.makedirs(dtm_dir, exist_ok=True)
    if with_evidence:
        os.makedirs(ev_dir, exist_ok=True)

    # # Step 1: Tiles
    # print("Step 1: Downloading LAS data tiles...")
    # zoom_levels = [18]
    # download_tiles(base_url, bounds, zoom_levels, tile_dir)

    # # Step 2: Extract
    # print("\nStep 2: Extracting ZIP files...")
    # extract_all_zips(tile_dir, extract_dir)

    # Step 3: Find LAS
    print("\nStep 3: Finding LAS files...")
    las_files = find_las_files(las_dir)
    print(f"Found {len(las_files)} LAS files")
    if not las_files:
        raise FileNotFoundError(f"No LAS/LAZ files found under '{las_dir}'")

    # New: filter LAS files by AOI intersection in target CRS
    print("Filtering LAS files by AOI intersection...")
    las_files = filter_las_files_by_aoi(las_files, rectangle_vertices, target_crs, pad_m=crop_pad_m)
    if not las_files:
        raise FileNotFoundError("No LAS/LAZ files intersect the specified rectangle.")

    # Step 4: LAS→DSM/DTM (write with target_crs)
    print("\nStep 4: Processing LAS files to create DSM and DTM GeoTIFFs...")
    if with_evidence:
        dsm_list, dtm_list, ev_list = process_las_files(
            las_files, dsm_dir, dtm_dir, resolution, target_crs,
            evidence_output_dir=ev_dir,
        )
    else:
        dsm_list, dtm_list = process_las_files(las_files, dsm_dir, dtm_dir, resolution, target_crs)
        ev_list = []

    # Step 5 & 6: Merge (reproject to target_crs)
    print("\nStep 5: Merging DSM GeoTIFFs...")
    merged_dsm_path = merge_geotiffs(dsm_list, merged_dsm_file, target_crs) if dsm_list else None

    print("\nStep 6: Merging DTM GeoTIFFs...")
    merged_dtm_path = merge_geotiffs(dtm_list, merged_dtm_file, target_crs) if dtm_list else None

    merged_ev_path = None
    if ev_list:
        print("\nStep 6b: Merging evidence GeoTIFFs...")
        merged_ev_path = merge_geotiffs(ev_list, merged_ev_file, target_crs)

    # Step 7: nDSM
    print("\nStep 7: Creating normalized Digital Surface Model (nDSM)...")
    if merged_dsm_path and merged_dtm_path:
        build_ndsm(merged_dsm_path, merged_dtm_path, merged_ndsm,
                   evidence_path=merged_ev_path)
    else:
        raise RuntimeError("Both DSM and DTM are required to create nDSM")

    # Step 8: CRS-aware crop
    print("\nStep 8: Cropping nDSM to exact rectangle vertices...")
    print(f"Using rectangle vertices: {rectangle_vertices}")
    crop_geotiff_by_vertices_exact(
        merged_ndsm, final_ndsm,
        rectangle_vertices, pad_m=crop_pad_m, use_mask=use_polygon_mask
    )
    print(f"Successfully created and cropped nDSM: {final_ndsm}")
    return final_ndsm

def get_ndsm_grid(rectangle_vertices, meshsize_m, source, output_dir, **kwargs):
    if source == 'tokyo_dsm':
        las_dir = kwargs.get('las_dir', 'data/tokyo_las')
        tiff_path = get_ndsm_geotiff_from_tokyo_dsm(
            rectangle_vertices,
            las_dir=las_dir,
            output_dir=output_dir,
            geotiff_name='ndsm.tif'
        )
    else:
        tiff_path = kwargs.get('tiff_path')
        if not tiff_path:
            raise ValueError("Provide 'tiff_path' or use source='tokyo_dsm'.")
    grid = create_height_grid_from_geotiff_rectangle(tiff_path, meshsize_m, rectangle_vertices)
    return grid

 

def _resize_nearest_centered(arr: np.ndarray, new_shape):
    """Center-aligned nearest-neighbor resize for 2D arrays."""
    H, W = arr.shape
    Hn, Wn = new_shape
    # sample original index positions (0..H-1) at centers of new pixels
    r_idx = np.clip(np.round(np.linspace(0, H-1, Hn)).astype(int), 0, H-1)
    c_idx = np.clip(np.round(np.linspace(0, W-1, Wn)).astype(int), 0, W-1)
    return arr[np.ix_(r_idx, c_idx)]

def align_ndsm_to_landcover(
    ndsm_grid: np.ndarray,
    land_cover_grid: np.ndarray,
    *,
    tree_value=None,          # optional; only used if try_vertical_flip=True
    allow_resample: bool = True,
    try_vertical_flip: bool = False
):
    """
    Make nDSM align with land_cover grid. By default:
      - NO vertical flip is performed
      - Only a center-aligned nearest resample is done if shapes differ
    """
    nd = np.asarray(ndsm_grid)
    lc = np.asarray(land_cover_grid)

    info = {"resampled": False, "vertical_flipped": False}

    # 1) size-align (no flipping)
    if nd.shape != lc.shape:
        if not allow_resample:
            raise ValueError(f"Shape mismatch: nDSM {nd.shape} vs land_cover {lc.shape}")
        nd = _resize_nearest_centered(nd, lc.shape)
        info["resampled"] = True

    return nd, info

def _tree_mask_from_value(land_cover_grid: np.ndarray, tree_value):
    """
    Robust 'Tree' mask:
      - supports scalar (e.g. 'Tree' or 7) or an iterable of values
      - case-insensitive when comparing strings
    """
    def _as_set(x):
        if isinstance(x, (list, tuple, set)):
            return set(x)
        return {x}

    values = _as_set(tree_value)

    if land_cover_grid.dtype.kind in ("U", "S", "O"):
        # Compare as lowercase strings
        lc = land_cover_grid.astype("U")
        lc = np.char.casefold(lc)
        vals = {str(v).casefold() for v in values}
        mask = np.isin(lc, list(vals))
    else:
        mask = np.isin(land_cover_grid, list(values))
    return mask

def build_canopy_from_ndsm(
    ndsm_grid: np.ndarray,
    land_cover_grid: np.ndarray,
    tree_value,
    non_tree_fill=np.nan,
    clamp_negative_to_zero: bool = True
):
    """
    Base canopy from nDSM: keep heights only at tree cells; elsewhere 'non_tree_fill'.
    Assumes ndsm_grid already aligned & oriented with land_cover_grid.
    """
    if ndsm_grid.shape != land_cover_grid.shape:
        raise ValueError(f"Shape mismatch: nDSM {ndsm_grid.shape} vs land_cover {land_cover_grid.shape}")

    canopy = np.full(ndsm_grid.shape, non_tree_fill, dtype=float)
    tree_mask = _tree_mask_from_value(land_cover_grid, tree_value)

    ndsm = ndsm_grid.astype(float, copy=False)
    if clamp_negative_to_zero:
        ndsm = np.where(np.isnan(ndsm), np.nan, np.maximum(ndsm, 0.0))

    valid_tree = tree_mask & ~np.isnan(ndsm)
    canopy[valid_tree] = ndsm[valid_tree]
    return canopy


def infill_canopy_nearest_average(
    canopy_grid: np.ndarray,
    land_cover_grid: np.ndarray,
    tree_value,
    consider_zeros_missing: bool = False,  # treat zeros as missing too?
    k_for_ties: int = 8,
    atol: float = 1e-9,
    max_radius=None,
    verbose: bool = True
):
    """
    Fill missing tree cells in 'canopy_grid' from the nearest tree cells with valid heights.
    - If multiple nearest with equal distance, average them.
    - Uses SciPy KDTree if available; falls back to NumPy ring expansion.
    """
    canopy = canopy_grid.astype(float, copy=True)
    H, W = canopy.shape
    tree_mask = _tree_mask_from_value(land_cover_grid, tree_value)

    if consider_zeros_missing:
        miss_mask = tree_mask & (np.isnan(canopy) | (canopy == 0))
        src_mask  = tree_mask & ~np.isnan(canopy) & (canopy != 0)
    else:
        miss_mask = tree_mask & np.isnan(canopy)
        src_mask  = tree_mask & ~np.isnan(canopy)

    n_missing = int(miss_mask.sum())
    if verbose:
        print(f"[infill] tree cells missing: {n_missing}")
    if n_missing == 0:
        return canopy

    src_rows, src_cols = np.nonzero(src_mask)
    if src_rows.size == 0:
        raise ValueError("No tree cells with valid nDSM to infill from.")
    src_vals = canopy[src_rows, src_cols]

    # --- Fast path: SciPy KDTree ---
    try:
        from scipy.spatial import cKDTree
        tgt_rows, tgt_cols = np.nonzero(miss_mask)

        tree = cKDTree(np.c_[src_rows, src_cols])
        k = min(int(k_for_ties), src_rows.size)

        dists, idxs = tree.query(np.c_[tgt_rows, tgt_cols], k=k, workers=-1)

        if k == 1:
            dists = dists[:, None]
            idxs  = idxs[:, None]

        mins = dists[:, 0]
        eq = np.isclose(dists, mins[:, None], rtol=0.0, atol=atol)

        for i, (r, c) in enumerate(zip(tgt_rows, tgt_cols)):
            sel = eq[i]
            if not sel.any():
                canopy[r, c] = src_vals[idxs[i, 0]]
            else:
                canopy[r, c] = float(np.mean(src_vals[idxs[i, sel]]))

        if verbose:
            print("[infill] method: SciPy KDTree, Euclidean nearest with tie-averaging")
        return canopy

    except Exception as e:
        if verbose:
            print(f"[infill] KDTree unavailable ({e}); using pure-NumPy ring search (Chebyshev distance).")

    # --- Fallback: NumPy ring search ---
    src_mask = src_mask.astype(bool)
    if max_radius is None:
        max_radius = max(H, W)

    tgt_rows, tgt_cols = np.nonzero(miss_mask)
    for r, c in zip(tgt_rows, tgt_cols):
        filled = False
        for rad in range(1, max_radius + 1):
            r0, r1 = max(0, r - rad), min(H - 1, r + rad)
            c0, c1 = max(0, c - rad), min(W - 1, c + rad)

            coords = []
            coords.extend([(r0, cc) for cc in range(c0, c1 + 1)])             # top
            if r1 != r0:
                coords.extend([(r1, cc) for cc in range(c0, c1 + 1)])         # bottom
            for rr in range(r0 + 1, r1):                                      # sides
                coords.append((rr, c0))
                if c1 != c0:
                    coords.append((rr, c1))

            if not coords:
                continue

            rr = np.fromiter((p[0] for p in coords), dtype=int)
            cc = np.fromiter((p[1] for p in coords), dtype=int)

            ring_src = src_mask[rr, cc]
            if not ring_src.any():
                continue

            rr = rr[ring_src]
            cc = cc[ring_src]
            vals = canopy[rr, cc]

            dr = rr - r
            dc = cc - c
            d2 = dr * dr + dc * dc
            dmin = d2.min()
            sel = (d2 == dmin)

            canopy[r, c] = float(np.mean(vals[sel]))
            filled = True
            break

        if not filled:
            # leave as NaN
            pass

    if verbose:
        print("[infill] method: NumPy ring search, Chebyshev/Euclidean-on-ring tie-averaging")
    return canopy


def summarize_grid(name: str, grid: np.ndarray):
    if np.issubdtype(grid.dtype, np.floating):
        valid = ~np.isnan(grid)
        if not valid.any():
            print(f"{name}: shape={grid.shape}, no valid cells")
            return
        print(f"{name}: shape={grid.shape}, min={np.nanmin(grid):.2f}, "
              f"mean={np.nanmean(grid):.2f}, max={np.nanmax(grid):.2f}, "
              f"valid={int(valid.sum())}/{grid.size}")
    else:
        # For non-float grids (e.g., land-cover labels)
        print(f"{name}: shape={grid.shape}, dtype={grid.dtype}")

import numpy as np

def _resize_nearest(arr: np.ndarray, new_shape):
    r_idx = np.clip(np.round(np.linspace(0, arr.shape[0]-1, new_shape[0])).astype(int), 0, arr.shape[0]-1)
    c_idx = np.clip(np.round(np.linspace(0, arr.shape[1]-1, new_shape[1])).astype(int), 0, arr.shape[1]-1)
    return arr[np.ix_(r_idx, c_idx)]

def fill_canopy_gaps_with_nearest(
    canopy_height_grid: np.ndarray,
    ndsm_grid: np.ndarray,
    land_cover_grid: np.ndarray,
    tree_value: int = 4,
    treat_zero_as_missing: bool = False,
    restrict_neighbors_to_tree: bool = False,
    allow_resample: bool = True,
    tie_tol: float = 1e-9,
    k_for_knn: int = 8,
    non_tree_fill: float = 0.0,          # ← ensure non-tree cells are this (default 0)
    max_neighbor_distance_m: float = None,  # if set, donors must be within this distance (meters)
    cell_size_m: float = 1.0,            # size of a grid cell (meters)
    fallback_tree_height_m: float = None # if set, tree cells still missing after fill get this value
) -> np.ndarray:
    """
    Fill missing canopy heights at tree cells using nearest valid nDSM cells.
    If multiple nearest neighbors are at the same distance, use their average.
    The returned grid has non-tree cells set to `non_tree_fill` (default 0).
    """
    # 1) Ensure shapes line up (we take land_cover as the reference)
    H, W = land_cover_grid.shape
    ndsm = ndsm_grid.astype(float)
    canopy = canopy_height_grid.astype(float)

    if ndsm.shape != (H, W):
        if not allow_resample:
            raise ValueError(f"nDSM shape {ndsm.shape} != land_cover shape {(H, W)} "
                             "(set allow_resample=True to resize)")
        ndsm = _resize_nearest(ndsm, (H, W))
    if canopy.shape != (H, W):
        if not allow_resample:
            raise ValueError(f"canopy shape {canopy.shape} != land_cover shape {(H, W)} "
                             "(set allow_resample=True to resize)")
        canopy = _resize_nearest(canopy, (H, W))

    # 2) Define masks
    tree_mask = (land_cover_grid == tree_value)

    if treat_zero_as_missing:
        missing = tree_mask & (np.isnan(canopy) | (canopy == 0))
    else:
        missing = tree_mask & np.isnan(canopy)

    # Donor (valid) nDSM cells
    valid_ndsm = ~np.isnan(ndsm)
    if restrict_neighbors_to_tree:
        valid_ndsm &= tree_mask

    # Nothing to fill → just enforce non-tree zeros and return
    if not np.any(missing):
        out = canopy.copy()
        out[~tree_mask] = non_tree_fill
        return out

    # 3) Try fast path: SciPy KDTree
    try:
        from scipy.spatial import cKDTree  # type: ignore
        donor_rows, donor_cols = np.where(valid_ndsm)
        if donor_rows.size == 0:
            out = canopy.copy()
            out[~tree_mask] = non_tree_fill
            return out

        donor_coords = np.column_stack([donor_rows, donor_cols]).astype(float)
        donor_vals = ndsm[donor_rows, donor_cols].astype(float)

        tree = cKDTree(donor_coords)

        miss_rows, miss_cols = np.where(missing)
        query_coords = np.column_stack([miss_rows, miss_cols]).astype(float)

        # Query k nearest (sorted), then average ties at min distance
        dists, idxs = tree.query(query_coords, k=min(k_for_knn, donor_coords.shape[0]))
        if idxs.ndim == 1:  # k==1 case
            dists = dists[:, None]
            idxs = idxs[:, None]

        filled_vals = np.full(miss_rows.shape, np.nan, dtype=float)
        for i in range(query_coords.shape[0]):
            di = dists[i]
            ii = idxs[i]
            finite = np.isfinite(di)
            if not np.any(finite):
                continue
            di = di[finite]
            ii = ii[finite]

            # Apply distance threshold in meters if requested
            if max_neighbor_distance_m is not None:
                allowed = (di * float(cell_size_m)) <= float(max_neighbor_distance_m)
                if not np.any(allowed):
                    continue
                di = di[allowed]
                ii = ii[allowed]

            d0 = di[0]
            tie_mask = np.abs(di - d0) <= tie_tol
            tied_vals = donor_vals[ii[tie_mask]]
            if tied_vals.size > 0:
                filled_vals[i] = float(np.mean(tied_vals))

        out = canopy.copy()
        out[miss_rows, miss_cols] = np.where(np.isnan(filled_vals), out[miss_rows, miss_cols], filled_vals)

        # **Enforce:** non-tree cells = 0 (or non_tree_fill)
        out[~tree_mask] = non_tree_fill

        # Fallback: assign static height to any remaining missing tree cells
        if fallback_tree_height_m is not None:
            still_missing = tree_mask & np.isnan(out)
            if np.any(still_missing):
                out[still_missing] = float(fallback_tree_height_m)
        return out

    except Exception:
        # 4) Fallback: ring search (pure NumPy). Slower, but robust.
        out = canopy.copy()

        donor_mask = valid_ndsm
        if not np.any(donor_mask):
            out[~tree_mask] = non_tree_fill
            return out

        # Limit search radius by max_neighbor_distance_m if provided
        max_r_cells = max(H, W)
        if max_neighbor_distance_m is not None:
            try:
                max_r_cells = max(1, int(math.floor(float(max_neighbor_distance_m) / float(cell_size_m))))
            except Exception:
                max_r_cells = max(H, W)

        for (r0, c0) in zip(*np.where(missing)):
            if not np.isnan(out[r0, c0]):
                continue

            found_val = np.nan
            for rad in range(1, max_r_cells + 1):
                r1 = max(0, r0 - rad); r2 = min(H - 1, r0 + rad)
                c1 = max(0, c0 - rad); c2 = min(W - 1, c0 + rad)

                submask = donor_mask[r1:r2+1, c1:c2+1]
                if not np.any(submask):
                    continue

                rr, cc = np.where(submask)
                rr_abs = rr + r1; cc_abs = cc + c1
                dr = rr_abs - r0; dc = cc_abs - c0
                d2 = dr*dr + dc*dc
                if d2.size == 0:
                    continue
                min_d2 = np.min(d2)
                tie = (d2 == min_d2)
                vals = ndsm[rr_abs[tie], cc_abs[tie]]
                if vals.size > 0:
                    found_val = float(np.mean(vals))
                    break

            if not np.isnan(found_val):
                out[r0, c0] = found_val

        # **Enforce:** non-tree cells = 0 (or non_tree_fill)
        out[~(land_cover_grid == tree_value)] = non_tree_fill

        # Fallback: assign static height to any remaining missing tree cells
        if fallback_tree_height_m is not None:
            tree_mask_local = (land_cover_grid == tree_value)
            still_missing = tree_mask_local & np.isnan(out)
            if np.any(still_missing):
                out[still_missing] = float(fallback_tree_height_m)
        return out

def build_canopy_height_grid(
    ndsm_grid: np.ndarray,
    land_cover_grid: np.ndarray,
    tree_value: int = 4,
    non_tree_fill: float = 0.0,
    clamp_negative_to_zero: bool = True,
    allow_resample: bool = True
) -> np.ndarray:
    """
    Create canopy_height_grid by copying nDSM heights where land_cover == tree_value.
    Non-tree cells are set to `non_tree_fill` (default 0).
    """
    if ndsm_grid.shape != land_cover_grid.shape:
        if not allow_resample:
            raise ValueError(f"Shape mismatch: nDSM {ndsm_grid.shape} vs land_cover {land_cover_grid.shape}. "
                             "Pass allow_resample=True to resize nDSM to land_cover shape.")
        ndsm_grid = _resize_nearest(ndsm_grid, land_cover_grid.shape)

    ndsm = ndsm_grid.copy()

    if clamp_negative_to_zero:
        ndsm = np.where(np.isnan(ndsm), np.nan, np.maximum(ndsm, 0.0))

    tree_mask = (land_cover_grid == tree_value)

    canopy = np.full_like(ndsm, non_tree_fill, dtype=float)  # non-tree -> 0
    valid = ~np.isnan(ndsm)
    canopy[tree_mask & valid] = ndsm[tree_mask & valid]
    return canopy

def _binary_dilation_square(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    """Binary dilation with a square structuring element without wrap-around."""
    if radius_cells <= 0:
        return mask.astype(bool)
    m = mask.astype(bool)
    H, W = m.shape
    pad = int(radius_cells)
    padded = np.pad(m, pad_width=pad, mode='constant', constant_values=False)
    out_core = np.zeros((H, W), dtype=bool)
    for dr in range(-pad, pad + 1):
        for dc in range(-pad, pad + 1):
            out_core |= padded[pad + dr: pad + dr + H, pad + dc: pad + dc + W]
    return out_core

def remove_local_spikes_in_canopy(
    canopy_grid: np.ndarray,
    land_cover_grid: np.ndarray,
    *,
    tree_value,
    building_value,
    high_threshold_m: float = 10.0,
    min_adjacent_tree_neighbors: int = 4,   # "more than 2"
    building_buffer_m: float = 15.0,
    cell_size_m: float = 1.0,
    replacement_tree_height_m: float = None
) -> np.ndarray:
    """
    Remove canopy spikes:
      - Any tree cell with height > high_threshold_m must have at least
        `min_adjacent_tree_neighbors` tree neighbors (8-neighborhood).
      - Suppress tree heights > high_threshold_m within `building_buffer_m`
        of building cells.
    Spikes are set to NaN.
    """
    can = canopy_grid.astype(float).copy()

    tree_mask = _tree_mask_from_value(land_cover_grid, tree_value)
    bld_mask = _tree_mask_from_value(land_cover_grid, building_value)

    # 1) Neighbor tree count (8-connectivity)
    # Neighbor count (8-connectivity) without wrap-around
    H, W = tree_mask.shape
    padded = np.pad(tree_mask.astype(np.uint8), pad_width=1, mode='constant', constant_values=0)
    neighbor_count = np.zeros((H, W), dtype=np.int32)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            neighbor_count += padded[1 + dr: 1 + dr + H, 1 + dc: 1 + dc + W].astype(np.int32)

    high = (tree_mask) & np.isfinite(can) & (can > float(high_threshold_m))
    insufficient_neighbors = high & (neighbor_count < int(min_adjacent_tree_neighbors))

    # 2) Proximity to buildings
    radius_cells = int(max(0, math.ceil(float(building_buffer_m) / float(cell_size_m))))
    near_building = _binary_dilation_square(bld_mask, radius_cells)
    high_near_building = high & near_building

    to_suppress = insufficient_neighbors | high_near_building
    if replacement_tree_height_m is None:
        can[to_suppress] = np.nan
    else:
        can[to_suppress] = float(replacement_tree_height_m)
    return can

 