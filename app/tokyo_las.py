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

import os, re, math, zipfile
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
 

import laspy
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
    to_target = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True).transform
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
            if las_crs is not None and str(las_crs) != str(target_crs):
                to_target_from_las = Transformer.from_crs(las_crs, target_crs, always_xy=True).transform
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


# =========================
# LAS → DSM / DTM (fast)
# =========================
def process_las_to_raster(las_path, resolution=0.5, dsm_classes=(1, 3), dtm_classes=(2,)):
    """
    Build DSM (max Z) and DTM (min Z) on a regular grid aligned to LAS extents.
    Returns dicts: {'array','transform','bounds','nodata'}
    """
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
        return dsm, dtm

    except Exception as e:
        print(f"Error processing LAS file {las_path}: {e}")
        return None, None

def save_raster(raster_data, output_path, crs):
    array = raster_data['array'].astype(np.float32, copy=False)
    transform = raster_data['transform']
    nodata = float(raster_data.get('nodata', -9999.0))

    profile = {
        'driver': 'GTiff',
        'height': array.shape[0],
        'width':  array.shape[1],
        'count':  1,
        'dtype':  rasterio.float32,
        'transform': transform,
        'nodata': nodata,
        'crs': crs
    }

    array = np.where(np.isnan(array), nodata, array).astype(np.float32, copy=False)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(array, 1)
    print(f"Saved raster to {output_path}")
    return True

def process_las_files(las_files, dsm_output_dir, dtm_output_dir, resolution, target_crs):
    os.makedirs(dsm_output_dir, exist_ok=True)
    os.makedirs(dtm_output_dir, exist_ok=True)

    dsm_files, dtm_files = [], []
    for i, las_path in enumerate(las_files):
        print(f"Processing file {i+1}/{len(las_files)}: {os.path.basename(las_path)}")
        dsm, dtm = process_las_to_raster(las_path, resolution=resolution)
        if dsm is None or dtm is None:
            continue
        base = os.path.splitext(os.path.basename(las_path))[0]
        dsm_path = os.path.join(dsm_output_dir, f"{base}_dsm.tif")
        dtm_path = os.path.join(dtm_output_dir, f"{base}_dtm.tif")
        if save_raster(dsm, dsm_path, target_crs):
            dsm_files.append(dsm_path)
        if save_raster(dtm, dtm_path, target_crs):
            dtm_files.append(dtm_path)
    print(f"Created {len(dsm_files)} DSM GeoTIFFs and {len(dtm_files)} DTM GeoTIFFs")
    return dsm_files, dtm_files


# =========================
# Merge & nDSM
# =========================
def merge_geotiffs(input_files, output_path, target_crs, nodata_value=-9999.0, res=None):
    """
    Merge multiple GeoTIFFs into one, pre-warping each to target_crs so we
    don’t rely on merge(dst_crs=...), which isn’t available in older rasterio.
    """
    if not input_files:
        print("No GeoTIFF files to merge")
        return None

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

        for fp in input_files:
            src = rasterio.open(fp)
            opened.append(src)
            if src.crs is None:
                raise ValueError(f"{fp} has no CRS; cannot reproject.")
            if str(src.crs) == str(target_crs):
                # Already in target CRS → still wrap so nodata/resampling are uniform
                vrt = WarpedVRT(
                    src,
                    crs=target_crs,
                    src_nodata=src.nodata,
                    dst_nodata=nodata_value,
                    resampling=Resampling.nearest,
                    resolution=res
                )
            else:
                vrt = WarpedVRT(
                    src,
                    crs=target_crs,
                    src_nodata=src.nodata,
                    dst_nodata=nodata_value,
                    resampling=Resampling.nearest,
                    resolution=res
                )
            warped.append(vrt)

        print(f"Target CRS: {target_crs}")
        print("Merging files...")
        # No dst_crs here — all inputs are already warped VRTs
        merged, transform = merge(warped, nodata=nodata_value, method="first")

        meta = {
            "driver": "GTiff",
            "height": merged.shape[1],
            "width": merged.shape[2],
            "count": merged.shape[0],
            "dtype": merged.dtype,
            "crs": target_crs,
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

def build_ndsm(dsm_path, dtm_path, out_path, nodata_value=-9999.0):
    with rasterio.open(dsm_path) as dsm, rasterio.open(dtm_path) as dtm:
        if dsm.crs != dtm.crs or dsm.transform != dtm.transform or dsm.shape != dtm.shape:
            raise ValueError("DSM and DTM are not perfectly aligned.")
        d = dsm.read(1).astype(np.float32)
        g = dtm.read(1).astype(np.float32)
        d[d == dsm.nodata] = np.nan
        g[g == dtm.nodata] = np.nan
        ndsm = d - g
        ndsm = np.where(np.isnan(ndsm), nodata_value, ndsm).astype(np.float32)

        meta = dsm.meta.copy()
        meta.update({"nodata": nodata_value, "dtype": rasterio.float32})
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(ndsm, 1)
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

        return grid

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
                                    resolution=0.5, crop_pad_m=2.0, use_polygon_mask=False):
    """
    Full pipeline: tiles -> LAS -> DSM/DTM -> merge (to target CRS) -> nDSM -> CRS-aware crop.
    Returns path to cropped nDSM GeoTIFF.
    """
    # base_url = "https://gic-tokyo.s3.ap-northeast-1.amazonaws.com/2024/dig/Vectortile/23ku/lp/{z}/{x}/{y}.pbf"

    # Choose target CRS from AOI longitudes (central Tokyo → EPSG:6677)
    lons = [v[0] for v in rectangle_vertices]
    target_crs = choose_jprcs_epsg(lons)

    # IO layout
    os.makedirs(output_dir, exist_ok=True)
    dsm_dir         = f"{output_dir}/dsm_geotiffs"
    dtm_dir         = f"{output_dir}/dtm_geotiffs"
    merged_dsm_file = f"{output_dir}/merged_dsm.tif"
    merged_dtm_file = f"{output_dir}/merged_dtm.tif"
    merged_ndsm     = f"{output_dir}/merged_ndsm.tif"
    final_ndsm      = f"{output_dir}/{geotiff_name}"
    os.makedirs(dsm_dir, exist_ok=True)
    os.makedirs(dtm_dir, exist_ok=True)

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
    dsm_list, dtm_list = process_las_files(las_files, dsm_dir, dtm_dir, resolution, target_crs)

    # Step 5 & 6: Merge (reproject to target_crs)
    print("\nStep 5: Merging DSM GeoTIFFs...")
    merged_dsm_path = merge_geotiffs(dsm_list, merged_dsm_file, target_crs) if dsm_list else None

    print("\nStep 6: Merging DTM GeoTIFFs...")
    merged_dtm_path = merge_geotiffs(dtm_list, merged_dtm_file, target_crs) if dtm_list else None

    # Step 7: nDSM
    print("\nStep 7: Creating normalized Digital Surface Model (nDSM)...")
    if merged_dsm_path and merged_dtm_path:
        build_ndsm(merged_dsm_path, merged_dtm_path, merged_ndsm)
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

    # 2) Optional: *explicitly* try a vertical flip heuristic
    if try_vertical_flip:
        # Heuristic uses valid-mask overlap ONLY when a tree mask is provided.
        if tree_value is not None:
            tree_mask = (lc == tree_value)
        else:
            # fall back to full domain (weak signal)
            tree_mask = np.ones_like(lc, dtype=bool)

        valid = ~np.isnan(nd)
        score_as_is = np.count_nonzero(valid & tree_mask)
        score_flip  = np.count_nonzero(np.flipud(valid) & tree_mask)

        if score_flip > score_as_is:
            nd = np.flipud(nd)
            info["vertical_flipped"] = True

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

            d0 = di[0]
            tie_mask = np.abs(di - d0) <= tie_tol
            tied_vals = donor_vals[ii[tie_mask]]
            if tied_vals.size > 0:
                filled_vals[i] = float(np.mean(tied_vals))

        out = canopy.copy()
        out[miss_rows, miss_cols] = np.where(np.isnan(filled_vals), out[miss_rows, miss_cols], filled_vals)

        # **Enforce:** non-tree cells = 0 (or non_tree_fill)
        out[~tree_mask] = non_tree_fill
        return out

    except Exception:
        # 4) Fallback: ring search (pure NumPy). Slower, but robust.
        out = canopy.copy()

        donor_mask = valid_ndsm
        if not np.any(donor_mask):
            out[~tree_mask] = non_tree_fill
            return out

        for (r0, c0) in zip(*np.where(missing)):
            if not np.isnan(out[r0, c0]):
                continue

            found_val = np.nan
            max_r = max(H, W)
            for rad in range(1, max_r):
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

 