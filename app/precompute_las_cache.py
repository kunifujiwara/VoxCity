"""
Precompute LAS-derived rasters for fast reuse:
 - Per-tile DSM/DTM (GeoTIFF)
 - Merged DSM/DTM (target CRS)
 - nDSM (DSM-DTM), optionally as COG
 - Optional crop to AOI

Usage examples:
  python app/precompute_las_cache.py --las-dir app/data/tokyo_las \
      --output-dir app/data/temp --aoi 139.70 35.63 139.78 35.69 --resolution 0.5 --cog
"""

import os
from pathlib import Path
from typing import List, Optional

import typer
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
import rasterio

from tokyo_las import (
    choose_jprcs_epsg,
    find_las_files,
    filter_las_files_by_aoi,
    process_las_files,
    merge_geotiffs,
    build_ndsm,
    crop_geotiff_by_vertices_exact,
)

app = typer.Typer(add_completion=False, help="Precompute LAS-derived caches (DSM/DTM/nDSM)")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@app.command()
def main(
    las_dir: Path = typer.Option(Path("app/data/tokyo_las"), exists=True, file_okay=False, dir_okay=True),
    output_dir: Path = typer.Option(Path("app/data/temp"), help="Base output dir (nDSM saved under here)"),
    aoi: Optional[List[float]] = typer.Option(None, help="min_lon min_lat max_lon max_lat"),
    resolution: float = typer.Option(0.5, help="DSM/DTM raster resolution (m)"),
    crop_pad_m: float = typer.Option(2.0, help="Buffer around AOI when cropping (m)"),
    use_mask: bool = typer.Option(False, help="Use polygon mask when cropping (slower)"),
    cog: bool = typer.Option(False, help="Write nDSM as Cloud-Optimized GeoTIFF (COG)"),
):
    _ensure_dir(output_dir)

    if aoi and len(aoi) != 4:
        typer.echo("--aoi expects 4 numbers: min_lon min_lat max_lon max_lat", err=True)
        raise typer.Exit(code=2)

    rectangle_vertices = None
    if aoi:
        min_lon, min_lat, max_lon, max_lat = aoi
        rectangle_vertices = [
            (min_lon, min_lat), (min_lon, max_lat), (max_lon, max_lat), (max_lon, min_lat)
        ]

    # Choose target CRS from AOI (Tokyo defaults work well when unspecified)
    if rectangle_vertices:
        lons = [v[0] for v in rectangle_vertices]
        target_crs = choose_jprcs_epsg(lons)
    else:
        target_crs = "EPSG:6677"

    # Layout
    dsm_dir         = output_dir / "dsm_geotiffs"
    dtm_dir         = output_dir / "dtm_geotiffs"
    merged_dsm_file = output_dir / "merged_dsm.tif"
    merged_dtm_file = output_dir / "merged_dtm.tif"
    merged_ndsm     = output_dir / "merged_ndsm.tif"
    final_ndsm      = output_dir / "ndsm.tif"
    _ensure_dir(dsm_dir); _ensure_dir(dtm_dir)

    # Gather LAS files and optionally filter by AOI
    las_files = find_las_files(str(las_dir))
    if not las_files:
        typer.echo(f"No LAS/LAZ found in {las_dir}")
        raise typer.Exit(code=1)
    if rectangle_vertices:
        las_files = filter_las_files_by_aoi(las_files, rectangle_vertices, target_crs, pad_m=crop_pad_m)
        if not las_files:
            typer.echo("No LAS files intersect AOI")
            raise typer.Exit(code=1)

    # Per-file DSM/DTM
    dsm_list, dtm_list = process_las_files(las_files, str(dsm_dir), str(dtm_dir), resolution, target_crs)

    # Merge
    merged_dsm_path = merge_geotiffs(dsm_list, str(merged_dsm_file), target_crs) if dsm_list else None
    merged_dtm_path = merge_geotiffs(dtm_list, str(merged_dtm_file), target_crs) if dtm_list else None
    if not (merged_dsm_path and merged_dtm_path):
        typer.echo("Missing merged DSM/DTM; cannot create nDSM")
        raise typer.Exit(code=1)

    # nDSM
    build_ndsm(str(merged_dsm_file), str(merged_dtm_file), str(merged_ndsm))

    # Optional crop to AOI
    if rectangle_vertices:
        crop_geotiff_by_vertices_exact(
            str(merged_ndsm), str(final_ndsm), rectangle_vertices, pad_m=crop_pad_m, use_mask=use_mask
        )
    else:
        # Save uncropped as final
        with rasterio.open(str(merged_ndsm)) as src:
            profile = src.profile.copy()
            data = src.read()
        with rasterio.open(str(final_ndsm), 'w', **profile) as dst:
            dst.write(data)

    # Optional COG conversion of final nDSM
    if cog:
        dst_cog = output_dir / "ndsm_cog.tif"
        profile = cog_profiles.get("deflate")
        cog_translate(
            str(final_ndsm), str(dst_cog), profile,
            in_memory=False, quiet=True
        )
        typer.echo(f"COG written: {dst_cog}")

    typer.echo(f"nDSM ready: {final_ndsm}")


if __name__ == "__main__":
    app()


