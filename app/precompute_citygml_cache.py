"""
CLI to precompute and cache minimal datasets from PLATEAU CityGML:
 - Buildings: 2D footprint, building_id, height_m, ground_z, source_file
 - Terrain: points/polygons with elevation (optional export)

Outputs (configurable):
 - GeoParquet (.parquet): columnar, compressed
 - FlatGeobuf (.fgb): spatially indexed

This script reuses existing CityGML parsing utilities from voxcity and
is intended to run once to avoid repeated heavy XML parsing in workflows.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd
import pandas as pd
import typer
from shapely.geometry import Polygon

# Import internal utilities
from voxcity.downloader.citygml import (
    load_buid_dem_veg_from_citygml,
)

app = typer.Typer(add_completion=False, help="Precompute/cache CityGML-derived datasets")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _parse_rectangle(aoi: Optional[List[float]]) -> Optional[List[Tuple[float, float]]]:
    if aoi is None:
        return None
    if len(aoi) != 4:
        raise typer.BadParameter("--aoi expects 4 numbers: min_lon min_lat max_lon max_lat")
    min_lon, min_lat, max_lon, max_lat = aoi
    return [
        (min_lon, min_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
        (min_lon, max_lat),
        (min_lon, min_lat),
    ]


def _normalize_crs(gdf: Optional[gpd.GeoDataFrame], to_epsg: int = 4326) -> Optional[gpd.GeoDataFrame]:
    if gdf is None:
        return None
    try:
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=to_epsg)
        elif getattr(gdf.crs, 'to_epsg', lambda: None)() != to_epsg and gdf.crs != f"EPSG:{to_epsg}":
            gdf = gdf.to_crs(epsg=to_epsg)
    except Exception:
        pass
    return gdf


def _compute_minimal_building_schema(building_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    cols = []
    if 'building_id' in building_gdf.columns:
        cols.append('building_id')
    if 'height' in building_gdf.columns:
        cols.append('height')
    if 'ground_elevation' in building_gdf.columns:
        cols.append('ground_elevation')
    if 'source_file' in building_gdf.columns:
        cols.append('source_file')

    out = building_gdf[cols + ['geometry']].copy() if cols else building_gdf[['geometry']].copy()
    # Rename to canonical names
    if 'height' in out.columns:
        out = out.rename(columns={'height': 'height_m'})
    if 'ground_elevation' in out.columns:
        out = out.rename(columns={'ground_elevation': 'ground_z'})
    return out


def _maybe_clip_to_aoi(gdf: Optional[gpd.GeoDataFrame], rect: Optional[List[Tuple[float, float]]]) -> Optional[gpd.GeoDataFrame]:
    if gdf is None or rect is None:
        return gdf
    try:
        aoi_poly = Polygon(rect)
        return gdf[gdf.intersects(aoi_poly)].copy()
    except Exception:
        return gdf


def _find_dataset_roots(root: Path) -> List[Path]:
    """Recursively find dataset roots that contain an 'udx' directory."""
    roots: List[Path] = []
    for dirpath, dirnames, _ in os.walk(root):
        if 'udx' in dirnames:
            roots.append(Path(dirpath))
    return roots


@app.command()
def run(
    citygml_path: Optional[Path] = typer.Option(
        None,
        exists=True,
        dir_okay=True,
        file_okay=False,
        help="Path to PLATEAU CityGML root (folder containing udx/).",
    ),
    url: Optional[str] = typer.Option(
        None, help="URL to PLATEAU CityGML zip (will download and extract)."
    ),
    output_dir: Path = typer.Option(Path("app/data/temp"), help="Output directory for cache files."),
    aoi: Optional[List[float]] = typer.Option(
        None,
        help="AOI rectangle as min_lon min_lat max_lon max_lat",
    ),
    write_parquet: bool = typer.Option(True, help="Write GeoParquet (.parquet)"),
    write_fgb: bool = typer.Option(True, help="Write FlatGeobuf (.fgb)"),
    target_crs: int = typer.Option(4326, help="Target CRS EPSG code for outputs."),
    export_terrain: bool = typer.Option(False, help="Also export terrain elements."),
    ssl_verify: bool = typer.Option(True, help="Verify SSL when downloading."),
    ca_bundle: Optional[Path] = typer.Option(None, help="Path to CA bundle for TLS."),
    timeout: int = typer.Option(60, help="Network timeout in seconds."),
):
    """Precompute minimal building (and optional terrain) datasets from CityGML."""

    if not citygml_path and not url:
        typer.echo("Either --citygml-path or --url must be provided", err=True)
        raise typer.Exit(code=2)

    _ensure_dir(output_dir)

    rectangle_vertices = _parse_rectangle(aoi)

    # Determine single dataset vs batch mode
    building_gdf: Optional[gpd.GeoDataFrame] = None
    terrain_gdf: Optional[gpd.GeoDataFrame] = None

    if url:
        # Single dataset via URL
        b, t, _ = load_buid_dem_veg_from_citygml(
            url=url,
            base_dir=str(output_dir),
            citygml_path=None,
            rectangle_vertices=rectangle_vertices,
            ssl_verify=ssl_verify,
            ca_bundle=str(ca_bundle) if ca_bundle else None,
            timeout=timeout,
        )
        building_gdf = b
        terrain_gdf = t
    else:
        # Local path provided: check if it is a dataset root (contains 'udx') or a batch root
        citygml_path = Path(citygml_path) if citygml_path else None
        assert citygml_path is not None

        udx_dir = citygml_path / 'udx'
        dataset_paths: List[Path]
        if udx_dir.exists() and udx_dir.is_dir():
            dataset_paths = [citygml_path]
        else:
            dataset_paths = _find_dataset_roots(citygml_path)
            if not dataset_paths:
                typer.echo(f"No CityGML datasets (with 'udx/') found under {citygml_path}")
                raise typer.Exit(code=1)

        b_list: List[gpd.GeoDataFrame] = []
        t_list: List[gpd.GeoDataFrame] = []

        for ds in dataset_paths:
            typer.echo(f"Processing dataset: {ds}")
            b, t, _ = load_buid_dem_veg_from_citygml(
                url=None,
                base_dir=str(output_dir),
                citygml_path=str(ds),
                rectangle_vertices=rectangle_vertices,
                ssl_verify=ssl_verify,
                ca_bundle=str(ca_bundle) if ca_bundle else None,
                timeout=timeout,
            )
            if b is not None and not b.empty:
                b_list.append(b)
            if t is not None and not t.empty:
                t_list.append(t)

        if b_list:
            building_gdf = gpd.GeoDataFrame(pd.concat(b_list, ignore_index=True), geometry='geometry')
        if t_list:
            terrain_gdf = gpd.GeoDataFrame(pd.concat(t_list, ignore_index=True), geometry='geometry')

    # Normalize CRS and optionally clip
    building_gdf = _normalize_crs(building_gdf, to_epsg=target_crs)
    terrain_gdf = _normalize_crs(terrain_gdf, to_epsg=target_crs)

    building_gdf = _maybe_clip_to_aoi(building_gdf, rectangle_vertices)
    terrain_gdf = _maybe_clip_to_aoi(terrain_gdf, rectangle_vertices)

    if building_gdf is None or building_gdf.empty:
        typer.echo("No buildings extracted from CityGML.")
        raise typer.Exit(code=1)

    b_min = _compute_minimal_building_schema(building_gdf)

    # Write outputs
    base = output_dir / "citygml_cache"
    _ensure_dir(base)
    outputs = []
    if write_parquet:
        b_parquet = base / "buildings.parquet"
        b_min.to_parquet(b_parquet, index=False)
        outputs.append(b_parquet)
    if write_fgb:
        b_fgb = base / "buildings.fgb"
        b_min.to_file(b_fgb, driver="FlatGeobuf")
        outputs.append(b_fgb)

    if export_terrain and terrain_gdf is not None and not terrain_gdf.empty:
        t_min = terrain_gdf.copy()
        t_min = t_min[[c for c in t_min.columns if c in ("elevation", "geometry", "source_file") or c == "geometry"]]
        if write_parquet:
            t_parquet = base / "terrain.parquet"
            t_min.to_parquet(t_parquet, index=False)
            outputs.append(t_parquet)
        if write_fgb:
            t_fgb = base / "terrain.fgb"
            t_min.to_file(t_fgb, driver="FlatGeobuf")
            outputs.append(t_fgb)

    typer.echo("Written:" )
    for p in outputs:
        typer.echo(f" - {p}")


if __name__ == "__main__":
    app()


