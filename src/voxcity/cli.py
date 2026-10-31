"""Command-line interface: ``voxcity generate | export | examples | info``.

The CLI covers the common path for people who do not write Python: build a
model for an area, save it, and export it to ENVI-met, PALM, OBJ, MagicaVoxel,
NetCDF or GeoTIFF. Anything beyond that (simulations, custom data, plots) is
done in Python with the same functions the CLI calls.

Examples
--------
::

    voxcity generate --bbox -122.3359 47.5983 -122.3292 47.6028 --mesh 5 --out seattle.h5
    voxcity generate --center 139.765 35.681 --size 500 --export obj,envimet
    voxcity export seattle.h5 --to palm,geotiff --out exports
    voxcity examples list
    voxcity examples fetch tokyo --export obj
    voxcity info seattle.h5
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

__all__ = ["main", "build_parser", "FORMATS"]

# Export format name -> writer(city, output_dir, basename). Each writer imports
# its exporter lazily so ``voxcity --help`` stays fast.


def _write_obj(city, out, name):
    from .exporter.obj import export_obj
    export_obj(city, out, name)


def _write_envimet(city, out, name):
    from .exporter.envimet import export_inx, generate_edb_file
    export_inx(city, out, name)
    generate_edb_file(output_dir=out)


def _write_palm(city, out, name):
    from .exporter.palm import export_palm
    export_palm(city, out, domain_name=name)


def _write_vox(city, out, name):
    from .exporter.magicavoxel import export_magicavoxel_vox
    export_magicavoxel_vox(city, out, name)


def _write_netcdf(city, out, name):
    from .exporter.netcdf import save_voxel_netcdf
    save_voxel_netcdf(
        city.voxels.classes,
        os.path.join(out, f"{name}.nc"),
        float(city.voxels.meta.meshsize),
        rectangle_vertices=city.extras.get("rectangle_vertices"),
    )


def _write_geotiff(city, out, name):
    from .exporter.geotiff import export_geotiffs
    export_geotiffs(city, out, name)


FORMATS = {
    "obj": _write_obj,
    "envimet": _write_envimet,
    "palm": _write_palm,
    "vox": _write_vox,
    "netcdf": _write_netcdf,
    "geotiff": _write_geotiff,
}


def _formats(text):
    """argparse type for ``--export obj,envimet``: a list of known formats."""
    names = [n.strip().lower() for n in text.split(",") if n.strip()]
    unknown = [n for n in names if n not in FORMATS]
    if unknown or not names:
        raise argparse.ArgumentTypeError(
            f"unknown format {', '.join(unknown) or text!r}; choose from {', '.join(FORMATS)}"
        )
    return list(dict.fromkeys(names))


def export_city(city, formats, output_dir, basename="voxcity"):
    """Write ``city`` in each of ``formats`` under ``output_dir``; returns the dir."""
    os.makedirs(output_dir, exist_ok=True)
    for fmt in formats:
        FORMATS[fmt](city, output_dir, basename)
        print(f"exported {fmt} -> {output_dir}", flush=True)
    return output_dir


def rectangle_from_bbox(west, south, east, north):
    """(lon, lat) corners [SW, NW, NE, SE] of a west/south/east/north box."""
    if not (-180 <= west < east <= 180) or not (-90 <= south < north <= 90):
        raise ValueError("--bbox expects WEST SOUTH EAST NORTH in degrees, with west < east and south < north")
    return [(west, south), (west, north), (east, north), (east, south)]


def rectangle_from_center(lon, lat, size_m):
    """(lon, lat) corners of a square ``size_m`` across centred on a point."""
    from geopy.distance import distance
    if size_m <= 0:
        raise ValueError("--size must be a positive number of metres")
    half = size_m / 2
    north = distance(meters=half).destination((lat, lon), bearing=0).latitude
    south = distance(meters=half).destination((lat, lon), bearing=180).latitude
    east = distance(meters=half).destination((lat, lon), bearing=90).longitude
    west = distance(meters=half).destination((lat, lon), bearing=270).longitude
    return [(west, south), (west, north), (east, north), (east, south)]


def _load(path):
    from .io import load_voxcity
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no such model file: {path}")
    return load_voxcity(path)


def _add_export_args(p, flag="--export"):
    p.add_argument(flag, dest="formats", type=_formats, metavar="FORMATS",
                   help=f"comma-separated list of {', '.join(FORMATS)}")
    p.add_argument("--export-dir", metavar="DIR",
                   help="folder for exported files (default: 'export' next to the model)")
    p.add_argument("--name", default="voxcity", help="base name of exported files (default: voxcity)")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="voxcity",
        description="Generate 3D voxel city models from open data and export them.",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="only print warnings and errors")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    g = sub.add_parser("generate", help="build a model for an area and save it as HDF5",
                       description="Build a VoxCity model for an area and save it as an HDF5 file.")
    area = g.add_mutually_exclusive_group(required=True)
    area.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"),
                      help="area as lon/lat bounds in degrees")
    area.add_argument("--center", nargs=2, type=float, metavar=("LON", "LAT"),
                      help="centre of a square area; use with --size")
    g.add_argument("--size", type=float, metavar="METERS", help="side of the square around --center")
    g.add_argument("--mesh", type=float, default=5.0, metavar="METERS", help="voxel size (default: 5)")
    g.add_argument("--out", default="voxcity.h5", metavar="FILE", help="model file to write (default: voxcity.h5)")
    g.add_argument("--building-source", metavar="NAME")
    g.add_argument("--building-complement", metavar="NAME", help="building_complementary_source")
    g.add_argument("--land-cover-source", metavar="NAME")
    g.add_argument("--canopy-source", metavar="NAME", help="canopy_height_source")
    g.add_argument("--dem-source", metavar="NAME")
    g.add_argument("--work-dir", metavar="DIR", help="keep downloaded rasters here (default: a temporary folder)")
    _add_export_args(g)

    e = sub.add_parser("export", help="export a saved model to other formats",
                       description="Export a saved VoxCity model (.h5) to ENVI-met, PALM, OBJ and more.")
    e.add_argument("model", metavar="MODEL", help="model file written by generate, examples fetch or save_voxcity")
    _add_export_args(e, "--to")
    e.add_argument("--out", dest="export_dir_alias", metavar="DIR", help=argparse.SUPPRESS)

    x = sub.add_parser("examples", help="list or download ready-made example cities")
    xs = x.add_subparsers(dest="examples_command", metavar="ACTION")
    xs.required = True
    xs.add_parser("list", help="print the available cities")
    f = xs.add_parser("fetch", help="download a city model (cached) and copy it to a file")
    f.add_argument("city", metavar="NAME", help="a name from 'voxcity examples list', e.g. tokyo")
    f.add_argument("--out", metavar="FILE", help="where to copy the model (default: NAME.h5)")
    _add_export_args(f)

    i = sub.add_parser("info", help="print the size, sources and extent of a saved model")
    i.add_argument("model", metavar="MODEL")
    return parser


def _cmd_generate(args):
    from .generator import get_voxcity
    from .io import save_voxcity

    if args.center is not None and args.size is None:
        raise ValueError("--center needs --size METERS")
    rect = (rectangle_from_bbox(*args.bbox) if args.bbox is not None
            else rectangle_from_center(*args.center, args.size))
    sources = {
        "building_source": args.building_source,
        "building_complementary_source": args.building_complement,
        "land_cover_source": args.land_cover_source,
        "canopy_height_source": args.canopy_source,
        "dem_source": args.dem_source,
    }
    sources = {k: v for k, v in sources.items() if v is not None}

    work = args.work_dir or tempfile.mkdtemp(prefix="voxcity-")
    try:
        city = get_voxcity(rect, args.mesh, output_dir=work, save_voxcity_data=False, **sources)
    finally:
        if args.work_dir is None:
            shutil.rmtree(work, ignore_errors=True)
    save_voxcity(args.out, city)
    print(f"saved {args.out}  grid {tuple(city.voxels.classes.shape)} at {args.mesh} m", flush=True)
    if args.formats:
        export_city(city, args.formats, args.export_dir or _default_export_dir(args.out), args.name)
    return 0


def _default_export_dir(model_path):
    return os.path.join(os.path.dirname(os.path.abspath(model_path)), "export")


def _cmd_export(args):
    if not args.formats:
        raise ValueError("--to FORMATS is required, e.g. --to obj,envimet")
    city = _load(args.model)
    out = args.export_dir or args.export_dir_alias or _default_export_dir(args.model)
    export_city(city, args.formats, out, args.name)
    return 0


def _cmd_examples(args):
    from . import examples

    if args.examples_command == "list":
        for c in examples.list_example_cities():
            size = c["bytes"] / 1e6
            print(f"{c['name']:<14} {c['title']:<36} {c['size_m']} m at {c['meshsize']} m, {size:.1f} MB")
        return 0

    path = examples.example_city_path(args.city)
    out = args.out or f"{examples._normalize(args.city)}.h5"
    shutil.copyfile(path, out)
    print(f"saved {out}", flush=True)
    if args.formats:
        city = _load(out)
        export_city(city, args.formats, args.export_dir or _default_export_dir(out), args.name)
    return 0


def _cmd_info(args):
    import numpy as np
    city = _load(args.model)
    shape = city.voxels.classes.shape
    ids = city.buildings.ids
    n_buildings = int(len(np.unique(ids[ids > 0])))
    print(f"file:       {args.model}")
    print(f"grid:       {shape[0]} x {shape[1]} x {shape[2]} voxels at {city.voxels.meta.meshsize} m")
    print(f"buildings:  {n_buildings} (max height {float(np.nanmax(city.buildings.heights)):.1f} m)")
    print(f"terrain:    {float(np.nanmin(city.dem.elevation)):.1f} to {float(np.nanmax(city.dem.elevation)):.1f} m")
    canopy = getattr(city.tree_canopy, "top", None)
    if canopy is not None and canopy.size:
        print(f"canopy:     max {float(np.nanmax(canopy)):.1f} m")
    rect = city.extras.get("rectangle_vertices")
    if rect is not None:
        lons = [float(v[0]) for v in rect]
        lats = [float(v[1]) for v in rect]
        print(f"extent:     lon {min(lons):.5f} to {max(lons):.5f}, lat {min(lats):.5f} to {max(lats):.5f}")
    sources = city.extras.get("selected_sources") or {
        k: city.extras[k] for k in ("building_source", "land_cover_source", "canopy_height_source", "dem_source")
        if k in city.extras
    }
    for k, v in sources.items():
        if v not in (None, "None"):
            print(f"{k + ':':<28}{v}")
    return 0


_COMMANDS = {
    "generate": _cmd_generate,
    "export": _cmd_export,
    "examples": _cmd_examples,
    "info": _cmd_info,
}


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.quiet:
        os.environ.setdefault("VOXCITY_LOG_LEVEL", "WARNING")
        import logging
        logging.getLogger("voxcity").setLevel(logging.WARNING)
    try:
        return _COMMANDS[args.command](args)
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"voxcity: error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:                       # DownloaderError and friends
        if exc.__class__.__module__.startswith("voxcity"):
            print(f"voxcity: error: {exc}", file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    sys.exit(main())
