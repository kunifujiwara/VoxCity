"""Build the example city models published on Hugging Face (see voxcity.examples).

Usage:
    python scripts/build_example_cities.py OUT_DIR [slug ...]

Writes OUT_DIR/cities/<slug>.h5, OUT_DIR/previews/<slug>.png, OUT_DIR/cities.csv and
OUT_DIR/README.md (the dataset card), and refreshes src/voxcity/examples_catalog.json.
Cities whose .h5 already exists are not rebuilt, so the script can be re-run after a
failed download (public Overpass servers are sometimes down).

Every source is pinned and needs no account. Land cover comes from OpenStreetMap (ODbL)
rather than OpenEarthMap Japan or Earth Engine products. Terrain is FABDEM outside Japan
and the US, so those models carry FABDEM's CC BY-NC-SA 4.0 terms (non-commercial use).
"""

import csv
import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG = REPO_ROOT / "src" / "voxcity" / "examples_catalog.json"
SIZE_M = 1000
MESHSIZE = 5

# slug, title, country, (lon, lat) of the box centre, region
CITIES = [
    ("tokyo", "Tokyo, Marunouchi", "Japan", (139.7650, 35.6810), "japan"),
    ("singapore", "Singapore, Downtown Core", "Singapore", (103.8515, 1.2830), "other"),
    ("hong_kong", "Hong Kong, Central", "China (Hong Kong)", (114.1590, 22.2815), "other"),
    ("seoul", "Seoul, Jongno", "South Korea", (126.9780, 37.5700), "other"),
    ("mumbai", "Mumbai, Fort", "India", (72.8330, 18.9320), "other"),
    ("dubai", "Dubai, Marina", "United Arab Emirates", (55.1400, 25.0800), "other"),
    ("new_york", "New York, Midtown Manhattan", "United States", (-73.9850, 40.7560), "usa"),
    ("san_francisco", "San Francisco, Financial District", "United States", (-122.4000, 37.7930), "usa"),
    ("chicago", "Chicago, The Loop", "United States", (-87.6300, 41.8810), "usa"),
    ("london", "London, City of London", "United Kingdom", (-0.0890, 51.5140), "europe"),
    ("paris", "Paris, Louvre and Palais-Royal", "France", (2.3390, 48.8630), "europe"),
    ("barcelona", "Barcelona, Eixample", "Spain", (2.1640, 41.3910), "europe"),
    ("amsterdam", "Amsterdam, Centrum", "Netherlands", (4.8950, 52.3720), "europe"),
    ("berlin", "Berlin, Mitte", "Germany", (13.3950, 52.5170), "europe"),
    ("sydney", "Sydney, Central Business District", "Australia", (151.2080, -33.8680), "australia"),
    ("sao_paulo", "São Paulo, Avenida Paulista", "Brazil", (-46.6560, -23.5630), "other"),
]


def sources_for(region):
    """Account-free sources for a region."""
    dem = {"japan": "GSI DEM Japan", "usa": "USGS 3DEP 10m"}.get(region, "FABDEM")
    comp = "Microsoft Building Footprints" if region in ("usa", "europe", "australia") else "None"
    return {
        "building_source": "OpenStreetMap",
        "building_complementary_source": comp,
        "land_cover_source": "OpenStreetMap",
        "canopy_height_source": "High Resolution 1m Global Canopy Height Maps v2",
        "dem_source": dem,
    }


def square_around(lon, lat, size_m):
    from geopy.distance import distance
    half = size_m / 2
    north = distance(meters=half).destination((lat, lon), bearing=0).latitude
    south = distance(meters=half).destination((lat, lon), bearing=180).latitude
    east = distance(meters=half).destination((lat, lon), bearing=90).longitude
    west = distance(meters=half).destination((lat, lon), bearing=270).longitude
    return [(west, south), (west, north), (east, north), (east, south)]   # SW, NW, NE, SE


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_one(slug, center, region, out):
    from voxcity.generator import get_voxcity
    from voxcity.io import save_voxcity

    rect = square_around(*center, SIZE_M)
    if sources_for(region)["dem_source"] == "FABDEM":
        # get_voxcity quietly falls back to Copernicus when FABDEM cannot be read.
        from voxcity.downloader.fabdem import fetch_fabdem
        with tempfile.TemporaryDirectory() as work:
            if fetch_fabdem(rect, work) is None:
                raise RuntimeError(f"FABDEM could not be read for {slug}")
    with tempfile.TemporaryDirectory() as work:
        city = get_voxcity(rect, MESHSIZE, output_dir=work, **sources_for(region))
    save_voxcity(str(out / "cities" / f"{slug}.h5"), city)
    return city


def preview(city, slug, out):
    from voxcity.visualizer import visualize_voxcity
    with tempfile.TemporaryDirectory() as work:
        visualize_voxcity(city, mode="static", output_directory=work, image_size=(800, 600))
        pngs = sorted(Path(work).rglob("*iso_front_right*.png")) or sorted(Path(work).rglob("*.png"))
        from PIL import Image, ImageChops
        img = Image.open(pngs[0]).convert("RGB")
        bbox = ImageChops.difference(img, Image.new("RGB", img.size, img.getpixel((0, 0)))).getbbox()
        if bbox:
            pad = 16
            img = img.crop((max(bbox[0] - pad, 0), max(bbox[1] - pad, 0),
                            min(bbox[2] + pad, img.width), min(bbox[3] + pad, img.height)))
        img.save(out / "previews" / f"{slug}.png", optimize=True)


def entry_for(slug, title, country, center, region, out):
    from voxcity.io import load_h5
    path = out / "cities" / f"{slug}.h5"
    city = load_h5(str(path))
    ny, nx, nz = city.voxels.classes.shape
    return {
        "name": slug,
        "title": title,
        "country": country,
        "center": [round(center[0], 6), round(center[1], 6)],
        "size_m": SIZE_M,
        "meshsize": MESHSIZE,
        "grid_shape": [int(ny), int(nx), int(nz)],
        "buildings": int(len(set(city.buildings.ids.ravel().tolist()) - {0})),
        "sources": sources_for(region),
        "file": f"cities/{slug}.h5",
        "preview": f"previews/{slug}.png",
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


CARD = """---
license: other
license_name: odbl-1.0-and-cc-by-nc-sa-4.0
license_link: LICENSE.md
pretty_name: VoxCity example cities
tags:
- urban
- 3d-city-model
- voxel
- geospatial
- urban-climate
- openstreetmap
size_categories:
- n<1K
configs:
- config_name: default
  data_files: cities.csv
---

# VoxCity example cities

Ready-made 3D voxel models of {n} city centres, built with
[VoxCity](https://github.com/kunifujiwara/VoxCity) from open data. Load one in a line and go straight
to visualization, solar and view simulations, or export to ENVI-met, PALM, OBJ and more.

```python
pip install voxcity
```

```python
from voxcity.examples import list_example_cities, load_example_city

print([c["name"] for c in list_example_cities()])
city = load_example_city("tokyo")          # downloaded once, then cached
```

Each model covers {size} m × {size} m in {mesh} m cubes. The file is a VoxCity HDF5 model: voxel classes,
building heights and IDs with footprints, land cover, terrain and tree canopy.

| Name | Area | Grid (y, x, z) | Buildings | Terrain |
|---|---|---|---|---|
{rows}

## Sources and licence

The models are derived databases of OpenStreetMap and are released under the
[Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/).

**Non-commercial use only outside Japan and the US.** Terrain in those models comes from
FABDEM, which is licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/),
so they may not be used commercially. The Japan and US models use public terrain data
and carry no such restriction. The Terrain column above shows the source of each model.

Credit the sources below when you use them.

- Buildings and land cover: © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors, ODbL 1.0.
- Building footprints (USA, Europe, Australia): [Microsoft Global ML Building Footprints](https://github.com/microsoft/GlobalMLBuildingFootprints), ODbL 1.0.
- Tree canopy height: Meta and World Resources Institute, High Resolution Canopy Height Maps v2, CC BY 4.0.
- Terrain: FABDEM v1-2, Hawker et al. (2022), University of Bristol, [doi:10.5523/bris.s5hqmjcdj8yo2ibzi9b4ew3sn](https://doi.org/10.5523/bris.s5hqmjcdj8yo2ibzi9b4ew3sn), CC BY-NC-SA 4.0, derived from Copernicus DEM GLO-30 (© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA); USGS 3DEP (public domain); GSI DEM, Geospatial Information Authority of Japan.

## Citation

```bibtex
@article{{fujiwara2025voxcity,
  title={{VoxCity: A Seamless Framework for Open Geospatial Data Integration, Grid-Based Semantic 3D City Model Generation, and Urban Environment Simulation}},
  author={{Fujiwara, Kunihiko and Tsurumi, Ryuta and Kiyono, Tomoki and Fan, Zicheng and Liang, Xiucheng and Lei, Binyu and Yap, Winston and Ito, Koichi and Biljecki, Filip}},
  journal={{Computers, Environment and Urban Systems}},
  volume={{123}},
  pages={{102366}},
  year={{2026}},
  doi={{10.1016/j.compenvurbsys.2025.102366}}
}}
```
"""


LICENSE = """# Licence

The VoxCity example city models are released under the
[Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/),
as derived databases of OpenStreetMap.

Models whose terrain source is FABDEM (every city outside Japan and the United States;
see `cities.csv`) also contain data derived from FABDEM v1-2, licensed
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Those models may
be used for non-commercial purposes only, and the FABDEM terms apply to their terrain.
"""


def write_outputs(entries, out):
    out.joinpath("cities.csv").parent.mkdir(parents=True, exist_ok=True)
    with open(out / "cities.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "title", "country", "lon", "lat", "size_m", "meshsize", "buildings",
                    "building_source", "land_cover_source", "canopy_height_source", "dem_source", "file", "preview"])
        for e in entries:
            s = e["sources"]
            w.writerow([e["name"], e["title"], e["country"], *e["center"], e["size_m"], e["meshsize"], e["buildings"],
                        s["building_source"] + ("" if s["building_complementary_source"] == "None" else " + " + s["building_complementary_source"]),
                        s["land_cover_source"], s["canopy_height_source"], s["dem_source"], e["file"], e["preview"]])
    rows = "\n".join(
        f"| `{e['name']}` | {e['title']} | {' × '.join(map(str, e['grid_shape']))} | {e['buildings']:,} | {e['sources']['dem_source']} |"
        for e in entries)
    (out / "LICENSE.md").write_text(LICENSE, encoding="utf-8")
    (out / "README.md").write_text(CARD.format(n=len(entries), size=SIZE_M, mesh=MESHSIZE, rows=rows), encoding="utf-8")
    CATALOG.write_text(json.dumps({"version": 1, "cities": entries}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    out = Path(sys.argv[1]).resolve()
    only = set(sys.argv[2:])
    (out / "cities").mkdir(parents=True, exist_ok=True)
    (out / "previews").mkdir(parents=True, exist_ok=True)
    for slug, title, country, center, region in CITIES:
        if only and slug not in only:
            continue
        h5 = out / "cities" / f"{slug}.h5"
        if h5.exists() and (out / "previews" / f"{slug}.png").exists():
            print(f"skip {slug} (exists)", flush=True)
            continue
        t = time.time()
        try:
            city = build_one(slug, center, region, out) if not h5.exists() else None
            if city is None:
                from voxcity.io import load_h5
                city = load_h5(str(h5))
            preview(city, slug, out)
            print(f"built {slug} in {time.time() - t:.0f}s", flush=True)
        except Exception:
            h5.unlink(missing_ok=True)
            print(f"FAILED {slug}:\n{traceback.format_exc()[-1500:]}", flush=True)
    entries = [entry_for(slug, title, country, center, region, out)
               for slug, title, country, center, region in CITIES if (out / "cities" / f"{slug}.h5").exists()]
    write_outputs(entries, out)
    print(f"catalog: {len(entries)} of {len(CITIES)} cities -> {CATALOG}", flush=True)


if __name__ == "__main__":
    main()
