"""Benchmark osm_json_to_geojson (src/voxcity/downloader/osm.py) on a
synthetic Overpass-style JSON payload with ~1000 closed building ways.

No network access: the OSM JSON is generated in-process with a fixed seed.
"""

import numpy as np

from voxcity.downloader.osm import osm_json_to_geojson

N_BUILDINGS = 1000


def _make_osm_json(n_buildings=N_BUILDINGS, seed=2):
    rng = np.random.default_rng(seed)

    elements = []
    node_id = 1
    way_id = 1_000_000

    for _ in range(n_buildings):
        lon0 = float(rng.uniform(-180.0, 180.0))
        lat0 = float(rng.uniform(-85.0, 85.0))
        w = float(rng.uniform(0.0001, 0.001))
        h = float(rng.uniform(0.0001, 0.001))

        # Four corners of a small rectangle, closed by repeating the first node.
        corners = [
            (lon0, lat0),
            (lon0 + w, lat0),
            (lon0 + w, lat0 + h),
            (lon0, lat0 + h),
        ]
        corner_ids = []
        for lon, lat in corners:
            elements.append({"type": "node", "id": node_id, "lon": lon, "lat": lat})
            corner_ids.append(node_id)
            node_id += 1

        way_nodes = corner_ids + [corner_ids[0]]  # closed ring
        elements.append(
            {
                "type": "way",
                "id": way_id,
                "nodes": way_nodes,
                "tags": {"building": "yes", "height": str(round(float(rng.uniform(3.0, 60.0)), 1))},
            }
        )
        way_id += 1

    return {"elements": elements}


def test_bench_osm_json_to_geojson(benchmark):
    osm_data = _make_osm_json()

    result = benchmark(osm_json_to_geojson, osm_data)

    assert len(result["features"]) == N_BUILDINGS
