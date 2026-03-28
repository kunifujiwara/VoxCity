"""
Building footprint display and polygon drawing utilities.

Provides:
- display_buildings_and_draw_polygon: Visualise buildings and draw polygons
- get_polygon_vertices: Extract vertices from drawn polygons
"""

from __future__ import annotations

import shapely.geometry as geom
from ipyleaflet import (
    Map,
    DrawControl,
    Polygon as LeafletPolygon,
)

# Import VoxCity for type checking (avoid circular import with TYPE_CHECKING)
try:
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from ...models import VoxCity
except ImportError:
    pass


def display_buildings_and_draw_polygon(
    voxcity=None,
    building_gdf=None,
    rectangle_vertices=None,
    zoom=17,
):
    """
    Display building footprints and enable polygon drawing on an interactive map.

    Args:
        voxcity (VoxCity, optional): VoxCity object to extract data from.
        building_gdf (GeoDataFrame, optional): Building footprints.
        rectangle_vertices (list, optional): [lon, lat] rectangle corners.
        zoom (int): Initial zoom level. Default=17.

    Returns:
        tuple: (Map, drawn_polygons list of dicts with 'id', 'vertices', 'color')
    """
    # Extract data from VoxCity if provided
    if voxcity is not None:
        if building_gdf is None:
            building_gdf = voxcity.extras.get("building_gdf", None)
        if rectangle_vertices is None:
            rectangle_vertices = voxcity.extras.get("rectangle_vertices", None)

    # Determine map center
    if rectangle_vertices is not None:
        lons = [v[0] for v in rectangle_vertices]
        lats = [v[1] for v in rectangle_vertices]
        center_lon = (min(lons) + max(lons)) / 2
        center_lat = (min(lats) + max(lats)) / 2
    elif building_gdf is not None and len(building_gdf) > 0:
        bounds = building_gdf.total_bounds
        center_lon = (bounds[0] + bounds[2]) / 2
        center_lat = (bounds[1] + bounds[3]) / 2
    else:
        center_lon, center_lat = -100.0, 40.0

    m = Map(center=(center_lat, center_lon), zoom=zoom, scroll_wheel_zoom=True)

    # Add building footprints
    if building_gdf is not None:
        for _idx, row in building_gdf.iterrows():
            if isinstance(row.geometry, geom.Polygon):
                coords = list(row.geometry.exterior.coords)
                lat_lon_coords = [(c[1], c[0]) for c in coords[:-1]]
                bldg_layer = LeafletPolygon(
                    locations=lat_lon_coords,
                    color="blue",
                    fill_color="blue",
                    fill_opacity=0.2,
                    weight=2,
                )
                m.add_layer(bldg_layer)

    # Polygon drawing
    drawn_polygons: list[dict] = []
    polygon_counter = 0
    polygon_colors = [
        "red", "blue", "green", "orange", "purple",
        "brown", "pink", "gray", "olive", "cyan",
    ]

    draw_control = DrawControl(
        polygon={
            "shapeOptions": {"color": "red", "fillColor": "red", "fillOpacity": 0.2}
        },
        rectangle={},
        circle={},
        circlemarker={},
        polyline={},
        marker={},
    )

    def handle_draw(self, action, geo_json):
        if action == "created" and geo_json["geometry"]["type"] == "Polygon":
            nonlocal polygon_counter
            polygon_counter += 1
            coordinates = geo_json["geometry"]["coordinates"][0]
            vertices = [(coord[0], coord[1]) for coord in coordinates[:-1]]
            color = polygon_colors[polygon_counter % len(polygon_colors)]
            polygon_data = {"id": polygon_counter, "vertices": vertices, "color": color}
            drawn_polygons.append(polygon_data)
            print(
                f"Polygon {polygon_counter} drawn with {len(vertices)} vertices "
                f"(color: {color}):"
            )
            for i, (lon, lat) in enumerate(vertices):
                print(f"  Vertex {i + 1}: (lon, lat) = ({lon}, {lat})")
            print(f"Total polygons: {len(drawn_polygons)}")

    draw_control.on_draw(handle_draw)
    m.add_control(draw_control)

    return m, drawn_polygons


def get_polygon_vertices(drawn_polygons, polygon_id=None):
    """
    Extract vertices from drawn polygons data structure.

    Args:
        drawn_polygons: List returned from display_buildings_and_draw_polygon().
        polygon_id (int, optional): Specific polygon ID. If None, returns all.

    Returns:
        List of (lon, lat) tuples for the specified polygon, or list of lists for all.
    """
    if not drawn_polygons:
        return []
    if polygon_id is not None:
        for polygon in drawn_polygons:
            if polygon["id"] == polygon_id:
                return polygon["vertices"]
        return []
    return [polygon["vertices"] for polygon in drawn_polygons]
