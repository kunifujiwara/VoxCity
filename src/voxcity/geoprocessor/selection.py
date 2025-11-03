"""
Selection and filtering helpers for building footprints.
"""

from typing import List, Dict, Tuple

from shapely.geometry import Polygon, Point, shape
from shapely.errors import ShapelyError

from .utils import validate_polygon_coordinates
from ..utils.logging import get_logger


def filter_buildings(geojson_data, plotting_box):
    """
    Filter building features that intersect with a given bounding box.
    """
    logger = get_logger(__name__)
    filtered_features = []

    for feature in geojson_data:
        if not validate_polygon_coordinates(feature['geometry']):
            logger.warning("Skipping feature with invalid geometry: %s", feature.get('geometry'))
            continue

        try:
            geom = shape(feature['geometry'])
            if not geom.is_valid:
                logger.warning("Skipping invalid geometry: %s", geom)
                continue

            if plotting_box.intersects(geom):
                filtered_features.append(feature)

        except ShapelyError as e:
            logger.warning("Skipping feature due to geometry error: %s", e)

    return filtered_features


def find_building_containing_point(building_gdf, target_point):
    """
    Find building IDs that contain a given point in their footprint.
    """
    point = Point(target_point[0], target_point[1])

    id_list = []
    for _, row in building_gdf.iterrows():
        if not isinstance(row.geometry, Polygon):
            continue
        if row.geometry.contains(point):
            id_list.append(row.get('id', None))

    return id_list


def get_buildings_in_drawn_polygon(building_gdf, drawn_polygons, operation='within'):
    """
    Find buildings that intersect with or are contained within user-drawn polygons.
    """
    if not drawn_polygons:
        return []

    included_building_ids = set()

    for polygon_data in drawn_polygons:
        vertices = polygon_data['vertices']
        drawn_polygon_shapely = Polygon(vertices)

        for _, row in building_gdf.iterrows():
            if not isinstance(row.geometry, Polygon):
                continue

            if operation == 'intersect':
                if row.geometry.intersects(drawn_polygon_shapely):
                    included_building_ids.add(row.get('id', None))
            elif operation == 'within':
                if row.geometry.within(drawn_polygon_shapely):
                    included_building_ids.add(row.get('id', None))
            else:
                raise ValueError("operation must be 'intersect' or 'within'")

    return list(included_building_ids)


