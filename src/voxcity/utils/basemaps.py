"""
Key-free basemap tile sources shared by VoxCity's interactive and static maps.

CARTO's raster basemaps now require an API key (unkeyed requests get an
"API KEY REQUIRED" placeholder tile) and tile.openstreetmap.org blocks Google
Colab under its tile usage policy, so VoxCity defaults to Esri's ArcGIS Online
basemaps, which need neither.

Static (matplotlib/contextily) plots take a ``basemap`` name from
:data:`ESRI_BASEMAPS`; the old ``'CartoDB ...'`` names are still accepted and
mapped to the closest Esri style. Any contextily source (an ``xyzservices``
``TileProvider`` or an XYZ URL template) is passed through unchanged, so a
keyed provider can still be used explicitly.
"""

from __future__ import annotations

import warnings

_ESRI_ROOT = "https://server.arcgisonline.com/ArcGIS/rest/services"

ESRI_ATTRIBUTION = "Esri, HERE, Garmin, (c) OpenStreetMap contributors"
ESRI_HTML_ATTRIBUTION = (
    "Tiles &copy; Esri &mdash; Esri, HERE, Garmin, OpenStreetMap contributors"
)

# name -> (ArcGIS Online service path, highest zoom with real tiles).
# The gray canvases return a "Map data not yet available" tile above 16.
ESRI_BASEMAPS = {
    "Esri light gray": ("Canvas/World_Light_Gray_Base", 16),
    "Esri dark gray": ("Canvas/World_Dark_Gray_Base", 16),
    "Esri street": ("World_Street_Map", 19),
    "Esri topo": ("World_Topo_Map", 19),
    "Esri satellite": ("World_Imagery", 19),
}

DEFAULT_STATIC_BASEMAP = "Esri light gray"

_CARTO_ALIASES = {
    "CartoDB light": "Esri light gray",
    "CartoDB light no labels": "Esri light gray",
    "CartoDB dark": "Esri dark gray",
    "CartoDB dark no labels": "Esri dark gray",
    "CartoDB voyager": "Esri street",
}


def esri_tile_url(name: str) -> str:
    """Return the XYZ URL template of the Esri basemap called *name*."""
    path, _ = ESRI_BASEMAPS[name]
    return f"{_ESRI_ROOT}/{path}/MapServer/tile/{{z}}/{{y}}/{{x}}"


def resolve_static_basemap(basemap=None):
    """Turn a ``basemap`` argument into a contextily ``source``.

    Args:
        basemap: A name from :data:`ESRI_BASEMAPS`, a legacy ``'CartoDB ...'``
            name, an ``xyzservices.TileProvider``, an XYZ URL template, or
            ``None`` for :data:`DEFAULT_STATIC_BASEMAP`.

    Returns:
        An ``xyzservices.TileProvider`` (for names) or *basemap* unchanged.

    Raises:
        ValueError: If *basemap* is a name VoxCity does not know.
    """
    if basemap is None:
        basemap = DEFAULT_STATIC_BASEMAP
    if not isinstance(basemap, str) or "{z}" in basemap:
        return basemap
    if basemap in _CARTO_ALIASES:
        replacement = _CARTO_ALIASES[basemap]
        warnings.warn(
            f"basemap={basemap!r} now draws {replacement!r}: CARTO basemap tiles "
            "require an API key. Pass a VoxCity basemap name "
            f"({', '.join(map(repr, ESRI_BASEMAPS))}) or your own contextily "
            "provider to silence this warning.",
            FutureWarning,
            stacklevel=3,
        )
        basemap = replacement
    if basemap not in ESRI_BASEMAPS:
        known = ", ".join(map(repr, [*ESRI_BASEMAPS, *_CARTO_ALIASES]))
        raise ValueError(f"Unknown basemap {basemap!r}. Choose one of: {known}.")

    from xyzservices import TileProvider

    _, max_zoom = ESRI_BASEMAPS[basemap]
    return TileProvider(
        name=basemap,
        url=esri_tile_url(basemap),
        attribution=ESRI_ATTRIBUTION,
        html_attribution=ESRI_HTML_ATTRIBUTION,
        max_zoom=max_zoom,
    )


def clamp_zoom(source, zoom):
    """Cap an explicit *zoom* at *source*'s ``max_zoom``; pass ``'auto'`` through."""
    max_zoom = source.get("max_zoom") if isinstance(source, dict) else None
    if zoom == "auto" or max_zoom is None:
        return zoom
    return min(zoom, max_zoom)


def add_static_basemap(ax, basemap=None, zoom="auto", **kwargs):
    """Draw *basemap* under *ax* with contextily.

    Zoom never exceeds the provider's real tiles: an explicit *zoom* is
    clamped, and contextily's warning when it clips an inferred zoom (every
    small area on a gray canvas) is suppressed.
    """
    import contextily as ctx

    source = resolve_static_basemap(basemap)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"The inferred zoom level", category=UserWarning
        )
        ctx.add_basemap(ax, source=source, zoom=clamp_zoom(source, zoom), **kwargs)
