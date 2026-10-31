"""Ready-made example city models, downloaded on first use from Hugging Face.

Each model is a VoxCity HDF5 file built from open data with no account
(OpenStreetMap buildings and land cover, Microsoft Building Footprints where
available, Meta/WRI CHMv2 canopy, and Copernicus, USGS 3DEP or GSI terrain).
Files are cached locally, so a model is downloaded once.

Example
-------
>>> from voxcity.examples import list_example_cities, load_example_city
>>> [c["name"] for c in list_example_cities()]      # doctest: +SKIP
['tokyo', 'singapore', ...]
>>> city = load_example_city("tokyo")               # doctest: +SKIP

The dataset lives at https://huggingface.co/datasets/<EXAMPLES_REPO>. Set the
``VOXCITY_EXAMPLES_REPO`` environment variable to use a fork or mirror, and
``VOXCITY_CACHE_DIR`` to change where downloads are kept.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import requests

from .errors import DownloaderError

__all__ = [
    "EXAMPLES_REPO",
    "list_example_cities",
    "example_city_path",
    "load_example_city",
]

#: Hugging Face dataset repository that holds the example models.
EXAMPLES_REPO = "kunifujiwara/voxcity-example-cities"

_CATALOG_FILE = Path(__file__).with_name("examples_catalog.json")
_CHUNK = 1 << 20


def _load_catalog():
    with open(_CATALOG_FILE, encoding="utf-8") as f:
        return json.load(f)["cities"]


def _repo():
    return os.environ.get("VOXCITY_EXAMPLES_REPO") or EXAMPLES_REPO


def _default_cache_dir():
    root = os.environ.get("VOXCITY_CACHE_DIR")
    if root:
        return Path(root) / "examples"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "voxcity" / "examples"


def _normalize(name):
    return "_".join(str(name).strip().lower().replace("-", " ").split())


def _entry(name):
    catalog = _load_catalog()
    key = _normalize(name)
    for entry in catalog:
        if entry["name"] == key:
            return entry
    available = ", ".join(e["name"] for e in catalog)
    raise ValueError(f"No example city named {name!r}. Available: {available}.")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def list_example_cities():
    """Return the example cities as a list of dicts.

    Each dict has ``name`` (the key for :func:`load_example_city`), ``title``,
    ``country``, ``center`` (lon, lat), ``size_m``, ``meshsize``, ``sources``
    (the data source of each layer), ``file``, ``sha256`` and ``bytes``.
    """
    return copy.deepcopy(_load_catalog())


def example_city_path(name, cache_dir=None, force_download=False, revision="main"):
    """Download an example city's HDF5 file if needed and return its local path.

    Parameters
    ----------
    name : str
        A name from :func:`list_example_cities`, e.g. ``"tokyo"`` or ``"New York"``.
    cache_dir : str or Path, optional
        Folder for downloads. Default: ``$VOXCITY_CACHE_DIR/examples`` or
        ``~/.cache/voxcity/examples``.
    force_download : bool
        Download again even when a verified copy is cached.
    revision : str
        Branch, tag or commit of the dataset repository.
    """
    entry = _entry(name)
    repo = _repo()
    folder = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    path = folder / repo.replace("/", "--") / revision / entry["file"]
    expected = entry.get("sha256")

    if path.is_file() and not force_download:
        if not expected or _sha256(path) == expected:
            return path
        path.unlink()                         # damaged or outdated copy

    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{entry['file']}"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in response.iter_content(chunk_size=_CHUNK):
                    f.write(chunk)
    except requests.RequestException as exc:
        tmp.unlink(missing_ok=True)
        raise DownloaderError(f"Could not download example city {entry['name']!r} from {url}: {exc}") from exc

    if expected and _sha256(tmp) != expected:
        tmp.unlink(missing_ok=True)
        raise DownloaderError(
            f"Downloaded file for {entry['name']!r} failed its checksum; the dataset may have been "
            f"updated. Upgrade voxcity, or pass revision= to pin a dataset version."
        )
    os.replace(tmp, path)
    return path


def load_example_city(name, cache_dir=None, force_download=False, revision="main"):
    """Load a ready-made example city as a :class:`~voxcity.models.VoxCity`.

    The model is downloaded from Hugging Face on first use and cached. It works
    with every visualizer, simulator and exporter, like a model from
    :func:`voxcity.generator.get_voxcity`. See :func:`example_city_path` for
    the parameters.
    """
    from .io import load_h5

    path = example_city_path(name, cache_dir=cache_dir, force_download=force_download, revision=revision)
    return load_h5(str(path))
