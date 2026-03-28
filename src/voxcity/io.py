"""
Persistence functions for VoxCity models and simulation results.

Provides:
- ``save_voxcity`` / ``load_voxcity`` – pickle-based VoxCity model I/O
- ``save_h5`` / ``load_h5`` – HDF5-based VoxCity model I/O (recommended)
- ``save_results_h5`` / ``load_results_h5`` – HDF5-based combined model + results I/O
"""


# =============================================================================
# Pickle-based VoxCity model save/load
# =============================================================================

def load_voxcity(input_path, *, trusted: bool = False):
    """Load a VoxCity instance from a file.

    The format is detected automatically based on the file extension:

    * ``.h5`` / ``.hdf5`` → safe HDF5 format (no warning).
    * Anything else (e.g. ``.pkl``) → legacy pickle format.

    .. warning::
        Pickle files can execute arbitrary code on load. Only load files you
        created yourself or received from a trusted source.  Pass
        ``trusted=True`` to suppress this warning.

    Parameters
    ----------
    input_path : str or Path
        Path to the file.
    trusted : bool, default False
        (Pickle only) Set to ``True`` to acknowledge the security
        implications of loading a pickle file and suppress the warning.

    Returns
    -------
    VoxCity
    """
    import os

    ext = os.path.splitext(str(input_path))[1].lower()
    if ext in ('.h5', '.hdf5'):
        return load_h5(input_path)

    # Legacy pickle path
    import warnings
    import pickle
    import numpy as np
    from .models import GridMetadata, VoxelGrid, BuildingGrid, LandCoverGrid, DemGrid, CanopyGrid, VoxCity

    if not trusted:
        warnings.warn(
            "Loading a pickle file is potentially unsafe. Only load files you trust. "
            "Pass trusted=True to suppress this warning, or use load_h5() / "
            "load_results_h5() for a safer HDF5-based alternative.",
            UserWarning,
            stacklevel=2,
        )

    with open(input_path, 'rb') as f:
        obj = pickle.load(f)

    # New format: the entire VoxCity object (optionally wrapped)
    if isinstance(obj, VoxCity):
        return obj
    if isinstance(obj, dict) and obj.get('__format__') == 'voxcity.v2' and isinstance(obj.get('voxcity'), VoxCity):
        return obj['voxcity']

    # Legacy dict format fallback
    d = obj
    rv = d.get('rectangle_vertices') or []
    if rv:
        xs = [p[0] for p in rv]
        ys = [p[1] for p in rv]
        bounds = (min(xs), min(ys), max(xs), max(ys))
    else:
        ny, nx = d['land_cover_grid'].shape
        ms = float(d['meshsize'])
        bounds = (0.0, 0.0, nx * ms, ny * ms)

    meta = GridMetadata(crs='EPSG:4326', bounds=bounds, meshsize=float(d['meshsize']))

    voxels = VoxelGrid(classes=d['voxcity_grid'], meta=meta)
    buildings = BuildingGrid(
        heights=d['building_height_grid'],
        min_heights=d['building_min_height_grid'],
        ids=d['building_id_grid'],
        meta=meta,
    )
    land = LandCoverGrid(classes=d['land_cover_grid'], meta=meta)
    dem = DemGrid(elevation=d['dem_grid'], meta=meta)
    canopy = CanopyGrid(top=d.get('canopy_height_grid'), bottom=None, meta=meta)

    extras = {
        'rectangle_vertices': d.get('rectangle_vertices'),
        'building_gdf': d.get('building_gdf'),
    }

    return VoxCity(voxels=voxels, buildings=buildings, land_cover=land, dem=dem, tree_canopy=canopy, extras=extras)


def save_voxcity(output_path, city):
    """Save a VoxCity instance to disk.

    The format is chosen automatically based on the file extension:

    * ``.h5`` / ``.hdf5`` → safe HDF5 format (recommended).
    * Anything else (e.g. ``.pkl``) → legacy pickle format.

    Parameters
    ----------
    output_path : str or Path
        Destination file path.
    city : VoxCity
        The model to save.
    """
    import os
    from .models import VoxCity as _VoxCity
    from .utils.logging import get_logger

    if not isinstance(city, _VoxCity):
        raise TypeError("save_voxcity expects a VoxCity instance")

    ext = os.path.splitext(str(output_path))[1].lower()
    if ext in ('.h5', '.hdf5'):
        save_h5(output_path, city)
        return

    # Legacy pickle path
    import pickle
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'wb') as f:
        payload = {
            '__format__': 'voxcity.v2',
            'voxcity': city,
        }
        pickle.dump(payload, f)

    get_logger(__name__).info("Voxcity data saved to %s", output_path)


# =============================================================================
# HDF5 save/load for plain VoxCity models (no simulation results)
# =============================================================================

def save_h5(output_path, city):
    """Save a VoxCity model to an HDF5 file (no simulation results).

    This is the recommended alternative to :func:`save_voxcity` (pickle-based)
    because HDF5 files cannot execute arbitrary code on load.

    Parameters
    ----------
    output_path : str or Path
        Destination file path (e.g. ``"model.h5"``).
    city : VoxCity
        The VoxCity model instance.

    See Also
    --------
    load_h5 : Load a VoxCity model from an HDF5 file.
    save_results_h5 : Save a VoxCity model together with simulation results.
    """
    save_results_h5(output_path, city, ground_results=None, building_results=None)


def load_h5(input_path):
    """Load a VoxCity model from an HDF5 file.

    This is the safe counterpart to :func:`load_voxcity` (pickle-based).
    Works with files written by either :func:`save_h5` or
    :func:`save_results_h5` — simulation result groups are simply ignored.

    Parameters
    ----------
    input_path : str or Path
        Path to the HDF5 file.

    Returns
    -------
    VoxCity
        The reconstructed VoxCity model.

    See Also
    --------
    save_h5 : Save a VoxCity model to an HDF5 file.
    load_results_h5 : Load a VoxCity model together with simulation results.
    """
    results = load_results_h5(input_path)
    return results['voxcity']


# =============================================================================
# HDF5 save/load for VoxCity model + simulation results
# =============================================================================

def _serialize_min_heights(min_heights):
    """Serialize object-dtype min_heights array to a flat representation for HDF5.

    Each cell may contain a list of tuples (or an empty list / a scalar).
    We store three datasets:
      * offsets  – int array of length (ny*nx + 1); cell *i* owns
        ``values[offsets[i]:offsets[i+1]]``.
      * values   – flat float array of all numeric values (tuples flattened).
      * n_cols   – number of columns per tuple (0 if empty, typically 2).
    """
    import numpy as np
    flat_values = []
    offsets = [0]
    n_cols = 0

    for idx in np.ndindex(min_heights.shape):
        cell = min_heights[idx]
        if cell is None or (hasattr(cell, '__len__') and len(cell) == 0):
            offsets.append(offsets[-1])
            continue
        # scalar → treat as single value
        if not hasattr(cell, '__len__') and not hasattr(cell, '__iter__'):
            flat_values.append(float(cell))
            offsets.append(offsets[-1] + 1)
            n_cols = max(n_cols, 1)
            continue
        for item in cell:
            if hasattr(item, '__len__'):
                flat_values.extend(float(v) for v in item)
                n_cols = max(n_cols, len(item))
            else:
                flat_values.append(float(item))
                n_cols = max(n_cols, 1)
        n_added = len(flat_values) - offsets[-1]
        offsets.append(offsets[-1] + n_added)

    return (
        np.array(offsets, dtype=np.int64),
        np.array(flat_values, dtype=np.float64),
        n_cols,
    )


def _deserialize_min_heights(offsets, values, n_cols, shape):
    """Reconstruct object-dtype min_heights array from HDF5 flat form."""
    import numpy as np
    arr = np.empty(shape, dtype=object)
    flat_idx = 0
    for idx in np.ndindex(shape):
        start = int(offsets[flat_idx])
        end = int(offsets[flat_idx + 1])
        count = end - start
        if count == 0:
            arr[idx] = []
        elif n_cols <= 1:
            arr[idx] = list(values[start:end])
        else:
            items = []
            for k in range(start, end, n_cols):
                items.append(tuple(values[k:k + n_cols]))
            arr[idx] = items
        flat_idx += 1
    return arr


def save_results_h5(
    output_path,
    city,
    ground_results=None,
    building_results=None,
):
    """Save a VoxCity model and simulation results to an HDF5 file.

    Parameters
    ----------
    output_path : str
        Destination file path (e.g. ``"results.h5"``).
    city : VoxCity
        The VoxCity model instance.
    ground_results : dict, optional
        Ground-level simulation results.  Keys whose values are
        :class:`numpy.ndarray` are stored as HDF5 datasets (with gzip
        compression); all other JSON-serializable values are stored as
        group attributes.

        Typical keys produced by the GPU solar integration module::

            {
                'sunlight_hours':    np.ndarray (ny, nx),
                'cumulative_global': np.ndarray (ny, nx),
                'svf':               np.ndarray (ny, nx),
                'potential_sunlight_hours': float,
                'mode':              str,
                ...
            }

        If the input numpy array carries a ``.metadata`` dict (e.g.
        ``ArrayWithMetadata``), those metadata entries are also persisted
        as sub-attributes.
    building_results : dict, optional
        Building-surface simulation results.  The dict **must** contain
        a ``'mesh'`` key whose value is a Trimesh object.  Per-face data
        arrays are taken either from the ``'metadata'`` sub-dict or
        directly from the Trimesh's ``.metadata`` attribute.

        Typical keys produced by the GPU solar integration module::

            {
                'mesh': trimesh.Trimesh,
                'metadata': {
                    'irradiance_direct':  np.ndarray (n_faces,),
                    'irradiance_diffuse': np.ndarray (n_faces,),
                    'sunlight_hours':     np.ndarray (n_faces,),
                    'potential_sunlight_hours': float,
                    ...
                },
            }

        If ``'metadata'`` is *not* provided explicitly, the function
        falls back to ``mesh.metadata``.

    Notes
    -----
    * Requires the ``h5py`` package (``pip install h5py``).
    * The file is self-contained – no pickle dependency at load time.
    * Datasets use ``compression='gzip'`` by default for compact files.
    """
    import os
    import json
    import numpy as np

    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for HDF5 save/load. "
            "Install it with: pip install h5py"
        )

    from .models import VoxCity as _VoxCity
    if not isinstance(city, _VoxCity):
        raise TypeError("save_results_h5 expects a VoxCity instance as 'city'")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        # -- Root attributes ---------------------------------------------------
        f.attrs['__format__'] = 'voxcity_results.v1'
        f.attrs['crs'] = city.voxels.meta.crs
        f.attrs['meshsize'] = city.voxels.meta.meshsize
        f.attrs['bounds'] = list(city.voxels.meta.bounds)

        # -- VoxCity model group -----------------------------------------------
        vc = f.create_group('voxcity')
        vc.create_dataset('voxel_grid', data=np.asarray(city.voxels.classes), compression='gzip')
        vc.create_dataset('building_height', data=np.asarray(city.buildings.heights), compression='gzip')
        vc.create_dataset('building_id', data=np.asarray(city.buildings.ids), compression='gzip')
        vc.create_dataset('dem', data=np.asarray(city.dem.elevation), compression='gzip')
        vc.create_dataset('land_cover', data=np.asarray(city.land_cover.classes), compression='gzip')

        # min_heights (object-dtype → custom serialization)
        min_h = city.buildings.min_heights
        if min_h is not None:
            mh_grp = vc.create_group('building_min_heights')
            offsets, values, n_cols = _serialize_min_heights(min_h)
            mh_grp.create_dataset('offsets', data=offsets)
            mh_grp.create_dataset('values', data=values)
            mh_grp.attrs['n_cols'] = n_cols
            mh_grp.attrs['shape'] = list(min_h.shape)

        # canopy
        if city.tree_canopy is not None:
            can_grp = vc.create_group('canopy')
            if city.tree_canopy.top is not None:
                can_grp.create_dataset('top', data=np.asarray(city.tree_canopy.top), compression='gzip')
            if city.tree_canopy.bottom is not None:
                can_grp.create_dataset('bottom', data=np.asarray(city.tree_canopy.bottom), compression='gzip')

        # extras – store JSON-serializable entries
        extras = getattr(city, 'extras', None) or {}
        _safe = {}
        for k, v in extras.items():
            try:
                json.dumps(v)
                _safe[k] = v
            except (TypeError, ValueError):
                pass  # skip non-serializable (e.g. GeoDataFrame)
        if _safe:
            vc.attrs['extras_json'] = json.dumps(_safe)

        # -- Ground results group ----------------------------------------------
        if ground_results is not None:
            gr = f.create_group('ground')
            for key, value in ground_results.items():
                if isinstance(value, np.ndarray):
                    gr.create_dataset(key, data=value, compression='gzip')
                    # Persist .metadata from ArrayWithMetadata
                    meta = getattr(value, 'metadata', None)
                    if meta:
                        _store_scalar_attrs(gr[key].attrs, meta, prefix='')
                elif _is_attr_serializable(value):
                    _store_attr(gr.attrs, key, value)

        # -- Building results group --------------------------------------------
        if building_results is not None:
            bg = f.create_group('building')

            # Accept either a dict or a bare trimesh object
            if hasattr(building_results, 'vertices'):
                # Bare trimesh passed directly
                mesh = building_results
                meta_dict = getattr(mesh, 'metadata', {}) or {}
            else:
                mesh = building_results.get('mesh')
                meta_dict = building_results.get('metadata') or {}
                # Fall back to mesh.metadata if metadata not given explicitly
                if not meta_dict and mesh is not None:
                    meta_dict = getattr(mesh, 'metadata', {}) or {}

            if mesh is not None:
                mg = bg.create_group('mesh')
                mg.create_dataset('vertices', data=np.asarray(mesh.vertices), compression='gzip')
                mg.create_dataset('faces', data=np.asarray(mesh.faces), compression='gzip')
                if hasattr(mesh, 'face_normals') and mesh.face_normals is not None:
                    mg.create_dataset('face_normals', data=np.asarray(mesh.face_normals), compression='gzip')

            for key, value in meta_dict.items():
                if isinstance(value, np.ndarray):
                    bg.create_dataset(key, data=value, compression='gzip')
                elif _is_attr_serializable(value):
                    _store_attr(bg.attrs, key, value)

    print(f"VoxCity results saved to {output_path}")


def load_results_h5(input_path):
    """Load a VoxCity model and simulation results from an HDF5 file.

    Parameters
    ----------
    input_path : str
        Path to the HDF5 file created by :func:`save_results_h5`.

    Returns
    -------
    dict
        A dictionary with the following keys:

        ``'voxcity'``
            The reconstructed :class:`~voxcity.models.VoxCity` instance.
        ``'ground'`` *(if present)*
            dict mapping dataset names to numpy arrays, plus scalar
            metadata entries.
        ``'building'`` *(if present)*
            dict with ``'mesh_vertices'``, ``'mesh_faces'``,
            ``'mesh_face_normals'`` (numpy arrays) and per-face data
            arrays / scalar metadata.
        ``'meta'``
            dict with ``'crs'``, ``'meshsize'``, ``'bounds'``.

    Notes
    -----
    The returned ``'voxcity'`` object is a fully reconstructed
    :class:`VoxCity` dataclass – it can be passed directly to any
    simulator or visualizer function.
    """
    import json
    import numpy as np

    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for HDF5 save/load. "
            "Install it with: pip install h5py"
        )

    from .models import GridMetadata, VoxelGrid, BuildingGrid, LandCoverGrid, DemGrid, CanopyGrid, VoxCity

    results = {}

    with h5py.File(input_path, 'r') as f:
        # -- Root metadata -----------------------------------------------------
        crs = str(f.attrs['crs'])
        meshsize = float(f.attrs['meshsize'])
        bounds = tuple(f.attrs['bounds'])
        meta = GridMetadata(crs=crs, bounds=bounds, meshsize=meshsize)
        results['meta'] = {'crs': crs, 'meshsize': meshsize, 'bounds': bounds}

        # -- VoxCity model -----------------------------------------------------
        vc = f['voxcity']
        voxel_grid = vc['voxel_grid'][:]
        building_height = vc['building_height'][:]
        building_id = vc['building_id'][:]
        dem_arr = vc['dem'][:]
        land_cover_arr = vc['land_cover'][:]

        # min_heights
        if 'building_min_heights' in vc:
            mh = vc['building_min_heights']
            offsets = mh['offsets'][:]
            values = mh['values'][:]
            n_cols = int(mh.attrs['n_cols'])
            shape = tuple(mh.attrs['shape'])
            min_heights = _deserialize_min_heights(offsets, values, n_cols, shape)
        else:
            min_heights = np.zeros(building_height.shape)

        # canopy
        canopy_top = None
        canopy_bottom = None
        if 'canopy' in vc:
            if 'top' in vc['canopy']:
                canopy_top = vc['canopy']['top'][:]
            if 'bottom' in vc['canopy']:
                canopy_bottom = vc['canopy']['bottom'][:]

        # extras
        extras = {}
        extras_json = vc.attrs.get('extras_json', None)
        if extras_json:
            extras = json.loads(extras_json)

        voxels = VoxelGrid(classes=voxel_grid, meta=meta)
        buildings = BuildingGrid(
            heights=building_height,
            min_heights=min_heights,
            ids=building_id,
            meta=meta,
        )
        land = LandCoverGrid(classes=land_cover_arr, meta=meta)
        dem_grid = DemGrid(elevation=dem_arr, meta=meta)
        canopy = CanopyGrid(top=canopy_top, bottom=canopy_bottom, meta=meta)

        results['voxcity'] = VoxCity(
            voxels=voxels,
            buildings=buildings,
            land_cover=land,
            dem=dem_grid,
            tree_canopy=canopy,
            extras=extras,
        )

        # -- Ground results ----------------------------------------------------
        if 'ground' in f:
            ground = {}
            gr = f['ground']
            # Datasets → numpy arrays
            for key in gr:
                if isinstance(gr[key], h5py.Dataset):
                    ground[key] = gr[key][:]
            # Group attributes → scalars
            for key, value in gr.attrs.items():
                ground[key] = _decode_attr(value)
            results['ground'] = ground

        # -- Building results --------------------------------------------------
        if 'building' in f:
            building = {}
            bg = f['building']

            # Mesh geometry
            if 'mesh' in bg:
                building['mesh_vertices'] = bg['mesh']['vertices'][:]
                building['mesh_faces'] = bg['mesh']['faces'][:]
                if 'face_normals' in bg['mesh']:
                    building['mesh_face_normals'] = bg['mesh']['face_normals'][:]

            # Per-face datasets
            for key in bg:
                if key == 'mesh':
                    continue
                if isinstance(bg[key], h5py.Dataset):
                    building[key] = bg[key][:]

            # Scalar attributes
            for key, value in bg.attrs.items():
                building[key] = _decode_attr(value)

            results['building'] = building

    return results


# =============================================================================
# HDF5 attribute helpers (private)
# =============================================================================

def _is_attr_serializable(value):
    """Check if a value can be stored as an HDF5 attribute."""
    return isinstance(value, (int, float, str, bool, list, tuple))


def _store_attr(attrs, key, value):
    """Store a single value as an HDF5 attribute."""
    import numpy as np
    if isinstance(value, (list, tuple)):
        # HDF5 attributes handle homogeneous numeric lists natively
        try:
            attrs[key] = value
        except TypeError:
            import json
            attrs[key] = json.dumps(value)
    elif isinstance(value, np.generic):
        attrs[key] = value.item()
    else:
        attrs[key] = value


def _store_scalar_attrs(attrs, meta_dict, prefix=''):
    """Store scalar entries from a metadata dict as HDF5 attributes."""
    import numpy as np
    for k, v in meta_dict.items():
        if isinstance(v, np.ndarray):
            continue  # skip arrays – caller handles them
        if _is_attr_serializable(v):
            _store_attr(attrs, f"{prefix}{k}", v)


def _decode_attr(value):
    """Decode an HDF5 attribute value back to a Python type."""
    import numpy as np
    if isinstance(value, bytes):
        return value.decode('utf-8')
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
