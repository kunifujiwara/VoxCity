"""On-disk cache for downloaded geospatial data.

Downloads are keyed by the wrapped function's fully-qualified name plus ALL of
its bound arguments (positional and keyword, matched against the function's
signature so equivalent calls hash the same regardless of calling
convention). Parameters named ``output_dir`` or ``download_dir`` are excluded
from the key, since they only control where files are staged on disk and do
not affect the downloaded data itself -- two calls that differ only in
``output_dir`` should hit the same cache entry. Results are stored as pickles
under a per-user cache directory. The cache is best-effort: any read/write
failure falls back to a fresh download. Layout:

    <cache_dir>/downloads/<key>.pkl        # pickled return value
    <cache_dir>/downloads/<key>.meta.json  # source, args, created timestamp

The cache directory resolves, in order: the VOXCITY_CACHE_DIR environment
variable, else ``~/.voxcity/cache``.
"""

import functools
import hashlib
import inspect
import json
import os
import pickle
import time
from pathlib import Path

from .logging import get_logger

_logger = get_logger(__name__)

# Parameters that only control where files are staged on disk, not what data
# is downloaded -- excluded from the cache key so a different output_dir
# still hits the same cache entry.
_EXCLUDED_PARAMS = {"output_dir", "download_dir"}


def get_cache_dir() -> Path:
    """Return (creating if needed) the directory downloads are cached under."""
    root = os.environ.get("VOXCITY_CACHE_DIR")
    base = Path(root) if root else Path.home() / ".voxcity" / "cache"
    d = base / "downloads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(func, args, kwargs) -> str:
    """Build a cache key from all bound arguments of ``func``.

    Binds ``args``/``kwargs`` against ``func``'s signature so the key is
    independent of whether a given argument was passed positionally or by
    keyword, then drops directory-only parameters (see
    ``_EXCLUDED_PARAMS``) before hashing. Falls back to hashing the raw
    args/kwargs if signature binding fails for any reason.
    """
    try:
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        key_args = {
            name: value
            for name, value in bound.arguments.items()
            if name not in _EXCLUDED_PARAMS
        }
    except TypeError:
        key_args = {"args": args, "kwargs": kwargs}
    payload = json.dumps(key_args, default=str, sort_keys=True)
    digest_input = f"{func.__qualname__}:{payload}".encode()
    return hashlib.blake2b(digest_input, digest_size=16).hexdigest()


def cached_download(func):
    """Cache a downloader's return value on disk, keyed by its arguments.

    The wrapped function gains two keyword-only controls (consumed here, not
    forwarded to ``func``): ``use_download_cache`` (default True) and
    ``force_refresh`` (default False).
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        use_cache = kwargs.pop("use_download_cache", True)
        force_refresh = kwargs.pop("force_refresh", False)
        if not use_cache:
            return func(*args, **kwargs)

        key = _cache_key(func, args, kwargs)
        path = get_cache_dir() / f"{key}.pkl"

        if path.exists() and not force_refresh:
            try:
                with open(path, "rb") as f:
                    value = pickle.load(f)
                _logger.info("Using cached %s result (%s)", func.__name__, path.name)
                return value
            except Exception as e:  # corrupt cache: fall through to refresh
                _logger.info("Cache read failed for %s (%s); re-downloading", func.__name__, e)

        value = func(*args, **kwargs)
        try:
            with open(path, "wb") as f:
                pickle.dump(value, f)
            meta = {"function": func.__qualname__, "created": time.time()}
            with open(path.with_suffix(".meta.json"), "w") as f:
                json.dump(meta, f)
        except Exception as e:  # cache write is best-effort
            _logger.info("Cache write failed for %s: %s", func.__name__, e)
        return value

    return wrapper


def clear_download_cache() -> int:
    """Delete all cached downloads. Returns the number of files removed."""
    n = 0
    for p in get_cache_dir().glob("*"):
        try:
            p.unlink()
            n += 1
        except OSError:
            pass
    return n
