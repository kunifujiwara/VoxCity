"""Lazy third-party module loading built on ``importlib.util.LazyLoader``."""

import importlib.util
import sys


def lazy_import(name: str):
    """Return module ``name`` without executing it until first attribute access.

    Already-imported modules are returned as-is. A missing module raises
    ``ModuleNotFoundError`` immediately at the call site, so optional
    dependencies still fail with a clear error where they are declared.
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.find_spec(name)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"No module named {name!r}")
    spec.loader = importlib.util.LazyLoader(spec.loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
