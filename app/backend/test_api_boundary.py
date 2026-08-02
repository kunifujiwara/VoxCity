"""The app may only consume voxcity through public names.

Private imports (any ``voxcity...._x`` module, or a ``_x`` name pulled out of
a public voxcity module) are how the last split produced silent cross-repo
drift: the library refactors its internals freely, and nothing on either side
notices until runtime. Grep-based on purpose — it guards the import *text*,
so it fails at review time, not at import time.

Scope decisions:
- ``app/archive/`` is excluded: frozen prototypes, not shipped, not maintained.
- ``.ipynb`` files under app/ (e.g. ``app/preprocessing/tokyo_las.ipynb``) ARE
  scanned: preprocessing notebooks import voxcity and would drift exactly the
  same way as .py modules if they reached into private modules.
"""
import json
import re
from pathlib import Path

APP = Path(__file__).resolve().parents[1]          # app/

# `from voxcity.a._b import c` / `import voxcity.a._b`
PRIVATE_MODULE = re.compile(r"^\s*(?:from|import)\s+voxcity[\w.]*\._\w+", re.M)
# `from voxcity.a import _b` (possibly parenthesised / multi-line list)
FROM_VOX = re.compile(
    r"^[ \t]*from[ \t]+voxcity[\w.]*[ \t]+import[ \t]+(\([^)]*\)|[^\n]*)", re.M
)


def _private_names_in_import_list(blob: str) -> list[str]:
    """Return imported names starting with '_' from a from-import name list."""
    names = blob.strip().strip("()")
    offenders = []
    for part in names.split(","):
        base = part.split("#")[0].strip().split(" as ")[0].strip()
        if base.startswith("_"):
            offenders.append(base)
    return offenders


def _scan_source(text: str) -> list[str]:
    hits = PRIVATE_MODULE.findall(text)
    for m in FROM_VOX.finditer(text):
        hits.extend(_private_names_in_import_list(m.group(1)))
    return [h.strip() for h in hits]


def _iter_sources():
    for py in APP.rglob("*.py"):
        if "archive" in py.parts or "node_modules" in py.parts:
            continue
        yield py, py.read_text(encoding="utf-8", errors="replace")
    for nb in APP.rglob("*.ipynb"):
        if "archive" in nb.parts or "node_modules" in nb.parts or ".ipynb_checkpoints" in nb.parts:
            continue
        try:
            cells = json.loads(nb.read_text(encoding="utf-8", errors="replace")).get("cells", [])
        except (json.JSONDecodeError, AttributeError):
            continue
        src = "\n".join(
            "".join(c.get("source", [])) for c in cells if c.get("cell_type") == "code"
        )
        yield nb, src


def test_no_private_voxcity_imports_anywhere_under_app():
    offenders = []
    for path, text in _iter_sources():
        m = _scan_source(text)
        if m:
            offenders.append((str(path.relative_to(APP)), m))
    assert not offenders, f"private voxcity imports under app/: {offenders}"
