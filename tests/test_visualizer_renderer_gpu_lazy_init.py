"""
Tests for the lazy Taichi initialization behavior of
``voxcity.visualizer.renderer_gpu``.

The renderer module used to call ``ti.init`` at import time, which bound
Taichi's CUDA context to the importing thread and broke host applications
that drive Taichi from a dedicated worker thread. The module was changed
to defer initialization to the constructors of the renderer classes via
``voxcity.simulator_gpu.init_taichi.ensure_initialized``.

These tests verify that contract using AST inspection so they can run
without a GPU (or even without ``taichi`` installed).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


RENDERER_GPU_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "voxcity"
    / "visualizer"
    / "renderer_gpu.py"
)


@pytest.fixture(scope="module")
def renderer_module_ast() -> ast.Module:
    source = RENDERER_GPU_PATH.read_text(encoding="utf-8")
    return ast.parse(source, filename=str(RENDERER_GPU_PATH))


def _iter_calls(node: ast.AST):
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            yield child


def _is_ti_init_call(call: ast.Call) -> bool:
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "init"
        and isinstance(func.value, ast.Name)
        and func.value.id == "ti"
    )


def _is_ensure_initialized_call(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Name) and func.id == "ensure_initialized":
        return True
    if isinstance(func, ast.Attribute) and func.attr == "ensure_initialized":
        return True
    return False


def _find_class(module: ast.Module, name: str) -> ast.ClassDef:
    for node in ast.walk(module):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name!r} not found in renderer_gpu.py")


def _find_init(cls: ast.ClassDef) -> ast.FunctionDef:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            return node
    raise AssertionError(f"class {cls.name!r} has no __init__ method")


class TestRendererGpuLazyInit:
    """Verify Taichi is initialized lazily, not at module import."""

    def test_module_does_not_call_ti_init_at_import(
        self, renderer_module_ast: ast.Module
    ) -> None:
        """No ``ti.init(...)`` call should remain anywhere in the module.

        The previous implementation invoked ``ti.init`` inside the top-level
        ``try: import taichi`` block. That eager init has been removed and
        replaced with lazy initialization in the renderer constructors.
        """
        offending = [
            call
            for call in _iter_calls(renderer_module_ast)
            if _is_ti_init_call(call)
        ]
        assert not offending, (
            "renderer_gpu.py must not call ti.init() (it must be deferred "
            f"to ensure_initialized()); found {len(offending)} call(s) at "
            f"line(s) {[c.lineno for c in offending]}"
        )

    @pytest.mark.parametrize(
        "class_name",
        ["TriangleBVH", "TaichiRenderer", "GPURenderer"],
    )
    def test_constructor_calls_ensure_initialized(
        self, renderer_module_ast: ast.Module, class_name: str
    ) -> None:
        """Every renderer entry-point class lazily initializes Taichi."""
        cls = _find_class(renderer_module_ast, class_name)
        init = _find_init(cls)

        calls = [c for c in _iter_calls(init) if _is_ensure_initialized_call(c)]
        assert calls, (
            f"{class_name}.__init__ must call ensure_initialized() to "
            "lazily initialize Taichi"
        )

    def test_ensure_initialized_imported_from_simulator_gpu(
        self, renderer_module_ast: ast.Module
    ) -> None:
        """The lazy-init helper must come from the canonical location."""
        expected_module_suffix = "simulator_gpu.init_taichi"
        found = False
        for node in ast.walk(renderer_module_ast):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module is None:
                continue
            if not node.module.endswith(expected_module_suffix):
                continue
            if any(alias.name == "ensure_initialized" for alias in node.names):
                found = True
                break
        assert found, (
            "renderer_gpu.py must import ensure_initialized from "
            f"...{expected_module_suffix}"
        )


class TestEnsureInitializedHelper:
    """Sanity checks for the helper that renderer_gpu now relies on."""

    def test_helper_is_importable_and_idempotent(self) -> None:
        pytest.importorskip("taichi")
        import importlib

        it = importlib.import_module("voxcity.simulator_gpu.init_taichi")

        # Calling ensure_initialized() twice must not raise; the second call
        # is expected to be a no-op regardless of whether Taichi was already
        # initialized by another test in the session.
        it.ensure_initialized()
        assert it.is_initialized() is True
        it.ensure_initialized()
        assert it.is_initialized() is True
