"""Regression test for issue #97 — `pip install .` produces a wheel that
omits `prompts` (and other) packages, breaking `cheetahclaws` at startup.

Root cause: `pyproject.toml` listed `memory` in BOTH `py-modules` and
`packages`. setuptools ≥ 75 on Windows treats this as a hard error and
silently drops unrelated packages from the wheel; `memory.py` is also a
backward-compatibility shim shadowed at import time by the `memory/`
package, so it's dead code.

Fix: remove the shim, drop `memory` from `py-modules`, switch to
`[tool.setuptools.packages.find]` so future sub-packages auto-discover.

These tests verify:
  1. The pyproject.toml config is internally consistent
  2. Every directory with an __init__.py at the project root is reachable
     under `find`'s include patterns
  3. The dead `memory.py` shim is gone
  4. All Python packages we reference are actually importable from the
     installed editable / wheel-built copy
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read_pyproject() -> dict:
    """Parse pyproject.toml.

    `tomllib` is 3.11+ stdlib; on 3.10 we fall back to `tomli` (the
    backport, which the project's Python 3.10 wheels already depend on
    transitively via setuptools).  pyproject.toml `requires-python` is
    `>=3.10`, so we must not assume 3.11.
    """
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[import-not-found, no-redef]
    with open(_PROJECT_ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


# ── Configuration sanity ────────────────────────────────────────────────

def test_pyproject_no_module_package_collision():
    """A name listed in py-modules must not also be a directory package."""
    cfg = _read_pyproject()
    py_modules = cfg["tool"]["setuptools"].get("py-modules", [])
    for name in py_modules:
        pkg_dir = _PROJECT_ROOT / name
        if pkg_dir.is_dir() and (pkg_dir / "__init__.py").is_file():
            pytest.fail(
                f"`{name}` is in py-modules but also exists as a package "
                f"({pkg_dir}/__init__.py). This caused issue #97. "
                f"Remove from py-modules."
            )


def test_no_dead_memory_shim():
    """memory.py shim is dead code — Python prefers memory/ package."""
    p = _PROJECT_ROOT / "memory.py"
    assert not p.exists(), (
        f"{p} re-introduced. It collides with the memory/ package "
        f"(see issue #97) and Python silently shadows it. Use "
        f"`from memory import ...` directly via memory/__init__.py instead."
    )


def test_pyproject_uses_find_for_packages():
    """Modern config uses find to auto-include new sub-packages."""
    cfg = _read_pyproject()
    setuptools_cfg = cfg["tool"]["setuptools"]
    find = setuptools_cfg.get("packages", {}).get("find") if isinstance(
        setuptools_cfg.get("packages"), dict) else None
    # tomllib parses [tool.setuptools.packages.find] as nested dict
    if find is None:
        find = cfg["tool"]["setuptools"].get("packages", {}).get("find")
    assert find is not None, (
        "pyproject.toml should use [tool.setuptools.packages.find] "
        "with `include` patterns so new sub-packages are picked up "
        "automatically (regression guard for issue #97 follow-up)."
    )
    assert "include" in find, "find must specify include patterns"
    # Sanity: tests/ must be excluded
    assert "tests*" in find.get("exclude", []), (
        "tests/ must be excluded from packaged wheels"
    )


# ── Discovery: are all dir-packages reachable from the include patterns? ─

def _matches_any(name: str, patterns: list[str]) -> bool:
    """Trivial wildcard match: 'foo*' matches anything starting with 'foo'."""
    for p in patterns:
        if p.endswith("*"):
            if name == p[:-1] or name.startswith(p[:-1]):
                return True
        elif name == p:
            return True
    return False


def test_every_top_level_package_dir_reachable_by_find():
    """Walk top-level dirs with __init__.py — each must match include or exclude."""
    cfg = _read_pyproject()
    find = cfg["tool"]["setuptools"]["packages"]["find"]
    include = find.get("include", [])
    exclude = find.get("exclude", [])

    skip_names = {"build", "dist", "__pycache__", ".git", ".venv", "venv",
                  ".pytest_cache", "fixtures", "research_papers", "logs",
                  "research_reports", "report_outputs"}

    missing = []
    for child in _PROJECT_ROOT.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name in skip_names:
            continue
        if child.name.endswith(".egg-info"):
            continue
        init = child / "__init__.py"
        if not init.is_file():
            continue
        # This is a Python package. It must match include and not match exclude.
        if _matches_any(child.name, exclude):
            continue
        if not _matches_any(child.name, include):
            missing.append(child.name)

    assert not missing, (
        f"These top-level packages have __init__.py but aren't reached "
        f"by pyproject.toml's [tool.setuptools.packages.find] include "
        f"patterns: {missing}. Add to `include` so `pip install .` "
        f"actually ships them."
    )


# ── Importability: every advertised package must import without error ────

# These imports must always succeed in a healthy install. If any of them
# raise ImportError, the user's `pip install .` produced a broken wheel.
# Everything now lives under the single ``cheetahclaws`` package — that is the
# whole point of the layout (no generic top-level name to be shadowed).
_REQUIRED_IMPORTS = [
    "cheetahclaws",
    "cheetahclaws.prompts",
    "cheetahclaws.prompts.select",
    "cheetahclaws.memory",
    "cheetahclaws.memory.context",
    "cheetahclaws.ui",
    "cheetahclaws.web",
    "cheetahclaws.bridges",
    "cheetahclaws.commands",
    "cheetahclaws.research",
    "cheetahclaws.research.lab",
    "cheetahclaws.modular",
    "cheetahclaws.modular.trading",
    "cheetahclaws.modular.trading.data",
    "cheetahclaws.modular.trading.engines",
    "cheetahclaws.modular.trading.agents",
    "cheetahclaws.modular.trading.alt_data",
    "cheetahclaws.modular.trading.broker",
    "cheetahclaws.modular.trading.discover",
    "cheetahclaws.modular.trading.ml",
    "cheetahclaws.modular.video",
    "cheetahclaws.modular.voice",
    "cheetahclaws.context",
    "cheetahclaws.providers",
]


@pytest.mark.parametrize("modname", _REQUIRED_IMPORTS)
def test_required_module_imports(modname):
    """Each must import without ModuleNotFoundError. This is the exact
    failure mode reported in issue #97."""
    import importlib
    try:
        importlib.import_module(modname)
    except ModuleNotFoundError as e:
        pytest.fail(
            f"Cannot import `{modname}` — `pip install .` would ship a "
            f"broken wheel for end users (issue #97 regression). "
            f"Original error: {e}"
        )


def test_prompts_exports_pick_base_prompt():
    """The exact symbol context.py needs (failing line in issue #97)."""
    from cheetahclaws.prompts import pick_base_prompt, load_fragment  # noqa: F401
