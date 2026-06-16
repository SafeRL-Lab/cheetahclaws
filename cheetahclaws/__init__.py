"""CheetahClaws — package root.

Kept intentionally light.  Importing ``cheetahclaws`` — or any submodule such
as ``cheetahclaws.config`` — must NOT trigger the CLI's heavy module-level
setup in :mod:`cheetahclaws.cli` (``.env`` loading, ``stdout``/``stderr``
wrapping, the command-table build).  The CLI entry symbols (``cmd_*``,
``info``, ``err`` …) are exposed lazily through :func:`__getattr__`, so
``from cheetahclaws import cmd_init`` keeps working without paying that cost
on every submodule import.

Why a package at all: the modules used to live at the top level (``config``,
``daemon``, ``kernel`` …).  Generic names like ``config`` / ``daemon`` collide
with other things on ``sys.path`` (another project's ``config/`` dir, the
``python-daemon`` package) once CheetahClaws is *installed* and launched from
its entry point rather than the repo dir.  Owning a single ``cheetahclaws.*``
namespace removes that entire class of import-shadowing bug.
"""
from __future__ import annotations

from pathlib import Path


def _read_version() -> str:
    """Resolve the version: installed metadata first, pyproject.toml fallback."""
    try:
        from importlib.metadata import version as _v
        return _v("cheetahclaws")
    except Exception:
        pass
    try:
        # pyproject.toml sits at the repo root, one level above this package.
        _toml = Path(__file__).resolve().parent.parent / "pyproject.toml"
        for _line in _toml.read_text(encoding="utf-8").splitlines():
            if _line.startswith("version"):
                return _line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return "0.0.0"


VERSION = __version__ = _read_version()


def __getattr__(name: str):
    """Resolve ``cheetahclaws.<name>`` lazily (PEP 562).

    Submodules (``config``, ``daemon``, ``kernel`` …) are imported directly —
    crucially *without* touching :mod:`cheetahclaws.cli`, otherwise a
    ``from cheetahclaws import config`` would drag in the heavy CLI module and
    recurse through its own ``from cheetahclaws import <submodule>`` lines.
    Only names that are not submodules fall back to proxying a CLI entry symbol
    (``cmd_init``, ``info``, ``err`` …).
    """
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    import importlib
    try:
        return importlib.import_module(f"{__name__}.{name}")
    except ImportError:
        pass
    from . import cli
    try:
        return getattr(cli, name)
    except AttributeError:
        raise AttributeError(
            f"module 'cheetahclaws' has no attribute {name!r}") from None
