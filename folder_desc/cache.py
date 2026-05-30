"""JSON-based cache for file descriptions."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

CACHE_DIR = Path.home() / ".cheetahclaws" / "folder_desc_cache"


def _cache_key(file_path: str) -> str:
    return hashlib.sha256(file_path.encode()).hexdigest()[:16]


def _cache_path(file_path: str) -> Path:
    return CACHE_DIR / f"{_cache_key(file_path)}.json"


def get_cached_desc(file_path: str) -> str | None:
    cp = _cache_path(file_path)
    if not cp.exists():
        return None
    try:
        data = json.loads(cp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    try:
        stat = os.stat(file_path)
    except OSError:
        return None
    if data.get("mtime") != stat.st_mtime or data.get("size") != stat.st_size:
        return None
    return data.get("desc")


def set_cached_desc(file_path: str, desc: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        stat = os.stat(file_path)
    except OSError:
        return
    data = {"desc": desc, "mtime": stat.st_mtime, "size": stat.st_size, "path": file_path}
    _cache_path(file_path).write_text(json.dumps(data), encoding="utf-8")


def clear_cache() -> int:
    if not CACHE_DIR.exists():
        return 0
    count = 0
    for f in CACHE_DIR.glob("*.json"):
        f.unlink()
        count += 1
    return count
