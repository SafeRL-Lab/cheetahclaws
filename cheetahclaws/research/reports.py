"""Saved research reports — auto-save to `~/.cheetahclaws/research_reports/`,
list/load/export from there.

Each run writes two sibling files:
    <YYYY-MM-DD_HHMMSS>-<slug>.md    — the rendered brief (what user sees)
    <YYYY-MM-DD_HHMMSS>-<slug>.json  — serialized Brief for later inspection

`--save-as <path>` copies the .md to a user-chosen path AND still keeps the
auto-saved copy in the reports dir.

`/reports list` shows the 50 most recent; `/reports open <id>` prints a
saved report; `/reports delete <id>` removes it.
"""
from __future__ import annotations

import dataclasses
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

from .types import Brief


def _reports_dir() -> Path:
    d = Path.home() / ".cheetahclaws" / "research_reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _slug(topic: str, maxlen: int = 50) -> str:
    s = re.sub(r"[^\w\-]+", "-", topic.strip(), flags=re.UNICODE).strip("-")
    return (s or "untitled")[:maxlen]


def save(brief: Brief, rendered_markdown: str, notable: list | None = None,
         also_save_as: str | None = None) -> Path:
    """Write the brief to the reports dir. Returns the .md path.

    notable: optional list of NotableCiter dataclasses (serialized into the
    JSON sidecar for later `/reports open`).
    """
    d = _reports_dir()
    d.mkdir(parents=True, exist_ok=True)   # belt-and-braces — helps monkeypatched dirs
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    stem = f"{ts}-{_slug(brief.topic)}"
    md_path = d / f"{stem}.md"
    json_path = d / f"{stem}.json"

    md_path.write_text(rendered_markdown, encoding="utf-8")

    notable_data = []
    if notable:
        for n in notable:
            notable_data.append(dataclasses.asdict(n))

    sidecar = {
        "topic": brief.topic,
        "domains": brief.domains,
        "created_at": datetime.now().isoformat(),
        "total_duration_ms": brief.total_duration_ms,
        "cache_hits": brief.cache_hits,
        "results": [dataclasses.asdict(r) for r in brief.results],
        "statuses": [dataclasses.asdict(s) for s in brief.statuses],
        "synthesis": brief.synthesis,
        "notable_citers": notable_data,
        "rendered_markdown_file": md_path.name,
    }
    json_path.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if also_save_as:
        target = Path(also_save_as).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(md_path), str(target))

    return md_path


def list_reports(limit: int = 50) -> list[dict]:
    """Return metadata for the N most recent reports, newest first."""
    d = _reports_dir()
    json_files = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out: list[dict] = []
    for i, jp in enumerate(json_files[:limit], start=1):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append({
            "id": i,
            "stem": jp.stem,
            "topic": data.get("topic", ""),
            "created_at": data.get("created_at", ""),
            "domains": data.get("domains", []),
            "results_count": len(data.get("results", [])),
            "sources_ok": sum(1 for s in data.get("statuses", []) if s.get("ok")),
            "md_path": str(d / f"{jp.stem}.md"),
            "json_path": str(jp),
            "size_kb": round(jp.stat().st_size / 1024, 1),
        })
    return out


def get_by_id(report_id: int) -> dict | None:
    reports = list_reports(limit=200)
    for r in reports:
        if r["id"] == report_id:
            return r
    return None


def get_by_stem(stem: str) -> dict | None:
    d = _reports_dir()
    jp = d / f"{stem}.json"
    mp = d / f"{stem}.md"
    if not jp.exists():
        return None
    try:
        data = json.loads(jp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return {
        "stem": stem,
        "topic": data.get("topic", ""),
        "created_at": data.get("created_at", ""),
        "domains": data.get("domains", []),
        "results_count": len(data.get("results", [])),
        "sources_ok": sum(1 for s in data.get("statuses", []) if s.get("ok")),
        "md_path": str(mp),
        "json_path": str(jp),
        "size_kb": round(jp.stat().st_size / 1024, 1),
    }


def delete(report_id: int) -> bool:
    r = get_by_id(report_id)
    if not r:
        return False
    for k in ("md_path", "json_path"):
        p = Path(r[k])
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass
    return True


def read_markdown(report_id: int | None = None,
                  stem: str | None = None) -> str | None:
    info = None
    if report_id is not None:
        info = get_by_id(report_id)
    elif stem:
        info = get_by_stem(stem)
    if not info:
        return None
    p = Path(info["md_path"])
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return None
