"""LLM-based file description generator with parallel execution."""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from folder_desc.cache import get_cached_desc, set_cached_desc

_DESC_RE = re.compile(r"#\s*\[desc\]\s*(.+?)\s*\[/desc\]")
_MAX_PREVIEW_LINES = 100
_MAX_WORKERS = 8


def extract_inline_desc(file_path: str) -> str | None:
    """Return the `# [desc] ... [/desc]` tag on the first line, or None."""
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            first_line = next(iter(f), "")
    except OSError:
        return None  # unreadable file = no inline description
    m = _DESC_RE.search(first_line)
    return m.group(1).strip() if m else None


def _read_preview(file_path: str) -> str:
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= _MAX_PREVIEW_LINES:
                    break
                lines.append(line)
            return "".join(lines)
    except OSError:
        return ""


def describe_file(file_path: str, config: dict | None = None) -> str:
    inline = extract_inline_desc(file_path)
    if inline:
        set_cached_desc(file_path, inline)
        return inline

    cached = get_cached_desc(file_path)
    if cached:
        return cached

    preview = _read_preview(file_path)
    if not preview.strip():
        return "Empty file"

    desc = _call_llm_for_desc(file_path, preview, config)
    set_cached_desc(file_path, desc)
    return desc


def _call_llm_for_desc(file_path: str, preview: str, config: dict | None) -> str:
    try:
        from auxiliary import stream_auxiliary
        name = Path(file_path).name
        prompt = (
            f"Describe what the file '{name}' does in ONE short sentence (max 15 words). "
            f"No markdown, no quotes, just the description.\n\n```\n{preview[:3000]}\n```"
        )
        result = stream_auxiliary(
            system="You generate concise one-line file descriptions.",
            messages=[{"role": "user", "content": prompt}],
            config=config or {},
        )
        return result.strip().rstrip(".")
    except Exception:
        return f"({Path(file_path).suffix or 'unknown'} file)"


def describe_files_parallel(
    file_paths: list[str], config: dict | None = None,
) -> dict[str, str]:
    results: dict[str, str] = {}
    to_describe: list[str] = []

    for fp in file_paths:
        inline = extract_inline_desc(fp)
        if inline:
            results[fp] = inline
            set_cached_desc(fp, inline)
            continue
        cached = get_cached_desc(fp)
        if cached:
            results[fp] = cached
            continue
        to_describe.append(fp)

    if not to_describe:
        return results

    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(to_describe))) as pool:
        futures = {pool.submit(describe_file, fp, config): fp for fp in to_describe}
        for future in as_completed(futures):
            fp = futures[future]
            try:
                results[fp] = future.result()
            except Exception:
                results[fp] = "(description unavailable)"

    return results
