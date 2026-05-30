"""Recursive directory tree builder with file descriptions."""
from __future__ import annotations

import os
from pathlib import Path

from folder_desc.describer import describe_files_parallel

SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "node_modules", ".tox",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    ".egg-info", ".eggs", ".nano_claude",
}

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".rb",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".php", ".swift", ".kt",
    ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd",
    ".yaml", ".yml", ".toml", ".json", ".xml", ".ini", ".cfg",
    ".md", ".rst", ".txt",
    ".html", ".css", ".scss", ".less",
    ".sql", ".r", ".R", ".lua", ".zig", ".nim",
    ".dockerfile", ".Dockerfile",
}

MAX_FILES = 500


def _is_code_file(path: Path) -> bool:
    if path.suffix.lower() in CODE_EXTENSIONS:
        return True
    if path.name in ("Makefile", "Dockerfile", "Jenkinsfile", "Procfile", ".gitignore"):
        return True
    return False


def _collect_files(folder: Path) -> list[Path]:
    files: list[Path] = []

    def _walk(current: Path, depth: int = 0) -> None:
        if depth > 10 or len(files) >= MAX_FILES:
            return
        try:
            entries = sorted(current.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except OSError:
            return
        for entry in entries:
            if entry.is_dir():
                if entry.name in SKIP_DIRS or entry.name.startswith("."):
                    continue
                _walk(entry, depth + 1)
            elif entry.is_file() and _is_code_file(entry):
                files.append(entry)

    _walk(folder)
    return files


def _build_tree_string(folder: Path, descriptions: dict[str, str]) -> str:
    lines: list[str] = []
    folder_str = str(folder)

    def _walk(current: Path, prefix: str = "", depth: int = 0) -> None:
        if depth > 10:
            return
        try:
            entries = sorted(current.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except OSError:
            return

        visible = []
        for entry in entries:
            if entry.is_dir():
                if entry.name in SKIP_DIRS or entry.name.startswith("."):
                    continue
                visible.append(entry)
            elif entry.is_file() and _is_code_file(entry):
                visible.append(entry)

        for i, entry in enumerate(visible):
            is_last = i == len(visible) - 1
            connector = "`-- " if is_last else "|-- "
            child_prefix = prefix + ("    " if is_last else "|   ")

            if entry.is_dir():
                lines.append(f"{prefix}{connector}{entry.name}/")
                _walk(entry, child_prefix, depth + 1)
            else:
                desc = descriptions.get(str(entry), "")
                desc_tag = f"  [desc] {desc} [/desc]" if desc else ""
                lines.append(f"{prefix}{connector}{entry.name}{desc_tag}")

    lines.append(f"{folder.name}/")
    _walk(folder)
    return "\n".join(lines)


def get_folder_description(folder_path: str, config: dict | None = None) -> str:
    folder = Path(folder_path)
    if not folder.is_dir():
        return f"Error: {folder_path} is not a directory"

    files = _collect_files(folder)
    if not files:
        return f"{folder.name}/ (empty or no code files found)"

    file_paths = [str(f) for f in files]
    descriptions = describe_files_parallel(file_paths, config)
    tree = _build_tree_string(folder, descriptions)

    return f"{len(files)} code files found.\n\n{tree}"
