"""tools_notebook.py — NotebookEdit tool implementation."""
from __future__ import annotations

import json
import re
from pathlib import Path


def _parse_cell_id(cell_id: str) -> int | None:
    """Convert 'cell-N' shorthand to integer index; return None otherwise."""
    m = re.fullmatch(r"cell-(\d+)", cell_id)
    return int(m.group(1)) if m else None


def _notebook_edit(
    notebook_path: str,
    new_source: str,
    cell_id: str = None,
    cell_type: str = None,
    edit_mode: str = "replace",
) -> str:
    p = Path(notebook_path)
    if p.suffix != ".ipynb":
        return "Error: file must be a Jupyter notebook (.ipynb)"
    if not p.exists():
        return f"Error: notebook not found: {notebook_path}"

    try:
        nb = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return f"Error: notebook is not valid JSON: {e}"

    cells = nb.get("cells", [])

    def _resolve_index(cid: str) -> int | None:
        for i, c in enumerate(cells):
            if c.get("id") == cid:
                return i
        idx = _parse_cell_id(cid)
        if idx is not None and 0 <= idx < len(cells):
            return idx
        return None

    if edit_mode == "replace":
        if not cell_id:
            return "Error: cell_id is required for replace"
        idx = _resolve_index(cell_id)
        if idx is None:
            return f"Error: cell '{cell_id}' not found"
        target = cells[idx]
        target["source"] = new_source
        if cell_type and cell_type != target.get("cell_type"):
            target["cell_type"] = cell_type
        if target.get("cell_type") == "code":
            target["execution_count"] = None
            target["outputs"] = []

    elif edit_mode == "insert":
        if not cell_type:
            return "Error: cell_type is required for insert ('code' or 'markdown')"
        nbformat       = nb.get("nbformat", 4)
        nbformat_minor = nb.get("nbformat_minor", 0)
        use_ids        = nbformat > 4 or (nbformat == 4 and nbformat_minor >= 5)
        new_id         = None
        if use_ids:
            import random, string
            new_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))

        if cell_type == "markdown":
            new_cell = {"cell_type": "markdown", "source": new_source, "metadata": {}}
        else:
            new_cell = {
                "cell_type": "code", "source": new_source, "metadata": {},
                "execution_count": None, "outputs": [],
            }
        if use_ids and new_id:
            new_cell["id"] = new_id

        if cell_id:
            idx = _resolve_index(cell_id)
            if idx is None:
                return f"Error: cell '{cell_id}' not found"
            cells.insert(idx + 1, new_cell)
        else:
            cells.insert(0, new_cell)
        nb["cells"] = cells
        cell_id = new_id or cell_id

    elif edit_mode == "delete":
        if not cell_id:
            return "Error: cell_id is required for delete"
        idx = _resolve_index(cell_id)
        if idx is None:
            return f"Error: cell '{cell_id}' not found"
        cells.pop(idx)
    else:
        return f"Error: unknown edit_mode '{edit_mode}' — use replace, insert, or delete"

    nb["cells"] = cells
    p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    if edit_mode == "delete":
        return f"Deleted cell '{cell_id}' from {notebook_path}"
    return f"NotebookEdit({edit_mode}) applied to cell '{cell_id}' in {notebook_path}"
