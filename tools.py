"""Tool definitions and implementations for CheetahClaws."""
import json
import os
import re
import glob as _glob
import difflib
import subprocess
import threading
from pathlib import Path
from typing import Callable, Optional

from tool_registry import ToolDef, register_tool
from tool_registry import execute_tool as _registry_execute

# ── AskUserQuestion state ──────────────────────────────────────────────────────
# The main REPL loop drains _pending_questions and fills _question_answers.
_pending_questions: list[dict] = []   # [{id, question, options, allow_freetext, event, result_holder}]
_ask_lock = threading.Lock()

# ── Tool JSON schemas (sent to Claude API) ─────────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "Read",
        "description": (
            "Read a file's contents. Returns content with line numbers "
            "(format: 'N\\tline'). Use limit/offset to read large files in chunks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute file path"},
                "limit":     {"type": "integer", "description": "Max lines to read"},
                "offset":    {"type": "integer", "description": "Start line (0-indexed)"},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "Write",
        "description": "Write content to a file, creating parent directories as needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content":   {"type": "string"},
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "Edit",
        "description": (
            "Replace exact text in a file. old_string must match exactly (including whitespace). "
            "If old_string appears multiple times, use replace_all=true or add more context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path":   {"type": "string"},
                "old_string":  {"type": "string", "description": "Exact text to replace"},
                "new_string":  {"type": "string", "description": "Replacement text"},
                "replace_all": {"type": "boolean", "description": "Replace all occurrences"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
    {
        "name": "Bash",
        "description": "Execute a shell command. Returns stdout+stderr. Stateless (no cd persistence).",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "description": "Seconds before timeout (default 30)"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "Glob",
        "description": "Find files matching a glob pattern. Returns sorted list of matching paths.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern e.g. **/*.py"},
                "path":    {"type": "string", "description": "Base directory (default: cwd)"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "Grep",
        "description": "Search file contents with regex using ripgrep (falls back to grep).",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern":      {"type": "string", "description": "Regex pattern"},
                "path":         {"type": "string", "description": "File or directory to search"},
                "glob":         {"type": "string", "description": "File filter e.g. *.py"},
                "output_mode":  {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": "content=matching lines, files_with_matches=file paths, count=match counts",
                },
                "case_insensitive": {"type": "boolean"},
                "context":      {"type": "integer", "description": "Lines of context around matches"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "WebFetch",
        "description": "Fetch a URL and return its text content (HTML stripped).",
        "input_schema": {
            "type": "object",
            "properties": {
                "url":    {"type": "string"},
                "prompt": {"type": "string", "description": "Hint for what to extract"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "WebSearch",
        "description": "Search the web via DuckDuckGo and return top results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    # ── Task tools (schemas also listed here for Claude's tool list) ──────────
    {
        "name": "TaskCreate",
        "description": (
            "Create a new task in the task list. "
            "Use this to track work items, to-dos, and multi-step plans."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "subject":     {"type": "string", "description": "Brief title"},
                "description": {"type": "string", "description": "What needs to be done"},
                "active_form": {"type": "string", "description": "Present-continuous label while in_progress"},
                "metadata":    {"type": "object", "description": "Arbitrary metadata"},
            },
            "required": ["subject", "description"],
        },
    },
    {
        "name": "TaskUpdate",
        "description": (
            "Update a task: change status, subject, description, owner, "
            "dependency edges, or metadata. "
            "Set status='deleted' to remove. "
            "Statuses: pending, in_progress, completed, cancelled, deleted."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id":       {"type": "string"},
                "subject":       {"type": "string"},
                "description":   {"type": "string"},
                "status":        {"type": "string", "enum": ["pending","in_progress","completed","cancelled","deleted"]},
                "active_form":   {"type": "string"},
                "owner":         {"type": "string"},
                "add_blocks":    {"type": "array", "items": {"type": "string"}},
                "add_blocked_by":{"type": "array", "items": {"type": "string"}},
                "metadata":      {"type": "object"},
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "TaskGet",
        "description": "Retrieve full details of a single task by ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Task ID to retrieve"},
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "TaskList",
        "description": "List all tasks with their status, owner, and pending blockers.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "NotebookEdit",
        "description": (
            "Edit a Jupyter notebook (.ipynb) cell. "
            "Supports replace (modify existing cell), insert (add new cell after cell_id), "
            "and delete (remove cell) operations. "
            "Read the notebook with the Read tool first to see cell IDs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "notebook_path": {
                    "type": "string",
                    "description": "Absolute path to the .ipynb notebook file",
                },
                "new_source": {
                    "type": "string",
                    "description": "New source code/text for the cell",
                },
                "cell_id": {
                    "type": "string",
                    "description": (
                        "ID of the cell to edit. For insert, the new cell is inserted after this cell "
                        "(or at the beginning if omitted). Use 'cell-N' (0-indexed) if no IDs are set."
                    ),
                },
                "cell_type": {
                    "type": "string",
                    "enum": ["code", "markdown"],
                    "description": "Cell type. Required for insert; defaults to current type for replace.",
                },
                "edit_mode": {
                    "type": "string",
                    "enum": ["replace", "insert", "delete"],
                    "description": "replace (default) / insert / delete",
                },
            },
            "required": ["notebook_path", "new_source"],
        },
    },
    {
        "name": "GetDiagnostics",
        "description": (
            "Get LSP-style diagnostics (errors, warnings, hints) for a source file. "
            "Uses pyright/mypy/flake8 for Python, tsc for TypeScript/JavaScript, "
            "and shellcheck for shell scripts. Returns structured diagnostic output."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file to diagnose",
                },
                "language": {
                    "type": "string",
                    "description": (
                        "Override auto-detected language: python, javascript, typescript, "
                        "shellscript. Omit to auto-detect from file extension."
                    ),
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "AskUserQuestion",
        "description": (
            "Pause execution and ask the user a clarifying question. "
            "Use this when you need a decision from the user before proceeding. "
            "Returns the user's answer as a string."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user.",
                },
                "options": {
                    "type": "array",
                    "description": "Optional list of choices. Each item: {label, description}.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label":       {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["label"],
                    },
                },
                "allow_freetext": {
                    "type": "boolean",
                    "description": "If true (default), user may type a free-text answer instead of selecting an option.",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "SleepTimer",
        "description": (
            "Schedule a silent background timer. When the timer finishes, it injects an automated prompt into the chat history: "
            "'(System Automated Event): The timer has finished...' so you can seamlessly wake up and execute deferred monitoring tasks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "seconds": {"type": "integer", "description": "Number of seconds to sleep before waking up."}
            },
            "required": ["seconds"],
        },
    },
]

# ── Safe bash commands (never ask permission) ───────────────────────────────

_SAFE_PREFIXES = (
    "ls", "cat", "head", "tail", "wc", "pwd", "echo", "printf", "date",
    "which", "type", "env", "printenv", "uname", "whoami", "id",
    "git log", "git status", "git diff", "git show", "git branch",
    "git remote", "git stash list", "git tag",
    "find ", "grep ", "rg ", "ag ", "fd ",
    "python ", "python3 ", "node ", "ruby ", "perl ",
    "pip show", "pip list", "npm list", "cargo metadata",
    "df ", "du ", "free ", "top -bn", "ps ",
    "curl -I", "curl --head",
)

def _is_safe_bash(cmd: str) -> bool:
    c = cmd.strip()
    return any(c.startswith(p) for p in _SAFE_PREFIXES)


# ── Diff helpers ──────────────────────────────────────────────────────────

def generate_unified_diff(old, new, filename, context_lines=3):
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines,
        fromfile=f"a/{filename}", tofile=f"b/{filename}", n=context_lines)
    return "".join(diff)

def maybe_truncate_diff(diff_text, max_lines=80):
    lines = diff_text.splitlines()
    if len(lines) <= max_lines:
        return diff_text
    shown = lines[:max_lines]
    remaining = len(lines) - max_lines
    return "\n".join(shown) + f"\n\n[... {remaining} more lines ...]"


# ── Tool implementations ───────────────────────────────────────────────────

def _read(file_path: str, limit: int = None, offset: int = None) -> str:
    p = Path(file_path)
    if not p.exists():
        return f"Error: file not found: {file_path}"
    if p.is_dir():
        return f"Error: {file_path} is a directory"
    try:
        # Explicitly use utf-8 and newline="" to avoid encoding/line-ending mismatches
        lines = p.read_text(encoding="utf-8", errors="replace", newline="").splitlines(keepends=True)
        start = offset or 0
        chunk = lines[start:start + limit] if limit else lines[start:]
        if not chunk:
            return "(empty file)"
        # Use standard 6-char padding for line numbers, matching Claude's expected format
        return "".join(f"{start + i + 1:6}\t{l}" for i, l in enumerate(chunk))
    except Exception as e:
        return f"Error: {e}"


def _write(file_path: str, content: str) -> str:
    p = Path(file_path)
    try:
        is_new = not p.exists()
        # Ensure utf-8 and newline="" for reading existing content to generate diff
        old_content = "" if is_new else p.read_text(encoding="utf-8", errors="replace", newline="")
        p.parent.mkdir(parents=True, exist_ok=True)
        # Always write as utf-8 with newline="" to prevent double CRLF on Windows
        p.write_text(content, encoding="utf-8", newline="")
        if is_new:
            lc = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            return f"Created {file_path} ({lc} lines)"
        filename = p.name
        diff = generate_unified_diff(old_content, content, filename)
        if not diff:
            return f"No changes in {file_path}"
        truncated = maybe_truncate_diff(diff)
        return f"File updated — {file_path}:\n\n{truncated}"
    except Exception as e:
        return f"Error: {e}"


def _edit(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    p = Path(file_path)
    if not p.exists():
        return f"Error: file not found: {file_path}"
    try:
        # Read with newline="" to get original line endings
        content = p.read_text(encoding="utf-8", errors="replace", newline="")
        
        # Detect original line endings: only treat as pure CRLF if every \n is part of \r\n
        crlf_count = content.count("\r\n")
        lf_count = content.count("\n")
        is_pure_crlf = crlf_count > 0 and crlf_count == lf_count

        # Normalize line endings to avoid \r\n vs \n mismatch during matching
        content_norm = content.replace("\r\n", "\n")
        old_norm = old_string.replace("\r\n", "\n")
        new_norm = new_string.replace("\r\n", "\n")

        count = content_norm.count(old_norm)
        if count == 0:
            return "Error: old_string not found in file. Please ensure EXACT match, including all exact leading spaces/indentation and trailing newlines."
        if count > 1 and not replace_all:
            return (f"Error: old_string appears {count} times. "
                    "Provide more context to make it unique, or use replace_all=true.")

        old_content_norm = content_norm
        new_content_norm = content_norm.replace(old_norm, new_norm) if replace_all else \
                           content_norm.replace(old_norm, new_norm, 1)

        # Restore CRLF only for pure-CRLF files; mixed or LF-only files stay as LF
        if is_pure_crlf:
            final_content = new_content_norm.replace("\n", "\r\n")
            old_content_final = content
        else:
            final_content = new_content_norm
            old_content_final = content_norm
                      
        # Write with newline="" to prevent double CRLF translation on Windows
        p.write_text(final_content, encoding="utf-8", newline="")
        filename = p.name
        diff = generate_unified_diff(old_content_final, final_content, filename)
        return f"Changes applied to {filename}:\n\n{diff}"
    except Exception as e:
        return f"Error: {e}"


def _bash(command: str, timeout: int = 30) -> str:
    try:
        r = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=os.getcwd(),
        )
        out = r.stdout
        if r.stderr:
            out += ("\n" if out else "") + "[stderr]\n" + r.stderr
        return out.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


def _glob(pattern: str, path: str = None) -> str:
    base = Path(path) if path else Path.cwd()
    try:
        matches = sorted(base.glob(pattern))
        if not matches:
            return "No files matched"
        return "\n".join(str(m) for m in matches[:500])
    except Exception as e:
        return f"Error: {e}"


def _has_rg() -> bool:
    try:
        subprocess.run(["rg", "--version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def _grep(pattern: str, path: str = None, glob: str = None,
          output_mode: str = "files_with_matches",
          case_insensitive: bool = False, context: int = 0) -> str:
    use_rg = _has_rg()
    cmd = ["rg" if use_rg else "grep", "--no-heading"]
    if case_insensitive:
        cmd.append("-i")
    if output_mode == "files_with_matches":
        cmd.append("-l")
    elif output_mode == "count":
        cmd.append("-c")
    else:
        cmd.append("-n")
        if context:
            cmd += ["-C", str(context)]
    if glob:
        cmd += (["--glob", glob] if use_rg else ["--include", glob])
    cmd.append(pattern)
    cmd.append(path or str(Path.cwd()))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        out = r.stdout.strip()
        return out[:20000] if out else "No matches found"
    except Exception as e:
        return f"Error: {e}"


def _webfetch(url: str, prompt: str = None) -> str:
    try:
        import httpx
        r = httpx.get(url, headers={"User-Agent": "NanoClaude/1.0"},
                      timeout=30, follow_redirects=True)
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        if "html" in ct:
            text = re.sub(r"<script[^>]*>.*?</script>", "", r.text,
                          flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text,
                          flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
        else:
            text = r.text
        return text[:25000]
    except ImportError:
        return "Error: httpx not installed — run: pip install httpx"
    except Exception as e:
        return f"Error: {e}"


def _websearch(query: str) -> str:
    try:
        import httpx
        url = "https://html.duckduckgo.com/html/"
        r = httpx.get(url, params={"q": query},
                      headers={"User-Agent": "Mozilla/5.0 (compatible)"},
                      timeout=30, follow_redirects=True)
        titles   = re.findall(r'class="result__title"[^>]*>.*?<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                               r.text, re.DOTALL)
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</div>', r.text, re.DOTALL)
        results = []
        for i, (link, title) in enumerate(titles[:8]):
            t = re.sub(r"<[^>]+>", "", title).strip()
            s = re.sub(r"<[^>]+>", "", snippets[i]).strip() if i < len(snippets) else ""
            results.append(f"**{t}**\n{link}\n{s}")
        return "\n\n".join(results) if results else "No results found"
    except ImportError:
        return "Error: httpx not installed — run: pip install httpx"
    except Exception as e:
        return f"Error: {e}"


# ── NotebookEdit implementation ────────────────────────────────────────────

def _parse_cell_id(cell_id: str) -> int | None:
    """Convert 'cell-N' shorthand to integer index; return None if not that form."""
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

    # Resolve cell index
    def _resolve_index(cid: str) -> int | None:
        # Try exact id match first
        for i, c in enumerate(cells):
            if c.get("id") == cid:
                return i
        # Fallback: cell-N
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
        # Determine nb format for cell ids
        nbformat = nb.get("nbformat", 4)
        nbformat_minor = nb.get("nbformat_minor", 0)
        use_ids = nbformat > 4 or (nbformat == 4 and nbformat_minor >= 5)
        new_id = None
        if use_ids:
            import random, string
            new_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))

        if cell_type == "markdown":
            new_cell = {"cell_type": "markdown", "source": new_source, "metadata": {}}
        else:
            new_cell = {
                "cell_type": "code",
                "source": new_source,
                "metadata": {},
                "execution_count": None,
                "outputs": [],
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
        nb["cells"] = cells
        p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        return f"Deleted cell '{cell_id}' from {notebook_path}"
    else:
        return f"Error: unknown edit_mode '{edit_mode}' — use replace, insert, or delete"

    nb["cells"] = cells
    p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return f"NotebookEdit({edit_mode}) applied to cell '{cell_id}' in {notebook_path}"


# ── GetDiagnostics implementation ──────────────────────────────────────────

def _detect_language(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    return {
        ".py":   "python",
        ".js":   "javascript",
        ".mjs":  "javascript",
        ".cjs":  "javascript",
        ".ts":   "typescript",
        ".tsx":  "typescript",
        ".sh":   "shellscript",
        ".bash": "shellscript",
        ".zsh":  "shellscript",
    }.get(ext, "unknown")


def _run_quietly(cmd: list[str], cwd: str | None = None, timeout: int = 30) -> tuple[int, str]:
    """Run a command, return (returncode, combined_output)."""
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=cwd or os.getcwd(),
        )
        out = (r.stdout + ("\n" + r.stderr if r.stderr else "")).strip()
        return r.returncode, out
    except FileNotFoundError:
        return -1, f"(command not found: {cmd[0]})"
    except subprocess.TimeoutExpired:
        return -1, f"(timed out after {timeout}s)"
    except Exception as e:
        return -1, f"(error: {e})"


def _get_diagnostics(file_path: str, language: str = None) -> str:
    p = Path(file_path)
    if not p.exists():
        return f"Error: file not found: {file_path}"

    lang = language or _detect_language(file_path)
    abs_path = str(p.resolve())
    results: list[str] = []

    if lang == "python":
        # Try pyright first (most comprehensive)
        rc, out = _run_quietly(["pyright", "--outputjson", abs_path])
        if rc != -1:
            try:
                data = json.loads(out)
                diags = data.get("generalDiagnostics", [])
                if not diags:
                    results.append("pyright: no diagnostics")
                else:
                    lines = [f"pyright ({len(diags)} issue(s)):"]
                    for d in diags[:50]:
                        rng = d.get("range", {}).get("start", {})
                        ln = rng.get("line", 0) + 1
                        ch = rng.get("character", 0) + 1
                        sev = d.get("severity", "error")
                        msg = d.get("message", "")
                        rule = d.get("rule", "")
                        lines.append(f"  {ln}:{ch} [{sev}] {msg}" + (f" ({rule})" if rule else ""))
                    results.append("\n".join(lines))
            except json.JSONDecodeError:
                if out:
                    results.append(f"pyright:\n{out[:3000]}")
        else:
            # Try mypy
            rc2, out2 = _run_quietly(["mypy", "--no-error-summary", abs_path])
            if rc2 != -1:
                results.append(f"mypy:\n{out2[:3000]}" if out2 else "mypy: no diagnostics")
            else:
                # Fall back to flake8
                rc3, out3 = _run_quietly(["flake8", abs_path])
                if rc3 != -1:
                    results.append(f"flake8:\n{out3[:3000]}" if out3 else "flake8: no diagnostics")
                else:
                    # Last resort: py_compile syntax check
                    rc4, out4 = _run_quietly(["python3", "-m", "py_compile", abs_path])
                    if out4:
                        results.append(f"py_compile (syntax check):\n{out4}")
                    else:
                        results.append("py_compile: syntax OK (no further tools available)")

    elif lang in ("javascript", "typescript"):
        # Try tsc
        rc, out = _run_quietly(["tsc", "--noEmit", "--strict", abs_path])
        if rc != -1:
            results.append(f"tsc:\n{out[:3000]}" if out else "tsc: no errors")
        else:
            # Try eslint
            rc2, out2 = _run_quietly(["eslint", abs_path])
            if rc2 != -1:
                results.append(f"eslint:\n{out2[:3000]}" if out2 else "eslint: no issues")
            else:
                results.append("No TypeScript/JavaScript checker found (install tsc or eslint)")

    elif lang == "shellscript":
        rc, out = _run_quietly(["shellcheck", abs_path])
        if rc != -1:
            results.append(f"shellcheck:\n{out[:3000]}" if out else "shellcheck: no issues")
        else:
            # Basic bash syntax check
            rc2, out2 = _run_quietly(["bash", "-n", abs_path])
            results.append(f"bash -n (syntax check):\n{out2}" if out2 else "bash -n: syntax OK")

    else:
        results.append(f"No diagnostic tool available for language: {lang or 'unknown'} (ext: {Path(file_path).suffix})")

    return "\n\n".join(results) if results else "(no diagnostics output)"


# ── AskUserQuestion implementation ────────────────────────────────────────

def _ask_user_question(
    question: str,
    options: list[dict] | None = None,
    allow_freetext: bool = True,
) -> str:
    """
    Block the agent loop and surface a question to the user in the terminal.

    The REPL loop (cheetahclaws.py) periodically calls drain_pending_questions()
    to render any questions and collect answers.  We use a threading.Event to
    block this call until the user responds.
    """
    event = threading.Event()
    result_holder: list[str] = []
    entry = {
        "question": question,
        "options": options or [],
        "allow_freetext": allow_freetext,
        "event": event,
        "result": result_holder,
    }
    with _ask_lock:
        _pending_questions.append(entry)

    # Block until the REPL answers us
    event.wait(timeout=300)  # 5-minute max wait

    if result_holder:
        return result_holder[0]
    return "(no answer — timeout)"


def ask_input_interactive(prompt: str, config: dict, context: str = "") -> str:
    """Prompt the user for input, routing to Telegram if in a Telegram turn.

    Args:
        prompt:  The input prompt shown to the user.
        config:  Agent config dict (used to detect Telegram context).
        context: Optional menu/list text to send to Telegram *before* the prompt,
                 so the remote user can see the available options.
    """
    is_tg = config.get("_in_telegram_turn", False)
    if is_tg and "_tg_send_callback" in config:
        import re, threading
        token = config.get("telegram_token")
        chat_id = config.get("telegram_chat_id")
        send = config["_tg_send_callback"]
        strip_ansi = lambda s: re.sub(r'\x1b\[[0-9;]*m', '', s).strip()

        payload = ""
        if context:
            payload += f"{strip_ansi(context)}\n\n"
        payload += f"❓ *Input Required*\n{strip_ansi(prompt)}"

        send(token, chat_id, payload)

        evt = threading.Event()
        config["_tg_input_event"] = evt
        evt.wait()

        text = config.pop("_tg_input_value", "").strip()
        config.pop("_tg_input_event", None)
        return text
    else:
        try:
            return input(prompt)
        except (KeyboardInterrupt, EOFError):
            print()
            return ""

def drain_pending_questions(config: dict) -> bool:
    """
    Called by the REPL loop after each streaming turn.
    Renders pending questions and collects user input.
    Returns True if any questions were answered.
    """
    with _ask_lock:
        pending = list(_pending_questions)
        _pending_questions.clear()

    if not pending:
        return False

    for entry in pending:
        question = entry["question"]
        options  = entry["options"]
        allow_ft = entry["allow_freetext"]
        event    = entry["event"]
        result   = entry["result"]

        print()
        print("\033[1;35m❓ Question from assistant:\033[0m")
        print(f"   {question}")

        if options:
            print()
            for i, opt in enumerate(options, 1):
                label = opt.get("label", "")
                desc  = opt.get("description", "")
                line  = f"  [{i}] {label}"
                if desc:
                    line += f" — {desc}"
                print(line)
            if allow_ft:
                print("  [0] Type a custom answer")
            print()

            while True:
                raw = ask_input_interactive("Your choice (number or text): ", config).strip()
                if not raw:
                    break

                if raw.isdigit():
                    idx = int(raw)
                    if 1 <= idx <= len(options):
                        raw = options[idx - 1]["label"]
                        break
                    elif idx == 0 and allow_ft:
                        raw = ask_input_interactive("Your answer: ", config).strip()
                        break
                    else:
                        print(f"Invalid option: {idx}")
                        raw = ""
                        continue
                elif allow_ft:
                    break  # accept free text directly
        else:
            # Free-text only
            print()
            raw = ask_input_interactive("Your answer: ", config).strip()

        result.append(raw)
        event.set()

    return True


def _sleeptimer(seconds: int, config: dict) -> str:
    import threading
    cb = config.get("_run_query_callback")
    if not cb:
        return "Error: Internal callback missing, cheetahclaws did not provide _run_query_callback"
        
    def worker():
        import time
        time.sleep(seconds)
        cb("(System Automated Event): The timer has finished. Please wake up, perform any pending monitoring checks and report to the user now.")
        
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return f"Timer successfully scheduled for {seconds} seconds. You can output your final thoughts and end your turn. You will be automatically awakened."


# ── Dispatcher (backward-compatible wrapper) ──────────────────────────────

def execute_tool(
    name: str,
    inputs: dict,
    permission_mode: str = "auto",
    ask_permission: Optional[Callable[[str], bool]] = None,
    config: dict = None,
) -> str:
    """Dispatch tool execution; ask permission for write/destructive ops.

    Permission checking is done here, then delegation goes to the registry.
    The config dict is forwarded to tool functions so they can access
    runtime context like _depth, _system_prompt, model, etc.
    """
    cfg = config or {}

    def _check(desc: str) -> bool:
        """Return True if action is allowed."""
        if permission_mode == "accept-all":
            return True
        if ask_permission:
            return ask_permission(desc)
        return True  # headless: allow everything

    # --- permission gate ---
    if name == "Write":
        if not _check(f"Write to {inputs['file_path']}"):
            return "Denied: user rejected write operation"
    elif name == "Edit":
        if not _check(f"Edit {inputs['file_path']}"):
            return "Denied: user rejected edit operation"
    elif name == "Bash":
        cmd = inputs["command"]
        if permission_mode != "accept-all" and not _is_safe_bash(cmd):
            if not _check(f"Bash: {cmd}"):
                return "Denied: user rejected bash command"
    elif name == "NotebookEdit":
        if not _check(f"Edit notebook {inputs['notebook_path']}"):
            return "Denied: user rejected notebook edit operation"

    return _registry_execute(name, inputs, cfg)


# ── Register built-in tools with the plugin registry ─────────────────────

def _register_builtins() -> None:
    """Register all built-in tools into the central registry."""
    # Use a name → schema map so ordering changes in TOOL_SCHEMAS never break this.
    _schemas = {s["name"]: s for s in TOOL_SCHEMAS}

    _tool_defs = [
        ToolDef(
            name="Read",
            schema=_schemas["Read"],
            func=lambda p, c: _read(**p),
            read_only=True,
            concurrent_safe=True,
        ),
        ToolDef(
            name="Write",
            schema=_schemas["Write"],
            func=lambda p, c: _write(**p),
            read_only=False,
            concurrent_safe=False,
        ),
        ToolDef(
            name="Edit",
            schema=_schemas["Edit"],
            func=lambda p, c: _edit(**p),
            read_only=False,
            concurrent_safe=False,
        ),
        ToolDef(
            name="Bash",
            schema=_schemas["Bash"],
            func=lambda p, c: _bash(p["command"], p.get("timeout", 30)),
            read_only=False,
            concurrent_safe=False,
        ),
        ToolDef(
            name="Glob",
            schema=_schemas["Glob"],
            func=lambda p, c: _glob(p["pattern"], p.get("path")),
            read_only=True,
            concurrent_safe=True,
        ),
        ToolDef(
            name="Grep",
            schema=_schemas["Grep"],
            func=lambda p, c: _grep(
                p["pattern"], p.get("path"), p.get("glob"),
                p.get("output_mode", "files_with_matches"),
                p.get("case_insensitive", False),
                p.get("context", 0),
            ),
            read_only=True,
            concurrent_safe=True,
        ),
        ToolDef(
            name="WebFetch",
            schema=_schemas["WebFetch"],
            func=lambda p, c: _webfetch(p["url"], p.get("prompt")),
            read_only=True,
            concurrent_safe=True,
        ),
        ToolDef(
            name="WebSearch",
            schema=_schemas["WebSearch"],
            func=lambda p, c: _websearch(p["query"]),
            read_only=True,
            concurrent_safe=True,
        ),
        ToolDef(
            name="NotebookEdit",
            schema=_schemas["NotebookEdit"],
            func=lambda p, c: _notebook_edit(
                p["notebook_path"],
                p["new_source"],
                p.get("cell_id"),
                p.get("cell_type"),
                p.get("edit_mode", "replace"),
            ),
            read_only=False,
            concurrent_safe=False,
        ),
        ToolDef(
            name="GetDiagnostics",
            schema=_schemas["GetDiagnostics"],
            func=lambda p, c: _get_diagnostics(
                p["file_path"],
                p.get("language"),
            ),
            read_only=True,
            concurrent_safe=True,
        ),
        ToolDef(
            name="AskUserQuestion",
            schema=_schemas["AskUserQuestion"],
            func=lambda p, c: _ask_user_question(
                p["question"],
                p.get("options"),
                p.get("allow_freetext", True),
            ),
            read_only=True,
            concurrent_safe=False,
        ),
        ToolDef(
            name="SleepTimer",
            schema=_schemas["SleepTimer"],
            func=lambda p, c: _sleeptimer(p["seconds"], c),
            read_only=False,
            concurrent_safe=True,
        ),
    ]
    for td in _tool_defs:
        register_tool(td)


_register_builtins()


# ── Memory tools (MemorySave, MemoryDelete, MemorySearch, MemoryList) ────────
# Defined in memory/tools.py; importing registers them automatically.
import memory.tools as _memory_tools  # noqa: F401



# ── Multi-agent tools (Agent, SendMessage, CheckAgentResult, ListAgentTasks, ListAgentTypes) ──
# Defined in multi_agent/tools.py; importing registers them automatically.
import multi_agent.tools as _multiagent_tools  # noqa: F401

# Expose get_agent_manager at module level for backward compatibility
from multi_agent.tools import get_agent_manager as _get_agent_manager  # noqa: F401


# ── Skill tools (Skill, SkillList) ────────────────────────────────────────
# Defined in skill/tools.py; importing registers them automatically.
import skill.tools as _skill_tools  # noqa: F401


# ── MCP tools ─────────────────────────────────────────────────────────────────
# mcp/tools.py connects to configured MCP servers and registers their tools.
# Connection happens in a background thread so startup is not blocked.
import mcp.tools as _mcp_tools  # noqa: F401


# ── Plugin tools ───────────────────────────────────────────────────────────────
# Load tools contributed by installed+enabled plugins.
try:
    from plugin.loader import register_plugin_tools as _reg_plugin_tools
    _reg_plugin_tools()
except Exception as _plugin_err:
    pass  # Plugin loading is best-effort; never crash startup


# ── Task tools (TaskCreate, TaskUpdate, TaskGet, TaskList) ─────────────────────
# task/tools.py registers all four tools into the central registry on import.
import task.tools as _task_tools  # noqa: F401


# ── Checkpoint hooks (backup files before Write/Edit/NotebookEdit) ───────────
from checkpoint.hooks import install_hooks as _install_checkpoint_hooks
_install_checkpoint_hooks()


# ── Plan mode tools (EnterPlanMode / ExitPlanMode) ──────────────────────────

def _enter_plan_mode(params: dict, config: dict) -> str:
    """Enter plan mode: read-only except plan file."""
    if config.get("permission_mode") == "plan":
        return "Already in plan mode. Write your plan to the plan file, then call ExitPlanMode."

    session_id = config.get("_session_id", "default")
    plans_dir = Path.cwd() / ".nano_claude" / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plans_dir / f"{session_id}.md"

    task_desc = params.get("task_description", "")
    if not plan_path.exists() or plan_path.stat().st_size == 0:
        header = f"# Plan: {task_desc}\n\n" if task_desc else "# Plan\n\n"
        plan_path.write_text(header, encoding="utf-8")

    config["_prev_permission_mode"] = config.get("permission_mode", "auto")
    config["permission_mode"] = "plan"
    config["_plan_file"] = str(plan_path)

    return (
        f"Plan mode activated. You are now in read-only mode.\n"
        f"Plan file: {plan_path}\n\n"
        f"Instructions:\n"
        f"1. Analyze the codebase using Read, Glob, Grep, WebSearch\n"
        f"2. Write your detailed implementation plan to the plan file using Write or Edit\n"
        f"3. When the plan is ready, call ExitPlanMode to request user approval\n"
        f"4. Do NOT attempt to write to any other files — they will be blocked"
    )


def _exit_plan_mode(params: dict, config: dict) -> str:
    """Exit plan mode and present plan for user approval."""
    if config.get("permission_mode") != "plan":
        return "Not in plan mode. Use EnterPlanMode first."

    plan_file = config.get("_plan_file", "")
    plan_content = ""
    if plan_file:
        p = Path(plan_file)
        if p.exists():
            plan_content = p.read_text(encoding="utf-8").strip()

    if not plan_content or plan_content == "# Plan":
        return "Plan file is empty. Write your plan to the plan file before calling ExitPlanMode."

    # Restore permissions
    prev = config.pop("_prev_permission_mode", "auto")
    config["permission_mode"] = prev

    return (
        f"Plan mode exited. Permission mode restored to: {prev}\n"
        f"Plan file: {plan_file}\n\n"
        f"The plan is ready for the user to review. "
        f"Wait for the user to approve before starting implementation.\n\n"
        f"--- Plan Content ---\n{plan_content}"
    )


_PLAN_MODE_SCHEMAS = [
    {
        "name": "EnterPlanMode",
        "description": (
            "Enter plan mode to analyze the codebase and create an implementation plan "
            "before writing code. Use this for complex, multi-file tasks. "
            "In plan mode, only the plan file is writable; all other writes are blocked."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task_description": {
                    "type": "string",
                    "description": "Brief description of the task to plan for",
                },
            },
            "required": [],
        },
    },
    {
        "name": "ExitPlanMode",
        "description": (
            "Exit plan mode and present the plan for user approval. "
            "Call this after writing your implementation plan to the plan file. "
            "The user must approve the plan before you begin implementation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

register_tool(ToolDef(
    name="EnterPlanMode",
    schema=_PLAN_MODE_SCHEMAS[0],
    func=_enter_plan_mode,
    read_only=False,
    concurrent_safe=False,
))

register_tool(ToolDef(
    name="ExitPlanMode",
    schema=_PLAN_MODE_SCHEMAS[1],
    func=_exit_plan_mode,
    read_only=False,
    concurrent_safe=False,
))
