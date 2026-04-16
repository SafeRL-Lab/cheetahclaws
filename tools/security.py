"""tools_security.py — Path-traversal guard and bash safety check."""
from __future__ import annotations

from pathlib import Path

# Prefixes that are safe to run without a permission prompt
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


_CHAIN_OPERATORS = (";", "&&", "||", "|", "`", "$(", "\n")


def _is_safe_bash(cmd: str) -> bool:
    """Return True if cmd is read-only and never needs a permission prompt.

    Rejects commands that contain shell chaining operators (;, &&, ||, |,
    backticks, $(…)) — these could execute arbitrary code after a safe prefix.
    """
    c = cmd.strip()
    # Reject any command that chains multiple commands
    if any(op in c for op in _CHAIN_OPERATORS):
        return False
    return any(c.startswith(p) for p in _SAFE_PREFIXES)


def _check_path_allowed(file_path: str, config: dict) -> str | None:
    """Return an error string if file_path escapes the allowed root, else None.

    Only enforced when config["allowed_root"] is set (non-None, non-empty).
    For CLI usage the default is None (unrestricted). Production deployments
    should set allowed_root to the project/workspace directory.
    """
    allowed_root = config.get("allowed_root") or config.get("_worktree_cwd")
    if not allowed_root:
        return None
    try:
        resolved = Path(file_path).resolve()
        root     = Path(allowed_root).resolve()
        resolved.relative_to(root)
        return None
    except ValueError:
        return (
            f"Error: path '{file_path}' is outside the allowed root '{root}'. "
            "Set config['allowed_root'] to a broader directory if this is intentional."
        )
    except Exception as e:
        return f"Error: path validation failed: {e}"
