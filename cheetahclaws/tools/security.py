"""tools_security.py — Path-traversal guard and bash safety check."""
from __future__ import annotations

import os
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


# Path patterns that hold credentials or system secrets — never accessed by
# default, even when no allowed_root is configured. Set
# CHEETAHCLAWS_FS_NO_SANDBOX=1 to bypass (e.g. when intentionally auditing
# your own secrets).
_HOME = Path.home()
_SECRET_DIRS = (
    _HOME / ".aws",
    _HOME / ".gnupg",
    _HOME / ".kube",
    _HOME / ".docker",
    Path("/root"),
    Path("/etc/sudoers.d"),
)
_SECRET_FILES = (
    _HOME / ".netrc",
    _HOME / ".pgpass",
    Path("/etc/shadow"),
    Path("/etc/gshadow"),
    Path("/etc/sudoers"),
)
_SECRET_SSH_PREFIX = _HOME / ".ssh"
_SECRET_SSH_PUBLIC = {"config", "known_hosts", "known_hosts.old", "authorized_keys"}


def _is_secret_path(resolved: Path) -> bool:
    """Best-effort check: is this path a known credential / secret store?"""
    for d in _SECRET_DIRS:
        try:
            resolved.relative_to(d.resolve(strict=False))
            return True
        except ValueError:
            continue
    for f in _SECRET_FILES:
        try:
            if resolved == f.resolve(strict=False):
                return True
        except OSError:
            continue
    # ~/.ssh: deny everything except the documented public files (config,
    # known_hosts, authorized_keys). Private keys (id_*) are always denied.
    try:
        rel = resolved.relative_to(_SECRET_SSH_PREFIX.resolve(strict=False))
        return rel.name not in _SECRET_SSH_PUBLIC or rel.parent != Path(".")
    except ValueError:
        pass
    return False


def _check_path_allowed(file_path: str, config: dict) -> str | None:
    """Return an error string if file_path is disallowed, else None.

    Two layers of defense:
      1. If config["allowed_root"] / config["_worktree_cwd"] is set, the
         file_path must resolve inside that root.
      2. Independent of (1), a default credential denylist refuses paths
         like ~/.ssh/id_*, ~/.aws/credentials, /etc/shadow, etc.
         Set CHEETAHCLAWS_FS_NO_SANDBOX=1 to disable layer (2).
    """
    try:
        resolved = Path(file_path).resolve()
    except Exception as e:
        return f"Error: path validation failed: {e}"

    allowed_root = config.get("allowed_root") or config.get("_worktree_cwd")
    if allowed_root:
        try:
            root = Path(allowed_root).resolve()
            resolved.relative_to(root)
        except ValueError:
            return (
                f"Error: path '{file_path}' is outside the allowed root '{allowed_root}'. "
                "Set config['allowed_root'] to a broader directory if this is intentional."
            )
        except Exception as e:
            return f"Error: path validation failed: {e}"

    if os.environ.get("CHEETAHCLAWS_FS_NO_SANDBOX", "0") != "1":
        if _is_secret_path(resolved):
            return (
                f"Error: path '{file_path}' is on the credential denylist "
                f"(SSH keys, ~/.aws, ~/.gnupg, /etc/shadow, etc.). "
                f"Set CHEETAHCLAWS_FS_NO_SANDBOX=1 to override."
            )
    return None
