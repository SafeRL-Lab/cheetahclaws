"""tools_security.py — Path-traversal guard and bash safety check."""
from __future__ import annotations

import os
import shlex
from pathlib import Path

# ── Read-only shell vocabulary ────────────────────────────────────────────
#
# A command is auto-approved (no permission prompt) only when *every* segment
# of the pipeline is a known read-only invocation.  The classification is by
# parsed program name — not a string prefix — so `lsof` can never satisfy an
# `ls` rule and `rm` can never hide behind a leading `echo`.
#
# Three ways a segment can qualify:
#   1. the program is in _READ_ONLY_COMMANDS (plus its per-command flag guard)
#   2. the program is in _READ_ONLY_SUBCOMMANDS and its subcommand is listed
#   3. the invocation is nothing but an info flag (`--version`, `--help`)
#
# Anything else prompts.  Deliberately NOT here: interpreters (python, node,
# ruby, perl, sh), build/test runners (make, pytest, npm run), and anything
# that takes a user-supplied program to run (xargs, env VAR=…, timeout).
# Those execute arbitrary code, which is exactly what the prompt is for.

_READ_ONLY_COMMANDS = frozenset({
    # filesystem inspection
    "ls", "ll", "pwd", "stat", "file", "tree", "realpath", "readlink",
    "basename", "dirname", "du", "df", "mountpoint",
    # file contents
    "cat", "head", "tail", "nl", "wc", "od", "xxd", "strings", "base64",
    "md5sum", "sha1sum", "sha256sum", "sha512sum", "cksum",
    "diff", "cmp", "comm",
    # text processing (pipeline members)
    "sort", "uniq", "cut", "tr", "rev", "fold", "paste", "join", "column",
    "jq", "yq", "sed", "grep", "egrep", "fgrep", "rg", "ag", "fd", "find",
    # environment / system introspection
    "echo", "printf", "date", "cal", "uptime", "hostname", "uname",
    "whoami", "id", "groups", "which", "type", "command", "env", "printenv",
    "locale", "nproc", "arch", "lscpu", "lsblk", "free", "ps", "top",
    "netstat", "ss", "lsof", "ulimit", "tty",
    # archives — listing only (see _COMMAND_FLAG_GUARDS)
    "tar", "unzip", "zipinfo", "gunzip", "zcat",
    # network — header-only fetches (see _COMMAND_FLAG_GUARDS)
    "curl",
})

# program → subcommands that only read state.
_READ_ONLY_SUBCOMMANDS = {
    "git": frozenset({
        "log", "status", "diff", "show", "branch", "remote", "tag", "blame",
        "describe", "rev-parse", "rev-list", "ls-files", "ls-tree", "ls-remote",
        "cat-file", "shortlog", "whatchanged", "reflog", "count-objects",
        "grep", "annotate", "difftool", "verify-commit", "check-ignore",
        "config", "stash", "worktree", "submodule", "bisect", "notes",
    }),
    "docker":  frozenset({"ps", "images", "image", "logs", "inspect", "version",
                          "info", "top", "stats", "port", "diff", "history"}),
    "podman":  frozenset({"ps", "images", "logs", "inspect", "version", "info"}),
    "kubectl": frozenset({"get", "describe", "logs", "top", "version",
                          "explain", "api-resources", "api-versions",
                          "cluster-info"}),
    "npm":     frozenset({"ls", "list", "view", "info", "outdated", "why", "ping"}),
    "pnpm":    frozenset({"ls", "list", "outdated", "why"}),
    "yarn":    frozenset({"list", "info", "why", "outdated"}),
    "pip":     frozenset({"show", "list", "freeze", "check"}),
    "pip3":    frozenset({"show", "list", "freeze", "check"}),
    "uv":      frozenset({"tree", "version"}),
    "cargo":   frozenset({"metadata", "tree", "search", "verify-project"}),
    "go":      frozenset({"list", "env", "version", "doc"}),
    "brew":    frozenset({"list", "info", "outdated", "config", "--version"}),
    "systemctl": frozenset({"status", "list-units", "list-unit-files", "show",
                            "is-active", "is-enabled", "cat"}),
    "poetry":  frozenset({"show", "check", "env"}),
    "conda":   frozenset({"list", "info", "env"}),
}

# Subcommands that are read-only only with the right flag.
_SUBCOMMAND_FLAG_GUARDS = {
    ("git", "config"):    lambda args: any(a in ("--get", "--get-all", "--list", "-l",
                                                 "--get-regexp") for a in args),
    ("git", "stash"):     lambda args: bool(args) and args[0] in ("list", "show"),
    ("git", "worktree"):  lambda args: bool(args) and args[0] == "list",
    ("git", "submodule"): lambda args: bool(args) and args[0] in ("status", "summary"),
    ("git", "bisect"):    lambda args: bool(args) and args[0] in ("log", "view"),
    ("git", "notes"):     lambda args: bool(args) and args[0] in ("list", "show"),
    ("docker", "image"):  lambda args: bool(args) and args[0] in ("ls", "list",
                                                                  "inspect", "history"),
    ("conda", "env"):     lambda args: bool(args) and args[0] == "list",
    ("poetry", "env"):    lambda args: bool(args) and args[0] in ("info", "list"),
}

# Flags that turn an otherwise read-only command into a mutating one.
def _guard_find(args: list[str]) -> bool:
    banned = {"-delete", "-exec", "-execdir", "-ok", "-okdir",
              "-fprint", "-fprint0", "-fprintf", "-fls"}
    return not any(a in banned for a in args)


def _guard_no_flags(banned: set[str]):
    return lambda args: not any(a in banned or a.split("=", 1)[0] in banned
                                for a in args)


_COMMAND_FLAG_GUARDS = {
    "find":  _guard_find,
    "fd":    _guard_no_flags({"-x", "--exec", "-X", "--exec-batch"}),
    "sed":   _guard_no_flags({"-i", "--in-place", "-s"}),
    "sort":  _guard_no_flags({"-o", "--output"}),
    # archives: listing only
    "tar":   lambda args: any(a in ("-t", "--list") or
                              (a.startswith("-") and not a.startswith("--") and "t" in a)
                              for a in args)
             and not any(a in ("-x", "-c", "--extract", "--create", "-r", "-u")
                         or (a.startswith("-") and not a.startswith("--")
                             and any(ch in a for ch in "xcru"))
                         for a in args),
    "unzip": lambda args: any(a in ("-l", "-v", "-t") for a in args),
    # network: header-only requests, never a body fetch or an -o download
    "curl":  lambda args: any(a in ("-I", "--head") for a in args)
             and not any(a in ("-o", "--output", "-O", "--remote-name",
                               "-d", "--data", "-X", "--request", "-T",
                               "--upload-file") for a in args),
    # `top` must be in batch mode or it never exits
    "top":   lambda args: any(a.startswith("-b") for a in args),
    # `env` alone prints the environment; `env FOO=1 cmd` runs a command
    "env":   lambda args: not args,
    "command": lambda args: bool(args) and args[0] in ("-v", "-V"),
    "type":  lambda args: True,
}

# Never auto-approved, even as a pipeline member: these run whatever they are
# handed, so no argument inspection makes them safe.
_NEVER_SAFE = frozenset({
    "sudo", "doas", "su", "sh", "bash", "zsh", "fish", "dash", "ksh",
    "eval", "exec", "source", ".", "xargs", "nohup", "setsid", "timeout",
    "watch", "nice", "ionice", "strace", "ltrace", "gdb", "awk", "gawk",
    "mawk", "perl", "python", "python3", "ruby", "node", "php", "tee",
    "ssh", "scp", "rsync", "make", "cmake", "ninja", "npx", "pipx",
})

# An invocation that is nothing but one of these is always safe, whatever the
# program — `python --version` tells you a version, it does not run your code.
# Long forms only, plus `-V`: the single-letter short flags are too ambiguous
# to trust across unknown programs (`-v` is verbose for most commands, and
# `shutdown -h` halts the machine rather than printing help).
_INFO_ONLY_FLAGS = frozenset({"--version", "-V", "--help", "--usage",
                              "version", "help"})

# Structures that can smuggle a second command past segment parsing.
_UNSAFE_SUBSTRINGS = ("`", "$(", "<(", ">(", "&>", "|&", "${!")

# Shell operators that separate one command from the next.
_SEGMENT_SEPARATORS = {"|", "||", "&&", ";", "\n"}

# Operators we refuse outright: output redirection writes files, `&`
# backgrounds a process past the turn, parentheses spawn a subshell.
_REJECTED_OPERATORS = {">", ">>", ">|", "&", "(", ")"}


def _classify_segment(tokens: list[str], extra_safe: frozenset[str]) -> bool:
    """Return True if one pipeline segment is a known read-only invocation."""
    if not tokens:
        return False
    prog_raw = tokens[0]
    # `VAR=value cmd` — an assignment prefix hides the real program.
    if "=" in prog_raw and not prog_raw.startswith("-"):
        return False
    prog = os.path.basename(prog_raw)
    args = tokens[1:]

    if prog in _NEVER_SAFE and prog not in extra_safe:
        # `python --version` and friends stay safe — nothing is executed.
        return bool(args) and all(a in _INFO_ONLY_FLAGS for a in args)

    if args and all(a in _INFO_ONLY_FLAGS for a in args):
        return True

    if prog in _READ_ONLY_SUBCOMMANDS:
        # Skip global flags before the subcommand, including the ones that
        # take a value — `git -C /repo status` is still `git status`.
        _VALUE_FLAGS = {"-C", "-c", "--git-dir", "--work-tree", "--namespace",
                        "--exec-path", "--context", "-n", "--namespace"}
        sub, skip = "", False
        for a in args:
            if skip:
                skip = False
                continue
            if a.startswith("-"):
                skip = a in _VALUE_FLAGS and "=" not in a
                continue
            sub = a
            break
        if sub not in _READ_ONLY_SUBCOMMANDS[prog]:
            return False
        guard = _SUBCOMMAND_FLAG_GUARDS.get((prog, sub))
        if guard:
            rest = args[args.index(sub) + 1:]
            return bool(guard(rest))
        return True

    if prog in _READ_ONLY_COMMANDS or prog in extra_safe:
        guard = _COMMAND_FLAG_GUARDS.get(prog)
        return bool(guard(args)) if guard else True

    return False


def _is_safe_bash(cmd: str, config: dict | None = None) -> bool:
    """Return True if cmd is read-only and never needs a permission prompt.

    The command is parsed, split on pipeline/list operators, and *every*
    segment must be a known read-only invocation — so `git log | head -20`
    and `ls -la | grep test` auto-run, while `ls && rm -rf build` does not.
    Output redirection, backgrounding, subshells and command substitution are
    refused outright, since they can act (or hide a second command) after a
    safe-looking prefix.

    ``config['bash_safe_extra']`` may list additional program names to treat
    as read-only (see docs/guides/reference.md).
    """
    c = (cmd or "").strip()
    if not c:
        return False
    if any(s in c for s in _UNSAFE_SUBSTRINGS):
        return False

    extra = config.get("bash_safe_extra") if config else None
    extra_safe = frozenset(extra) if isinstance(extra, (list, tuple, set)) else frozenset()

    try:
        lexer = shlex.shlex(c, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return False   # unbalanced quotes — don't guess
    if not tokens:
        return False

    segment: list[str] = []
    for tok in tokens:
        if tok in _REJECTED_OPERATORS or tok.startswith(">"):
            return False
        if tok in _SEGMENT_SEPARATORS:
            if not _classify_segment(segment, extra_safe):
                return False
            segment = []
            continue
        if tok == "<":
            continue   # input redirection reads a file: harmless
        segment.append(tok)
    return _classify_segment(segment, extra_safe)


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
