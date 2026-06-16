# Design Note: Git tool — read-only repo inspector

- **Status:** Draft
- **Tracking issue:** _to be filed_
- **Author:** @shangdinggu
- **Last updated:** 2026-05-08
- **Builds on:** [`0023-shell-exec-tool.md`](./0023-shell-exec-tool.md), [`0030-diff-tool.md`](./0030-diff-tool.md), [`0031-ast-tool.md`](./0031-ast-tool.md)

Agents constantly ask "what changed since last release?",
"who wrote this line?", "show commit X." Today the only
path is Exec — which means the operator has to grant a much
broader capability than the agent actually needs. This RFC
ships a **read-only** Git tool that wraps a small allowlist
of subcommands behind the same RLIMIT/wall-clock sandbox
primitives Exec uses.

## 1. Args

```python
{
    "op":   "log",                  # one of the read-only ops
    "repo": "/abs/path/to/repo",    # must contain .git
    "args": ["--oneline", "-n", "5"],  # optional extra args
    "ref":  "HEAD",                 # optional ref / object
    "path": "src/foo.py",           # optional path filter
    "timeout_s": 30,                # 1..120, default 30
}
```

### Allowed ops

| op | Maps to | Notes |
|----|---------|-------|
| ``status`` | ``git status --porcelain`` | always uses --porcelain for parseable output |
| ``log`` | ``git log`` | argv allowlisted: ``--oneline`` ``--graph`` ``-n N`` ``--since=...`` ``--author=...`` |
| ``diff`` | ``git diff`` | refs / paths via ``ref`` + ``path`` |
| ``show`` | ``git show`` | requires ``ref`` (commit / blob) |
| ``branch`` | ``git branch -a --no-color`` | always read-only forms |
| ``blame`` | ``git blame`` | requires ``path`` |
| ``ls_files`` | ``git ls-files`` | optional ``path`` |
| ``rev_parse`` | ``git rev-parse`` | always ``--short HEAD`` style |

Anything else → ``invalid_args``.

### Args allowlist

Each op declares which extra ``args`` flags it accepts.
Disallowed flags raise ``invalid_args``. ``path`` is
validated as a relative path under ``repo``; absolute /
escaping paths raise ``invalid_args``. The arg ``ref`` is
validated against a tight regex
``^[A-Za-z0-9_./~^@-]{1,200}$``.

No flag with ``=`` containing newlines, NULs, or shell
metacharacters is allowed.

## 2. Capability + sandbox

- Tool capability: ``"Git"`` in tool_grants.
- ``fs_grants("r")`` on the ``repo`` path AND on
  ``/usr/bin/git`` (or whichever binary is configured).
- The git binary is found at module init via
  ``shutil.which("git")``; if none, the tool registers but
  every call fails with ``tool_failed`` ("git binary not
  available").
- Same RLIMIT + wall-clock + new_session + scrubbed env as
  Exec (RFC 0023). NOT registered by
  ``register_builtin_tools``; opt-in via
  ``register_git_tool``.

## 3. Output

```python
{
    "op":         "log",
    "exit_code":  0,
    "stdout":     "<git output>",
    "stderr":     "<git output>",
    "duration_s": 0.04,
    "timed_out":  False,
    "cmd":        ["/usr/bin/git", "-C", "/abs/path",
                    "log", "--oneline", "-n", "5"],
    "stdout_truncated": False,
    "stderr_truncated": False,
}
```

Output capped at 1 MB stdout + 256 KB stderr (stricter than
Exec because git can produce vast output).

## 4. Why not just Exec?

Exec already grants fork+exec on any binary the agent has
``"r"`` on. Git tool gives the agent **less** capability:

- A Git-only agent can't run /bin/sh, /usr/bin/curl, etc.
- The op + flag allowlist defends against bypasses like
  ``git --exec-path=/tmp/evil`` or
  ``git -c core.fsmonitor=...`` which Exec couldn't catch.
- Read-only by construction (no commit / push / fetch / GC),
  so even if the agent's logic is compromised, repo state
  doesn't drift.

## 5. Backwards compatibility

- New opt-in tool. ``register_builtin_tools`` unaffected.
- Pure stdlib + the system git binary.
- ``RFCS_IMPLEMENTED`` += 32.

## 6. Acceptance criteria

1. ``register_git_tool(registry)`` adds "Git" to the
   registry; called when git binary present, registers
   anyway when missing (handler then fails).
2. ``op`` must be in the allowlist; ``"push"``,
   ``"commit"``, ``"fetch"``, ``"clone"`` raise
   invalid_args.
3. ``ref`` rejected if it contains shell metachars, spaces,
   or newlines.
4. ``path`` rejected if absolute or contains ``..``.
5. ``args`` flags must be allowlisted per op; ``--exec-path``
   raises invalid_args.
6. ``status`` against an init'd repo returns exit_code=0 and
   parseable porcelain output.
7. ``log -n 1`` against a repo with a commit returns the
   commit's first line.
8. fs_denied raised when agent lacks "r" on repo.
9. timed_out returns True if op exceeds timeout.
10. No file outside ``kernel/``, ``tests/``,
    ``docs/RFC/`` modified.
