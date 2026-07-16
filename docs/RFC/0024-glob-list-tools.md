# Design Note: Glob + List built-in tools

- **Status:** Draft
- **Tracking issue:** _to be filed_
- **Author:** @shangdinggu
- **Last updated:** 2026-05-08
- **Builds on:** [`0021-tool-dispatch.md`](./0021-tool-dispatch.md), [`0005-capability-model.md`](./0005-capability-model.md)

Two new built-in tools that round out file-system reach without
adding the threat surface of Exec:

- ``Glob`` — pattern-based file matching (``"*.py"``,
  ``"src/**/*.ts"``).
- ``List`` — directory listing with metadata.

Both are read-only, capability-gated, and **auto-registered** by
``register_builtin_tools`` (unlike Exec which stays opt-in). The
threat model is the same as ``Read``: an agent can only enumerate
paths that ``fs_grants`` already cover.

## 1. Goals & non-goals

**Goals:**

1. **Glob** with full ``pathlib.Path.glob`` semantics (``*``,
   ``**``, ``?``, ``[abc]``).
2. **List** that returns entries with name, type (file/dir/
   symlink/other), size for files, mtime for both.
3. Per-tool capability gate (``"Glob"`` / ``"List"`` in
   ``tool_grants``) AND fs gate (cwd / path readable per
   ``fs_grants``).
4. **Defense-in-depth**: each Glob match is **also** filtered
   through ``cap.check_fs`` to catch sneaky symlink escapes.
5. Bounded result counts (default 1000, max 10000).

**Non-goals:**

- ``find``-style predicates (size, age, content).
- Recursive descent on ``List`` — that's what Glob is for.
- Pattern-unsafe paths (NUL, control chars) — rejected by the
  same path validation as RFC 0011.

## 2. Tool: ``Glob``

### Args

```jsonc
{
  "pattern":     "**/*.py",          // pathlib glob pattern
  "cwd":         "/home/user/proj",  // base dir, absolute
  "max_results": 1000                // optional, max 10000
}
```

### Validation

- ``pattern``: non-empty string. Reject if contains
  ``/../`` or ``..\\`` (path traversal). Reject NUL.
- ``cwd``: absolute path; ``Path(cwd).is_dir()`` must be True;
  fs_grants must cover ``cwd`` with mode ``"r"``.
- ``max_results``: 1 ≤ x ≤ 10000.

### Behaviour

```python
matches = sorted(Path(cwd).glob(pattern))[:max_results+1]
truncated = len(matches) > max_results
matches = matches[:max_results]
# Defense-in-depth: filter to only paths fs_grants covers.
filtered = [m for m in matches
            if kernel.cap.check_fs(pid, str(m), "r")]
```

### Result

```jsonc
{
  "matches":         ["/home/user/proj/src/a.py", ...],
  "count":           42,
  "truncated":       false,
  "filtered_out":    3       // matches dropped by fs_grants check
}
```

### Capability requirements

- ``tool_grants`` must include ``"Glob"``.
- ``fs_grants`` must include ``"r"`` on ``cwd``.
- Each match is independently fs-checked; results outside
  fs_grants don't appear in the response (and ``filtered_out``
  reflects the count).

## 3. Tool: ``List``

### Args

```jsonc
{
  "path":         "/home/user/proj",
  "max_entries":  1000,
  "include_hidden": false       // default false; dotfiles excluded
}
```

### Validation

- ``path``: absolute, ``is_dir()``, fs_grants ``"r"`` covered.
- ``max_entries``: 1 ≤ x ≤ 10000.

### Result

```jsonc
{
  "path":      "/home/user/proj",
  "entries":   [
    {"name": "src", "type": "dir",  "size": null,  "mtime": 1714867123.0},
    {"name": "a.py", "type": "file", "size": 1234, "mtime": 1714867123.0},
    {"name": "link", "type": "symlink", "size": null, "mtime": ...},
  ],
  "truncated": false
}
```

``type`` ∈ ``{"file", "dir", "symlink", "other"}``. ``size`` is
non-null for files only.

## 4. Backwards compatibility

- ``register_builtin_tools`` returns 5 names instead of 3
  (existing 3 + ``Glob`` + ``List``). Existing callers that
  iterate the return value see two extra entries.
- No other change to existing files.

## 5. Acceptance criteria

A PR claiming this RFC must:

1. ``register_builtin_tools(registry)`` returns
   ``["Echo", "Read", "Write", "Glob", "List"]``.
2. Glob returns matches under cwd, sorted, capped.
3. Glob filters out matches outside fs_grants (verified by
   creating a symlink target outside grants).
4. List entries have correct type for files / dirs / symlinks.
5. Path traversal in pattern (``"../../etc/*"``) → invalid_args.
6. NUL in pattern / path → invalid_args.
7. ``include_hidden=False`` (default) excludes dotfiles.
8. fs_grants denial → fs_denied.
9. Capability denial → permission_denied.
10. No file outside ``kernel/``, ``tests/``, ``docs/RFC/``
    modified.
