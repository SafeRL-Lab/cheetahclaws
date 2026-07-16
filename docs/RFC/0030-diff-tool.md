# Design Note: Diff tool — unified diff between files or text

- **Status:** Draft
- **Tracking issue:** _to be filed_
- **Author:** @shangdinggu
- **Last updated:** 2026-05-08
- **Builds on:** [`0021-tool-dispatch.md`](./0021-tool-dispatch.md), [`0024-glob-list-tools.md`](./0024-glob-list-tools.md)

A surprising omission: agents read files (Read), pattern-match
files (Glob), list dirs (List), execute (Exec, Fetch) — but
have no first-class way to ask "what's different between
these two files?" Today they have to spawn `/usr/bin/diff`
via Exec. This RFC ships a stdlib-pure built-in that does it
without giving agents an extra Exec capability.

## 1. Args

Two modes — exactly one set must be provided:

```python
# Path mode (preferred — fs-cap gated):
{ "path_a": "/abs/path/x", "path_b": "/abs/path/y",
  "context_lines": 3 }

# Text mode (no fs touch):
{ "text_a": "hello\nworld\n", "text_b": "hello\nthere\n",
  "label_a": "before", "label_b": "after",
  "context_lines": 3 }
```

- ``context_lines``: int in ``[0, 20]``, default 3 — passed
  through to ``difflib.unified_diff``.
- ``label_a`` / ``label_b``: optional strings shown in the
  diff header (defaults to the path or "a"/"b").
- Path mode reads the files via the same bounded-read code
  path as Read tool (4 MB cap).
- Mixing modes (both path_a + text_a) raises
  ``invalid_args``.

## 2. Output

```python
{
    "diff":      "<unified diff text>",
    "label_a":   "...",
    "label_b":   "...",
    "lines_a":   <int>,
    "lines_b":   <int>,
    "identical": <bool>,        # True iff both inputs equal
    "diff_lines": <int>,        # number of lines in the diff text
}
```

Empty diff (identical files) → ``diff = ""`` and
``identical = True``.

## 3. Capability

- ``requires_capability=True`` — agents need ``"Diff"`` in
  ``tool_grants``.
- Path mode: handler does its own fs check
  (``check_fs(pid, path_a, "r")`` AND
  ``check_fs(pid, path_b, "r")``) since
  ``requires_fs=(("r", "key"),)`` only supports a single
  args-key.
- Text mode: no fs check needed (no fs touch).

## 4. Output cap

Diff text is capped at **2 MB** to bound memory; if the
unified diff would exceed that, it's truncated with a
``[diff truncated at <bytes> bytes]`` line and a
``truncated: True`` flag in the result.

## 5. Backwards compatibility

- New tool — auto-registered by ``register_builtin_tools``.
- Pure stdlib (``difflib``); no new deps.
- Existing tests' expected ``register_builtin_tools`` return
  count of 5 (Echo/Read/Write/Glob/List) becomes 6.

## 6. Acceptance criteria

1. Path mode: identical files → ``identical=True``, ``diff=""``.
2. Path mode: different files → unified diff content
   includes both old + new lines.
3. Text mode: same shape, no fs check.
4. Mixing path + text args raises invalid_args.
5. ``context_lines`` outside ``[0, 20]`` raises invalid_args.
6. Path mode without "r" fs grant raises ``fs_denied``.
7. Path mode against directory raises tool_failed.
8. Diff > 2 MB → truncated flag + truncation marker.
9. ``register_builtin_tools`` includes "Diff" in its return
   list.
10. No file outside ``kernel/``, ``tests/``,
    ``docs/RFC/`` modified.
