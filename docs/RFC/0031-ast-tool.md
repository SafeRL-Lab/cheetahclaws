# Design Note: AST tool — Python source-structure inspector

- **Status:** Draft
- **Tracking issue:** _to be filed_
- **Author:** @shangdinggu
- **Last updated:** 2026-05-08
- **Builds on:** [`0021-tool-dispatch.md`](./0021-tool-dispatch.md), [`0030-diff-tool.md`](./0030-diff-tool.md)

A second omission: agents asking "what symbols does this
file define?" today have to grep, regex, or parse Python by
hand. This RFC ships a stdlib-only AST inspector that
returns a structured list of top-level + nested definitions
with line numbers.

## 1. Args

```python
# Path mode (preferred):
{ "path": "/abs/path/foo.py" }

# Text mode:
{ "text": "def f(): pass\n", "label": "snippet.py" }
```

Mixing path + text raises invalid_args. Path mode requires
``fs_grants("r")`` on the path.

Optional:

- ``include`` — list[str], filters node kinds. Default
  ``["function", "class", "import", "import_from"]``. Allowed
  values: ``function``, ``async_function``, ``class``,
  ``import``, ``import_from``, ``assign``, ``annotation``.
- ``max_depth`` — int in ``[1, 10]``, default 4. Caps nested
  scope traversal.

## 2. Output

```python
{
    "path":   "/abs/path/foo.py",     # or "label" in text mode
    "nodes": [
        {"kind": "function", "name": "foo",
         "lineno": 1, "end_lineno": 5,
         "args": ["x", "y"], "decorators": [],
         "scope": []},
        {"kind": "class",    "name": "Bar",
         "lineno": 7, "end_lineno": 30, "bases": ["object"],
         "decorators": [],
         "scope": []},
        {"kind": "function", "name": "method",
         "lineno": 10, "end_lineno": 12,
         "args": ["self"], "decorators": [],
         "scope": ["Bar"]},
        {"kind": "import",      "names": ["os", "sys"],
         "lineno": 1},
        {"kind": "import_from", "module": "os.path",
         "names": ["join"], "lineno": 2, "level": 0},
    ],
    "syntax_error": null,    # or {"message", "lineno", "offset"}
    "line_count": 30,
}
```

A SyntaxError doesn't raise — it's returned in
``syntax_error`` so agents can still get the file's line
count + a structured failure rather than an opaque
``tool_failed`` slug.

## 3. Capability

- ``requires_capability=True`` — agents need ``"AST"`` in
  ``tool_grants``.
- Path mode: handler runs its own
  ``check_fs(pid, path, "r")``.
- Text mode: no fs check.

## 4. Limits

- File size cap **2 MB** (smaller than Diff because AST
  parsing scales worse than line-by-line diff).
- Max nodes returned: **5000** (truncated with a
  ``truncated: true`` flag).
- Only Python files (``.py``); other extensions get
  ``invalid_args``.

## 5. Backwards compatibility

- New tool — auto-registered by ``register_builtin_tools``.
- Pure stdlib (``ast``); no new deps.
- ``register_builtin_tools`` return list grows from 6
  (post-Diff) to 7.

## 6. Acceptance criteria

1. Path mode: parses a real .py file, returns nodes for
   top-level functions / classes / imports.
2. Text mode: parses inline source.
3. Mixing path + text raises invalid_args.
4. Non-.py extension → invalid_args.
5. Syntax error → ``syntax_error`` populated, ``nodes`` may
   be empty, ``line_count`` still set.
6. Method inside class has ``scope: ["ClassName"]``.
7. ``include`` filter restricts emitted node kinds.
8. ``max_depth`` caps nested traversal.
9. Path mode without "r" fs grant raises fs_denied.
10. ``register_builtin_tools`` includes "AST" in its return
    list.
11. No file outside ``kernel/``, ``tests/``,
    ``docs/RFC/`` modified.
