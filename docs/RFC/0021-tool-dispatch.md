# Design Note: Tool Dispatch + Permission Routing

- **Status:** Draft
- **Tracking issue:** _to be filed_
- **Author:** @shangdinggu
- **Last updated:** 2026-05-08
- **Builds on:** [`0001-daemon-design-note.md`](./0001-daemon-design-note.md) §2 (originator-only permission), [`0005-capability-model.md`](./0005-capability-model.md), [`0011-agent-fs.md`](./0011-agent-fs.md), [`0016-subprocess-agent-runner.md`](./0016-subprocess-agent-runner.md)

This RFC ships the **tool dispatch substrate** that lets an agent
running inside a sandboxed subprocess request a tool execution
(Read a file, Write a file, run a checked Bash command, …) and
have the supervisor:

1. Look up the requested tool in a registry.
2. Check the agent's capability against the tool name.
3. Check the agent's fs/net capability against the tool's
   resource args (e.g. `Read("/etc/passwd")` → `check_fs(pid,
   "/etc/passwd", "r")`).
4. Execute the tool.
5. Send the result back over the IPC channel.

Subprocess agents can now **act**, not just **respond**. This is
the missing piece between RFC 0019's "single-turn LLM call" and a
real agent that can read, write, and run things on the host.

The RFC keeps surface area small. Three starter tools ship
(`Read`, `Write`, `Echo`); `Bash` is **explicitly deferred** to a
follow-up because shell execution warrants its own threat-model
treatment beyond capability checks. Custom tools plug in via the
registry.

The LLM runner from RFC 0019 is **not** modified here — it doesn't
do tool calls. A separate follow-up RFC (planned 0022) will wire
the LLM runner to use this substrate to support function-calling
multi-turn conversations.

## 1. Goals & non-goals

**Goals:**

1. **One IPC message → one tool call.** Runner emits
   `{"op":"tool_call", "tool_call_id":"…", "tool":"Read",
   "args":{…}}`. Supervisor responds
   `{"op":"tool_response", "tool_call_id":"…", "ok":true,
   "result":{…}}` (or `"ok":false, "error":"…"`).
2. **Capability enforcement at the supervisor.** The runner
   doesn't decide what's allowed — the kernel does. RFC 0005
   capabilities are read inside `_handle_tool_call`; denial sends
   `tool_response.ok=false` with `"permission_denied"`.
3. **Pluggable tool registry.** A `ToolRegistry` holds `Tool`
   entries; users register custom tools at supervisor
   construction time. Built-in tools are opt-in via
   `register_builtin_tools(registry, kernel=...)`.
4. **Fs-resource gating.** Tools declare `requires_fs=[("read",
   "path"), ("write", "path")]` in their `Tool` definition; the
   supervisor pulls the path from the call args at the named
   key and runs `kernel.cap.check_fs(pid, path, mode)` before
   dispatch.
5. **Audit.** Every tool call (allowed or denied) is appended to
   the agent's event log via `kernel.events.append` with kind
   `tool.call.dispatched` or `tool.call.denied`.

**Non-goals (this RFC):**

- **Bash / shell.** Out of scope. Future RFC handles cmd-injection
  prevention, env scrubbing, etc.
- **Long-running tools.** v1 tools run synchronously inside the
  supervisor's dispatch path; a 5-second tool blocks the runner
  for 5 seconds. Streaming tool output (e.g. shell command emitting
  lines) is a follow-up.
- **Tool-call streaming back to runner.** v1 sends one
  `tool_response` per `tool_call`. Future may add chunked
  responses for large file reads.
- **Cross-agent tool calls.** A tool runs in the supervisor's
  Python process, not in another agent. Use the mailbox if you
  want agent-to-agent.
- **LLM runner integration.** That's RFC 0022.

## 2. IPC protocol

### Runner → Supervisor

```jsonc
{
  "op":            "tool_call",
  "tool_call_id":  "<unique string>",
  "tool":          "<tool name>",
  "args":          { /* tool-specific */ }
}
```

`tool_call_id` is opaque to the supervisor; the runner uses it to
match responses (in case it issues multiple calls before reading).

### Supervisor → Runner

Success:

```jsonc
{
  "op":            "tool_response",
  "tool_call_id":  "<echoed>",
  "ok":            true,
  "result":        { /* tool-specific */ }
}
```

Failure:

```jsonc
{
  "op":            "tool_response",
  "tool_call_id":  "<echoed>",
  "ok":            false,
  "error":         "<short slug>",
  "message":       "<human-readable details>"
}
```

Error slugs:

| Slug | Meaning |
|---|---|
| `tool_not_found` | name absent from registry |
| `permission_denied` | `kernel.cap.check_tool` returned False |
| `fs_denied` | `kernel.cap.check_fs` returned False for one of the requires_fs paths |
| `invalid_args` | arg validation failed in the tool handler |
| `tool_failed` | handler raised something else |

## 3. `Tool` data model

```python
@dataclass(frozen=True)
class Tool:
    name:        str           # "Read", "Write", … (matches kernel.cap tool name)
    description: str           # human-readable; useful for tool listing
    handler:     Callable[[dict, ToolContext], dict]
    requires_capability: bool = True   # check kernel.cap.check_tool
    requires_fs: tuple = ()
        # each entry: (mode, args_key)  e.g. ("r", "path") or ("rw", "path")
        # The supervisor extracts args[key], canonicalises, calls
        # kernel.cap.check_fs(pid, path, mode).
```

`ToolContext` carries kernel access for handlers that need it
(e.g. tools that touch AgentFS):

```python
@dataclass(frozen=True)
class ToolContext:
    pid:    int             # owning agent
    kernel: Kernel          # for fs / mailbox / event_log access
```

## 4. Built-in tools (v1)

### `Echo`

```python
Echo(text="hello")  →  {"text": "hello"}
```

No fs / net requirement; deliberately simple — useful for testing
the dispatch path without touching real I/O.

### `Read`

```python
Read(path="/etc/hostname")  →  {"content": "hostname-here\n",
                                 "size": 14}
```

`requires_fs = (("r", "path"),)` — supervisor checks the agent has
`fs_grants` covering the requested path with read mode.

This is **host fs read** (via `open(path).read()`), not AgentFS.
A separate `FsRead` (or just an option) covers the AgentFS
case via RFC 0011's API.

### `Write`

```python
Write(path="/tmp/x.txt", content="hi")  →  {"size": 2}
```

`requires_fs = (("rw", "path"),)`.

Host fs write.

## 5. Supervisor integration

The supervisor accepts an optional `tool_registry` parameter:

```python
RunnerSupervisor(
    kernel_store, ...,
    tool_registry: ToolRegistry | None = None,
)
```

When `tool_registry` is None: any `tool_call` from a runner is
responded to with `tool_response.ok=false, error=tool_not_found`.

When set, supervisor.wait()'s message-drain loop adds:

```python
elif op == "tool_call":
    response = self._handle_tool_call(handle, msg, tool_registry, kernel_facade)
    handle.chan.send(response)
```

`_handle_tool_call` is the dispatch path:

```
1. Lookup tool by name → ToolNotFound = error tool_not_found
2. If tool.requires_capability:
     if not kernel.cap.check_tool(pid, tool.name): error permission_denied
3. For (mode, key) in tool.requires_fs:
     path = args.get(key)
     if not kernel.cap.check_fs(pid, path, mode): error fs_denied
4. event_log.append("tool.call.dispatched", payload={tool, args, tool_call_id})
5. result = tool.handler(args, ctx)  → dict
6. return {"op":"tool_response", "ok":true, "result":result, ...}
```

Errors at step 5 (handler raise) → `tool_response.ok=false,
error=tool_failed, message=<exception details>`.

## 6. ToolRegistry

```python
class ToolRegistry:
    def __init__(self): self._tools: dict[str, Tool] = {}
    def register(self, tool: Tool) -> None
    def get(self, name) -> Tool                # raises ToolNotFound
    def list(self) -> list[str]
    def has(self, name) -> bool
    def unregister(self, name) -> None         # idempotent
```

Registration is idempotent in name — re-registering the same
name replaces the previous entry. (Useful for hot-swapping during
tests; a future RFC may make it append-only.)

## 7. Backwards compatibility

- Supervisor's `tool_registry` parameter is optional with
  default None — every existing test that constructs
  RunnerSupervisor without it still works.
- New IPC message kind `tool_call` is unrecognised by existing
  runners; they'd never emit it.
- `runner_main.py` gets a new `CC_RUNNER_BEHAVIOR=tool_call=<json>`
  behavior for testing — additive, no impact on echo / loop /
  crash etc. (default is still `echo`).
- No schema changes.

## 8. Open questions

1. **Should the supervisor's response include the dispatch
   duration?** Useful for observability. Current draft does not
   include — keeps the protocol minimal. Easy to add via a
   `metadata` field if we want it.
2. **Per-tool quotas.** Beyond per-agent ledger
   `tool_calls`, we might want per-tool counts (Read 1000 vs
   Write 10). Out of scope for v1 — orchestrator can post-process
   via the event log.
3. **Argument schemas / validation.** Currently each tool
   handler validates its own args. A formal JSON Schema per tool
   would help auto-generate LLM tool definitions — but that's a
   follow-up for the LLM-runner integration RFC.

## 9. Acceptance criteria

A PR claiming this RFC must:

1. `Tool` + `ToolRegistry` + `ToolError` hierarchy exists.
2. `register_builtin_tools(registry, kernel=...)` registers
   Read, Write, Echo.
3. End-to-end via runner_main with
   `CC_RUNNER_BEHAVIOR=tool_call=<json>`:
   - capability granted: `tool_response.ok=true` with expected
     result.
   - capability denied: `tool_response.ok=false,
     error=permission_denied`.
   - tool not found: `tool_response.ok=false,
     error=tool_not_found`.
   - fs path not granted: `tool_response.ok=false,
     error=fs_denied`.
4. Supervisor without registry: `tool_call` returns
   `tool_not_found`.
5. Tool handler raise: `tool_response.ok=false,
   error=tool_failed`, runner doesn't crash.
6. Audit events `tool.call.dispatched` / `tool.call.denied`
   land in the event log.
7. No file outside `kernel/`, `tests/`, `docs/RFC/` modified.
