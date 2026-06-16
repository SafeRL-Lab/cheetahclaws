# Design Note: LLM Runner MVP — first real workload on the kernel

- **Status:** Draft
- **Tracking issue:** _to be filed_
- **Author:** @shangdinggu
- **Last updated:** 2026-05-08
- **Builds on:** [`0006-resource-ledger.md`](./0006-resource-ledger.md), [`0008-agent-sandbox.md`](./0008-agent-sandbox.md), [`0016-subprocess-agent-runner.md`](./0016-subprocess-agent-runner.md), [`0017-worker-loop.md`](./0017-worker-loop.md)

This RFC ships the **first real workload** on the kernel substrate:
a subprocess-based runner that calls an LLM and emits a single
response. Everything from RFC 0003 onwards has been infrastructure
waiting to be exercised — RFC 0019 is the proof that it works.

The MVP scope is intentionally narrow:

- **Single turn.** Read init payload (model + system + user
  message), make ONE provider call, emit response, exit.
- **No tool dispatch, no streaming, no permissions.** Those are
  separate, larger pieces. This RFC validates the kernel ↔ LLM
  contract; tools / streaming / perms compose on top in follow-up
  RFCs.
- **Provider abstraction.** A `Provider` is a callable `(LlmRequest)
  -> LlmResponse`. The shipped providers are `MockProvider` (for
  tests) and `AnthropicProvider` (defensive import of the
  `anthropic` SDK; raises `ProviderUnavailable` if the package is
  absent).
- **No touching `providers.py`.** The existing `providers.py`
  module — used by the in-process REPL and `agent_runner.py` — is
  not imported. A future RFC may unify the two; this one keeps
  them independent so the kernel runner doesn't drag in any
  existing-code BC risk.

The runner ships as a parallel `__main__` entry point:

```
python -m kernel.runner.runner_main      # echo runner (existing)
python -m kernel.runner.llm               # LLM runner (this RFC)
```

WorkerLoop / RunnerSupervisor are unchanged — they don't care which
runner the supervisor spawns, only that it speaks the JSON-line
protocol from RFC 0016.

## 1. Goals & non-goals

**Goals:**

1. **Real LLM calls work end-to-end.** Spawn → handshake → provider
   call → token charge → exit. Verifiable both with mock providers
   in tests and with real Anthropic keys outside CI.
2. **Ledger integration.** Charges `tokens` (input + output combined)
   and `cost_micro` (in micro-USD) so RFC 0006 budgets actually bite.
3. **Provider portability.** A Provider is a thin protocol; new
   providers (OpenAI, Gemini, Ollama) plug in by writing one file
   that implements `__call__(LlmRequest) -> LlmResponse`.
4. **Defensive imports.** Importing `kernel.runner.llm` on a
   machine without `anthropic` installed must NOT fail.
5. **Reproducible tests without API keys.** `MockProvider` reads
   its response shape from an env var so the subprocess pipeline
   can be exercised in CI without network or secrets.

**Non-goals (this RFC):**

- **Tool use.** Out of scope. Adds permission requests, tool
  registry coupling, multi-turn loops — substantial complexity.
- **Streaming.** The chunked output story belongs in a follow-up.
  v1 buffers and returns the full response in one charge message.
- **Multi-turn conversation.** One prompt → one response → exit.
  Multi-turn is a state-machine concern handled by an
  orchestrator above the runner, not a runner concern.
- **Caching.** No prompt cache. Anthropic's prompt cache is a
  great optimisation but adds Provider-specific surface; deferred.
- **Vision / files / function tools.** Same — out of scope.
- **Cost accuracy across pricing tiers.** Cost calculation is
  per-Provider; the MVP uses Anthropic's published per-million
  rates. Cache discounts, batch discounts, etc., are
  Provider-specific refinements.
- **Replacing `providers.py`.** That module continues to serve the
  in-process REPL; no migration in this RFC.

## 2. Data model

```python
@dataclass(frozen=True)
class LlmRequest:
    model:       str               # provider-specific name
    system:      str | None        # optional system prompt
    user:        str               # the user message
    max_tokens:  int = 1024
    temperature: float = 0.7
    metadata:    dict = field(default_factory=dict)


@dataclass(frozen=True)
class LlmResponse:
    text:           str            # the model's reply
    tokens_input:   int
    tokens_output:  int
    cost_micro:     int            # micro-USD (10⁻⁶ USD)
    model:          str            # echoed from request
    finish_reason:  str            # 'stop' | 'length' | 'error' | provider-specific
    metadata:       dict = field(default_factory=dict)
```

## 3. Provider protocol

A Provider is any callable matching:

```python
def __call__(self, request: LlmRequest) -> LlmResponse: ...
```

Implementations must:

1. Be safe to call from a subprocess (no global state that wedges
   under fork).
2. Raise `ProviderUnavailable` (subclass of `RuntimeError`) for
   transient failures (network, rate limit, auth) — the runner
   maps to `exit_kind=failed`.
3. Raise `ProviderInvalidRequest` (subclass of `ValueError`) for
   malformed inputs — runner maps to `exit_kind=failed` too, but
   with a different log message.
4. Honour `max_tokens` and `temperature`.
5. Return `cost_micro` as integer micro-USD, computed from the
   provider's published rates.

## 4. JSON-line protocol (init payload shape)

The runner expects `init.payload` to look like:

```jsonc
{
  "model":       "claude-opus-4-7",
  "system":      "You are a helpful assistant.",
  "user":        "What is 2 + 2?",
  "max_tokens":  256,
  "temperature": 0.7,
  "metadata":    { /* opaque to the runner */ }
}
```

The runner emits, in order:

```jsonc
{ "op": "ready", "pid": 42 }
{ "op": "iteration_start", "iter": 1 }
{ "op": "log", "level": "info", "msg": "calling <model>" }
{ "op": "charge", "dim": "tokens", "amount": 30 }
{ "op": "charge", "dim": "cost_micro", "amount": 250 }
{ "op": "iteration_done", "iter": 1, "tokens": 30, "cost_micro": 250 }
{ "op": "exit", "exit_kind": "completed", "summary": "<truncated text>" }
```

The supervisor reads `charge` messages and applies them to the
ledger; the runner emits both `charge` AND `iteration_done` (the
latter is informational; the former drives the actual charge —
RFC 0016's auto-charge for tokens/cost from iteration_done is
disabled here to avoid double-counting).

(Implementation note: RFC 0016 §7 "Custom dims" charges via
`charge` messages but ALSO auto-charges `tokens` and `cost_micro`
from `iteration_done`. To keep this MVP correct, the runner emits
`charge` messages and a `iteration_done` with `tokens=0`,
`cost_micro=0` — purely informational. See test
`test_no_double_charge`.)

## 5. Provider selection

The runner selects a Provider via the env var
`CC_LLM_PROVIDER`:

| Value | Provider class | Required setup |
|---|---|---|
| `mock` (default in tests) | `MockProvider` | `CC_LLM_MOCK_RESPONSE_JSON` env var |
| `anthropic` | `AnthropicProvider` | `ANTHROPIC_API_KEY` env var |

The runner refuses to start if `CC_LLM_PROVIDER` is unset or
unrecognised; the supervisor sees `exit_kind=failed` with a clear
stderr tail.

`MockProvider` reads `CC_LLM_MOCK_RESPONSE_JSON` (a JSON
LlmResponse). This makes the full subprocess pipeline testable
deterministically without network or API keys.

## 6. Backwards compatibility

- New file `kernel/runner/llm/__init__.py` and submodules.
  No file outside `kernel/`, `tests/`, `docs/RFC/` is touched.
- The existing `runner_main.py` is unchanged; tests using it stay
  green.
- `anthropic` is already a project dependency (`requirements.txt`),
  but its import in `kernel.runner.llm.anthropic_provider`
  happens lazily — only on first call — so `from kernel import
  *` works on machines without the SDK.

## 7. Failure modes

| Failure | Runner behaviour | Supervisor sees |
|---|---|---|
| `CC_LLM_PROVIDER` unset | exit code 2, log error | `exit_kind=failed` (no `exit` msg) |
| `CC_LLM_PROVIDER=anthropic` but `anthropic` not installed | `ProviderUnavailable`, log error, exit 2 | `exit_kind=failed` |
| `ANTHROPIC_API_KEY` missing | `ProviderUnavailable`, log error, exit 2 | `exit_kind=failed` |
| Provider raises mid-call (rate limit, network) | log, send `exit` with `exit_kind=failed`, exit 1 | `exit_kind=failed` (clean) |
| Provider returns response | normal flow | `exit_kind=completed` |
| Init payload missing `model` or `user` | log error, exit 2 | `exit_kind=failed` |

## 8. Open questions

1. **Single-line `charge` aggregation.** The runner could emit one
   `charge` per dim (current draft) or batch them in a single
   message with multiple dims. RFC 0006 doesn't have batch charge.
   Lean: keep one charge per dim; future RFC may add `charge_many`
   to scheduler/ledger.
2. **`finish_reason` taxonomy.** The MVP uses
   `'stop' | 'length' | 'error'` plus provider-specific extras.
   Standardising fully is a job for v2.
3. **Should the runner emit `kernel.events.append` calls for each
   iteration?** Currently no — supervisor's
   `kernel.process.transitioned` is enough audit. If a future
   user wants per-iteration audit, they call append from the
   runner.

## 9. Acceptance criteria

A PR claiming this RFC must:

1. `MockProvider` constructed with a frozen response returns it
   verbatim on every call.
2. `LlmRequest` / `LlmResponse` round-trip via dataclass.
3. `python -m kernel.runner.llm` with `CC_LLM_PROVIDER=mock`
   and a fixed `CC_LLM_MOCK_RESPONSE_JSON` exits 0, sends ready,
   sends `charge` messages for tokens + cost_micro, sends `exit`
   with `completed`.
4. End-to-end via supervisor: spawn the LLM runner with a `tokens`
   ledger row, drain → ledger reflects the charged tokens.
5. End-to-end via WorkerLoop: enqueue an LLM job, worker spawns,
   ledger gets charged, scheduler entry → completed.
6. AnthropicProvider import is lazy: importing
   `kernel.runner.llm` works without anthropic SDK installed.
7. CC_LLM_PROVIDER unset → runner exits 2, supervisor sees
   crashed/failed with non-zero exit code.
8. No file outside `kernel/`, `tests/`, `docs/RFC/` modified.
