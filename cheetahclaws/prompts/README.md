# `prompts/` — system prompt assets

This directory holds the raw Markdown used to build every system prompt
CheetahClaws sends to an LLM.  [`prompts/select.py`](select.py) loads
and assembles these files; the higher-level assembly logic (env block,
memory injection, conditional fragments) lives in
[`context.py`](../context.py).

## Design: single base + small overlays

```
final_prompt = base/default.md  +  overlays/<family>.md  (if matched)
```

Every model starts from the same `default.md` baseline.  Only when a
model has a **documented, authoritative quirk** do we append a small
overlay on top.  The previous "one full file per family" design has been
retired because it duplicated content and silently denied general
prompt-engineering guidance to families without a dedicated file.

## Layout

```
prompts/
├── __init__.py
├── select.py              # pick_base_prompt + load_fragment (lru_cache'd)
├── README.md              # this file
├── base/
│   └── default.md         # the shared baseline for every model
├── overlays/
│   ├── claude.md          # XML-tag preference (Anthropic guide)
│   ├── gemini.md          # explicit "Agentic Mode" framing (Gemini 3 guide)
│   ├── openai-reasoning.md # don't narrate CoT (o1 / o3 / o4 / gpt-5-codex)
│   └── qwen.md            # explicit "call the tool, don't ask the user" stance (Qwen function-calling guide)
└── fragments/
    ├── tmux.md            # appended when tmux is available
    └── plan.md            # appended when permission_mode == "plan"
```

## Routing — by model family, not by provider/runtime

`pick_base_prompt(provider, model_id)` returns the assembled
`default.md` + matched overlay (if any).  Overlay matching is a
case-insensitive substring check against the **last path segment** of
`model_id` (so `custom/anthropic/claude-sonnet-4-5` strips to
`claude-sonnet-4-5` and matches `claude`).

The `provider` argument is consulted only as a fallback when `model_id`
is empty.  Runtime providers (`ollama`, `lmstudio`, `custom`) are never
a prompt dimension — Qwen-3 served by DashScope, Ollama, vLLM, or
OpenRouter is the same model and gets the same prompt.  Tested by
`test_runtime_is_irrelevant_for_family_routing` and
`test_ollama_md_is_not_shipped`.

## What lives in `default.md` vs an overlay

`default.md` holds **everything that benefits every model**:

- Identity, capabilities, full tool catalog
- "Lead with the answer", "be concise", "no conversational filler"
- "Keep solutions minimal" (don't over-engineer)
- "Maximize parallel tool calls", "Read before Edit", "Glob vs Grep vs Read"
- "Trust your internal reasoning, do not narrate"
- Stop conditions, safe-vs-unsafe action list
- Multi-agent guidelines, plan-mode protocol

An **overlay** is allowed only when:

1. There is an **authoritative source** (vendor prompting guide URL) for
   the quirk.  The file must cite it in a top-of-file `<!-- Source: -->`
   comment.  Tested by `test_overlay_cites_source`.
2. The content does **not** duplicate anything already in `default.md`.
3. The overlay is **≤ 20 lines**.  Tested by `test_overlay_under_line_cap`.

Examples that meet the bar:

| Overlay | Quirk | Source |
|---|---|---|
| `claude.md` | XML tags around structured sections | Anthropic prompt-engineering guide |
| `gemini.md` | Explicit "Agentic Mode" framing + 4-step loop | Gemini 3 prompting guide |
| `openai-reasoning.md` | Don't narrate "Let me think step by step…" | OpenAI reasoning best practices |
| `qwen.md` | Override Qwen's chat-tuned "ask first" default — call the tool instead of echoing the user's path back as a question | Qwen function-calling guide |

Examples that do **not** meet the bar (would be rejected):

- "Use markdown headings" — already in default
- "Be helpful and accurate" — folklore, not vendor-documented
- "Always run tests before claiming done" — applies to every model

## 150-line cap on `default.md`

Rationale (from the [Gemini 3 prompting guide](https://ai.google.dev/gemini-api/docs/prompting-strategies)):

> "Once a system instruction becomes a 300-line constitution, you can
> no longer tell what's working and what's superstition."

CheetahClaws sets a stricter cap at 150 lines on `default.md` and 20
lines per overlay.  If `default.md` is getting long, extract long-lived
conditional content into `fragments/*.md` and append it from
`build_system_prompt()`.

## Fragments

`load_fragment(name)` reads `fragments/<name>.md`.  Fragments may
contain `{placeholder}` tokens that the caller formats at render time
(e.g. `plan.md` carries `{plan_file}`, filled in by
`context._render_plan_fragment`).  Literal `{` / `}` in a fragment must
be doubled (`{{` / `}}`).

`default.md` and overlays must NOT use placeholders — they are loaded
verbatim.  Per-run environment data (date, cwd, git info, CLAUDE.md) is
rendered separately by `context._render_env_block` and appended to the
base.

## Adding a new family overlay

1. Identify the quirk + locate the **vendor prompting guide URL** that
   documents it.  No URL = no overlay.
2. Write `overlays/<family>.md`.  Top comment must be the source link.
   Body ≤ 20 lines.  Do not repeat anything from `default.md`.
3. Add an entry to `_OVERLAY_RULES` in [`select.py`](select.py).
   Put more-specific keywords before broader ones.
4. Add a parametrized case to
   `tests/test_prompt_selection.py::test_overlay_routing` and update
   `test_overlays_directory_has_expected_files`.

## Adding a new fragment

1. Add `fragments/<name>.md`.
2. Append it conditionally in `context.build_system_prompt`.
3. Add a case to `tests/test_prompt_assembly.py`.

## What NOT to do

- **Don't read prompt files directly** from application code.  Go
  through `pick_base_prompt` / `load_fragment` so the cache stays
  coherent.
- **Don't put runtime state** (current cwd, git branch, CLAUDE.md) into
  base or overlay files.  Those live in `context._render_env_block` and
  are assembled fresh every turn.
- **Don't route by provider/runtime.**  Runtime ("ollama", "lmstudio",
  "custom", "vllm") is *how* a model is served; family ("claude",
  "qwen", "deepseek") is *what* the model is.  Prompts follow family.
- **Don't introduce a template engine** (jinja2, mustache, …).  Plain
  Markdown + `.format()` on one explicit placeholder is the design;
  anything richer belongs in a separate RFC.
- **Don't write a full per-family base file again.**  Family content
  goes in `overlays/`.  The dead-file regression test
  (`test_dead_family_base_files_are_gone`) prevents this.

## Known gaps

- **DeepSeek-R1** recommends *no* system prompt (all instructions in
  the user role).  Supporting that requires a bypass mechanism in
  `providers.py`; tracked separately.  No overlay for now.
- **Other open-source families** (Llama, Mistral, Gemma, Phi, GLM,
  MiniMax, Kimi) currently fall through to `default.md`.  Add an overlay
  when a concrete vendor-documented quirk emerges — not before.  The
  `default.md` "Investigate Before Asking" section + the runtime
  auto-nudge in `agent.py` (see `_looks_like_investigation`) are the
  baseline "be agentic" defenses for any model without an overlay.
