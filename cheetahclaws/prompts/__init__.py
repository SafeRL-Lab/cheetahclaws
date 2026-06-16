"""System prompt assets + selection.

Design: single shared base + small family overlays.

* ``prompts/base/default.md``     — the shared baseline for every model.
* ``prompts/overlays/<family>.md`` — appended on top when the model has a
  documented, authoritative quirk (Anthropic XML tags, Gemini 3 agentic
  framing, OpenAI reasoning models' no-narration rule).
* ``prompts/fragments/<name>.md`` — conditionally appended at runtime
  (tmux available, plan mode active).

See ``prompts/README.md`` for the overlay-admission policy (must cite a
vendor source) and line-count caps (150 for base, 20 per overlay).

Selection logic is in :mod:`prompts.select`.  Callers should not read .md
files directly — always go through ``pick_base_prompt`` / ``load_fragment``.
"""
from cheetahclaws.prompts.select import pick_base_prompt, load_fragment  # noqa: F401

__all__ = ["pick_base_prompt", "load_fragment"]
