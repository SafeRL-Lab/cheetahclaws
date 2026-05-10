"""
commands/advanced.py — Advanced power commands for CheetahClaws.

Commands: /brainstorm, /worker, /ssj, /memory, /agents, /skills,
          /mcp, /plugin, /tasks
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Union

from ui.render import (
    clr, info, ok, warn, err,
    _start_tool_spinner, _stop_tool_spinner,
)
from tools import _is_in_tg_turn, _is_in_web_turn


# ── Brainstorm ─────────────────────────────────────────────────────────────


def _parse_bg_flag(args: str) -> tuple[bool, str]:
    """Pull `--bg` (or `--background`) out of `args`. Returns (is_bg, rest).

    When set, /brainstorm spawns a daemon thread, returns to the REPL
    immediately, and prints stage progress / completion as background
    notifications. The user can keep typing during the run; bg output
    interleaves with their input but doesn't block the REPL."""
    import re as _re_bg
    pattern = _re_bg.compile(r"(?:^|\s)--(?:bg|background)(?:\s|$)")
    m = pattern.search(args)
    if not m:
        return False, args
    return True, (args[:m.start()] + " " + args[m.end():]).strip()


# Module-level registry of in-flight background brainstorms. Keyed by a
# short id; value is a dict with status / topic / start time / output_path.
# Threads update their own entry as they progress so `/brainstorm status`
# can read the snapshot.
_BG_BRAINSTORMS: dict[str, dict] = {}
_BG_BRAINSTORMS_LOCK = threading.Lock()


def _bg_register(bg_id: str, topic: str, output_path: str) -> None:
    with _BG_BRAINSTORMS_LOCK:
        _BG_BRAINSTORMS[bg_id] = {
            "id":       bg_id,
            "topic":    topic,
            "stage":    "starting",
            "started":  time.time(),
            "output":   output_path,
            "done":     False,
            "error":    "",
        }


def _bg_set_stage(bg_id: str, stage: str) -> None:
    with _BG_BRAINSTORMS_LOCK:
        if bg_id in _BG_BRAINSTORMS:
            _BG_BRAINSTORMS[bg_id]["stage"] = stage


def _bg_complete(bg_id: str, error: str = "") -> None:
    with _BG_BRAINSTORMS_LOCK:
        if bg_id in _BG_BRAINSTORMS:
            _BG_BRAINSTORMS[bg_id]["done"] = True
            _BG_BRAINSTORMS[bg_id]["error"] = error
            _BG_BRAINSTORMS[bg_id]["stage"] = "complete" if not error else "failed"


def _bg_snapshot() -> list[dict]:
    """Return a copy of all bg brainstorms (sorted by start time desc),
    excluding entries finished >1h ago to keep the list useful."""
    cutoff = time.time() - 3600
    with _BG_BRAINSTORMS_LOCK:
        items = [v.copy() for v in _BG_BRAINSTORMS.values()
                 if not v["done"] or v["started"] >= cutoff]
    items.sort(key=lambda v: v["started"], reverse=True)
    return items


def _parse_ground_flag(args: str) -> tuple[int, str]:
    """Pull `--ground` (boolean) or `--ground=N` (top-N cap) out of `args`.

    Returns (top_n_or_0, remaining):
      - 0       — flag absent → grounding off
      - 15      — `--ground` alone → fetch top 15 results
      - N       — `--ground=N` → fetch top N (clamped to [3, 50])

    Grounding fetches a real /research brief on the topic and inlines
    the top results into the snapshot personas see. For data-hungry
    topics (stocks, current events, recent news) this is the difference
    between "personas hallucinate from training memory" and "personas
    cite real sources". Costs 10-30s and one network round-trip per
    source — see research/aggregator.py.
    """
    import re as _re_g
    # Try `--ground=N` first
    m = _re_g.search(r"--ground=(\d+)", args)
    if m:
        n = max(3, min(int(m.group(1)), 50))
        remaining = (args[:m.start()] + args[m.end():]).strip()
        return n, remaining
    # Bare `--ground`
    m = _re_g.search(r"(?:^|\s)--ground(?:\s|$)", args)
    if m:
        remaining = (args[:m.start()] + " " + args[m.end():]).strip()
        return 15, remaining
    return 0, args


def _format_grounding_brief(brief, max_chars: int = 4000) -> str:
    """Render a research Brief into a compact markdown block ready to
    inline into a persona / lead system prompt.

    Keeps top results by engagement_score, capped at max_chars total so
    a 50-result brief doesn't blow the context window. Each entry is
    `[N] (source · domain) Title — URL — snippet[:200]`. Returns empty
    string if the brief has no usable results."""
    if not brief or not brief.results:
        return ""
    sorted_results = sorted(
        brief.results,
        key=lambda r: getattr(r, "engagement_score", 0.0),
        reverse=True,
    )
    lines: list[str] = []
    char_budget = max_chars
    for i, r in enumerate(sorted_results, start=1):
        snippet = (r.snippet or "").strip().replace("\n", " ")[:200]
        domain = getattr(r, "domain", "web")
        entry = (
            f"[{i}] ({r.source} · {domain}) **{r.title}**\n"
            f"    {r.url}\n"
            f"    {snippet}"
        )
        if len(entry) + 2 > char_budget:
            break
        lines.append(entry)
        char_budget -= len(entry) + 2
    if not lines:
        return ""
    n_kept = len(lines)
    n_total = len(sorted_results)
    header = f"### GROUNDING DATA (top {n_kept} of {n_total} results from /research)"
    suffix = (
        "\n\n_When you make a claim that this data supports or "
        "contradicts, cite by `[N]`. If your claim is NOT supported by "
        "any of these results, say so explicitly — do not invent figures._"
    )
    return header + "\n\n" + "\n\n".join(lines) + suffix


def _fetch_grounding(topic: str, top_n: int, config: dict) -> str:
    """Run a /research brief on `topic` and return the formatted
    grounding markdown. Empty string on any failure (so a flaky network
    or missing API keys doesn't break the brainstorm — we just degrade
    to the no-grounding flow with a logged warning).

    Uses the existing aggregator with a 12s per-source timeout and the
    same 24h SQLite cache /research uses, so back-to-back runs on the
    same topic are basically free.
    """
    try:
        from research.aggregator import research as _research
    except Exception as e:
        warn(f"  Grounding skipped — research module unavailable: {e}")
        return ""
    try:
        brief = _research(
            topic=topic,
            limit=max(8, top_n // 2),   # per-source cap
            max_total_results=top_n,
            synthesize=False,            # we don't need the LLM synthesis here
            use_cache=True,
            source_timeout=12.0,
            config=config,
        )
    except Exception as e:
        warn(f"  Grounding fetch failed — continuing without data ({type(e).__name__}: {str(e)[:120]})")
        return ""
    formatted = _format_grounding_brief(brief)
    if not formatted:
        warn("  Grounding returned no usable results — continuing without data.")
    return formatted


def _parse_rounds_flag(args: str) -> tuple[int | None, str]:
    """Pull `--rounds N` out of `args`, return (N_or_None, remaining).

    A round = every persona speaks once. Rounds > 1 are critique/revise
    rounds — personas see the full transcript and are explicitly asked
    to engage with what others said, not repeat their initial position.
    Default (no flag) is 2 rounds (initial positions + one critique
    round) which is what makes the result feel like an actual debate
    rather than three monologues stapled together.

    Capped to [1, 6] — beyond 6 rounds the marginal value drops sharply
    and token cost grows linearly.
    """
    import re as _re_rounds
    m = _re_rounds.search(r"--rounds(?:=|\s+)(\d+)", args)
    if not m:
        return None, args
    n = max(1, min(int(m.group(1)), 6))
    return n, (args[:m.start()] + args[m.end():]).strip()


def _parse_lead_flag(args: str) -> tuple[str | None, str]:
    """Pull `--lead <model>` out of `args`, return (lead_model_or_None, remaining).

    The lead is a separate model that opens the debate (frames the agenda
    and what to AVOID), probes weak/vague claims after each persona, and
    produces the final dense synthesis. Default (no flag) reuses
    config["model"] for all three roles.
    """
    import re as _re_lead
    pattern = _re_lead.compile(r"--lead(?:=|\s+)([^\s]+)")
    m = pattern.search(args)
    if not m:
        return None, args
    return m.group(1), (args[:m.start()] + args[m.end():]).strip()


# Default ban keywords applied to every action plan, regardless of what
# the lead's opening said. These are the cheap escape hatches that show
# up in *every* failed brainstorm transcript and that the lead's own
# self-check often misses (especially on weak models). English + Chinese.
_DEFAULT_BAN_KEYWORDS = (
    # English filler
    "consult an advisor", "consult a financial advisor", "consult an expert",
    "consult experts", "do your own research", "this is not legal advice",
    "monitor regularly", "monitor the market", "monitor your portfolio",
    "stay informed", "stay up to date", "consider diversification",
    "diversify your portfolio", "diversify across",
    "evaluate periodically", "review periodically", "review quarterly",
    "rebalance the portfolio quarterly", "rebalance quarterly",
    "research X further", "research the topic", "research the company",
    # Chinese filler
    "咨询财务顾问", "咨询专家", "咨询金融顾问", "咨询专业人士",
    "定期监控", "定期复盘", "定期评估", "定期检查", "定期评价",
    "考虑多元化", "多元化投资", "分散投资", "分散风险",
    "关注市场动态", "关注公司动态", "关注新闻", "关注行业动态",
    "做好风险管理", "评估风险偏好", "结合自身风险偏好",
    "保持谨慎", "审慎评估", "自行研究",
)


def _extract_ban_keywords(opening: str) -> list[str]:
    """Pull additional ban keywords out of the lead's own opening text.

    The opening typically contains a "不接受 / will NOT accept / forbidden"
    bullet list. We extract the quoted-string contents as topic-specific
    bans on top of the default set.
    """
    if not opening:
        return list(_DEFAULT_BAN_KEYWORDS)
    import re as _re_ban
    extra: list[str] = []
    # Strings inside Chinese 「」 quotes
    extra.extend(_re_ban.findall(r"「(.+?)」", opening))
    # Strings inside Chinese 「" "」 / English ASCII quotes
    extra.extend(_re_ban.findall(r'"([^"\n]{2,40})"', opening))
    # Strings inside curly Chinese quotes "..."
    extra.extend(_re_ban.findall(r'"([^"\n]{2,40})"', opening))
    # Strings inside English single quotes
    extra.extend(_re_ban.findall(r"'([^'\n]{2,40})'", opening))
    # Dedupe + strip
    seen = set()
    out: list[str] = []
    for kw in list(_DEFAULT_BAN_KEYWORDS) + [e.strip() for e in extra if e.strip()]:
        if kw and kw.lower() not in seen:
            seen.add(kw.lower())
            out.append(kw)
    return out


def _consensus_is_ranked(synthesis_md: str) -> bool:
    """Detect whether the Consensus section in `synthesis_md` is properly
    ranked: section header found AND ≥2 lines under it start with a digit
    + period (`1.`, `2.`, …).

    Used to gate the programmatic ranking-fallback LLM call so we only
    spend the extra round-trip when the lead model actually skipped the
    rank requirement (qwen2.5 routinely ignores ordered-list instructions
    on first pass)."""
    if not synthesis_md:
        return False
    import re as _re_rk
    section_re = _re_rk.compile(
        r"##\s*(?:Ranked\s*)?Consensus[^\n]*\n(.*?)(?=^##\s|\Z)",
        _re_rk.DOTALL | _re_rk.MULTILINE | _re_rk.IGNORECASE,
    )
    m = section_re.search(synthesis_md)
    if not m:
        return False
    body = m.group(1)
    numbered_lines = _re_rk.findall(r"^\s*\d+\.\s+\S", body, _re_rk.MULTILINE)
    return len(numbered_lines) >= 2


def _ensure_consensus_is_ranked(synthesis_md: str, topic: str,
                                  lead_model: str, config: dict) -> str:
    """If the synthesis's Consensus section isn't ranked, do ONE fallback
    LLM call asking the lead to rank it. Returns the (possibly updated)
    synthesis. Failure (LLM returns empty / call errors) silently keeps
    the original — no crash, just a missed ranking."""
    if _consensus_is_ranked(synthesis_md):
        return synthesis_md
    sys = (
        "You are the LEAD MODERATOR. The synthesis you wrote did not "
        "rank the Consensus items. Add a ranking now."
    )
    user = f"""TOPIC: {topic}

EXISTING SYNTHESIS (the Consensus section needs to be re-ranked):

{synthesis_md}

Rewrite ONLY the Consensus section so that:
1. Its first line is `**Ranked by: <one-sentence metric extracted from
   the topic>**` (e.g. `**Ranked by: highest expected return over the
   next 12 months**` for a stock topic).
2. Items are NUMBERED `1.`, `2.`, `3.`, … in priority order.
3. Each item ends with `(backed by: <letters>)` listing the agent letters.
4. Each item has an indented next line `→ Why this rank: <one sentence>`.

Output the FULL updated synthesis (all four sections), with the
Consensus section renamed to `## Ranked Consensus` and the rest
unchanged. No preamble."""
    out = _llm_oneshot(lead_model, sys, user, config)
    return out if out and _consensus_is_ranked(out) else synthesis_md


def _filter_action_plan(synthesis_md: str,
                          ban_keywords: list[str]) -> tuple[str, list[str]]:
    """Programmatically remove action-plan items that match ban keywords.

    Returns (filtered_markdown, list_of_removed_items_for_logging). Doesn't
    rely on the lead actually executing its SELF-CHECK prompt — qwen2.5
    and other weak leads often ignore the instruction and ship contradicted
    plans. This is the deterministic backstop.
    """
    if not synthesis_md or not ban_keywords:
        return synthesis_md, []
    import re as _re_filt
    # Locate the "## Concrete Action Plan" section. Match until next "##" or EOF.
    section_re = _re_filt.compile(
        r"(##\s*(?:Concrete\s*)?Action\s*Plan[^\n]*\n)(.*?)(?=^##\s|\Z)",
        _re_filt.DOTALL | _re_filt.MULTILINE | _re_filt.IGNORECASE,
    )
    m = section_re.search(synthesis_md)
    if not m:
        return synthesis_md, []
    header = m.group(1)
    body = m.group(2)
    # Split body into numbered items: "1. ...", "2. ...", … (anchored at line
    # start). We keep a leading "preamble" if any non-numbered text appears
    # before the first item.
    item_split = _re_filt.split(r"^(?=\s*\d+\.\s)", body, flags=_re_filt.MULTILINE)
    preamble = item_split[0] if item_split and not _re_filt.match(r"\s*\d+\.\s", item_split[0]) else ""
    items = item_split[1:] if preamble else item_split
    if not items:
        return synthesis_md, []

    kept: list[str] = []
    removed: list[str] = []
    lower_keywords = [kw.lower() for kw in ban_keywords]
    for item in items:
        text_lower = item.lower()
        hit_kws = [kw for kw in lower_keywords if kw in text_lower]
        if hit_kws:
            # Keep the first 80 chars of the removed item for the log line.
            short = " ".join(item.split())[:120]
            removed.append(f"{short}  ⟵ matched: {hit_kws[0]!r}")
        else:
            kept.append(item)

    if not removed:
        return synthesis_md, []

    new_body_parts = [preamble] + kept if preamble else kept
    new_body = "".join(new_body_parts).rstrip() + "\n"
    if removed:
        new_body += (
            f"\n_(programmatic self-check removed {len(removed)} action(s) "
            f"that matched the ban list — see brainstorm_outputs file for the "
            f"full transcript)_\n"
        )

    new_section = header + new_body + "\n"
    return synthesis_md[:m.start()] + new_section + synthesis_md[m.end():], removed


_WEAK_LEAD_FAMILIES = (
    "qwen", "qwq", "llama-3.2", "llama-3.1-8b", "llama-3.3-8b",
    "gemma", "phi-3", "phi-4-mini", "mistral-7b", "kimi-7b",
    "minimax-text", "abab", "deepseek-coder-6", "qwen2.5-coder-7",
    "qwen2.5-7b", "qwen2-7b",
)


def _is_weak_lead_model(model_id: str) -> bool:
    """Return True if `model_id` looks like a chat-tuned small / weak
    family that historically struggles with the lead-moderator role
    (running the opening + probes + synthesis adversarially).

    Used only to print a one-time warning recommending a stronger model
    via `--lead`. Never silently overrides the user's choice."""
    if not model_id:
        return False
    tail = model_id.rsplit("/", 1)[-1].lower()
    return any(fam in tail for fam in _WEAK_LEAD_FAMILIES)


def _extract_challenge_blocks(text: str) -> list[str]:
    """Pull the body of every `### [CHALLENGE → Agent X]` block out of a
    persona's round-2+ response. Returns a list of normalised strings
    (lower-cased, whitespace-collapsed, max 500 chars each) suitable for
    similarity comparison. Empty list if no challenge blocks found."""
    import re as _re_ch
    if not text:
        return []
    # Match heading `### [CHALLENGE` (or with → arrow / minor spelling
    # variation), capture body until next `###` heading or EOF.
    pattern = _re_ch.compile(
        r"###\s*\[?CHALLENGE.*?\](.*?)(?=###|\Z)",
        _re_ch.DOTALL | _re_ch.IGNORECASE,
    )
    out: list[str] = []
    for m in pattern.finditer(text):
        body = m.group(1).strip()
        # Normalise: lowercase, collapse whitespace, cap length
        body_norm = _re_ch.sub(r"\s+", " ", body.lower())[:500]
        if body_norm:
            out.append(body_norm)
    return out


def _jaccard_similarity(a: str, b: str) -> float:
    """Token-set Jaccard overlap of two strings. Cheap, language-agnostic
    enough to catch the qwen2.5 copy-paste pattern (round 2+ personas
    cloning another agent's CHALLENGE verbatim with maybe a word changed)."""
    if not a or not b:
        return 0.0
    import re as _re_jac
    # Tokenise on word boundaries; keeps Chinese as runs of CJK chars.
    tok_a = set(_re_jac.findall(r"\w+", a))
    tok_b = set(_re_jac.findall(r"\w+", b))
    if not tok_a or not tok_b:
        return 0.0
    inter = len(tok_a & tok_b)
    union = len(tok_a | tok_b)
    return inter / union if union else 0.0


def _is_redundant_challenge(new_text: str, prior_history: list[str],
                              threshold: float = 0.7) -> tuple[bool, float]:
    """Decide whether `new_text`'s CHALLENGE blocks duplicate any prior
    one in `prior_history`. Returns (is_redundant, max_similarity).

    `threshold` of 0.7 was picked empirically against the failure case
    in `brainstorm_outputs/brainstorm_20260509_000935.md` where 8 of 10
    round-2+ challenges were verbatim clones of the first one (Jaccard
    > 0.95). 0.7 is lenient enough to allow legitimate "two agents
    independently challenge the same claim with different angles".
    """
    new_blocks = _extract_challenge_blocks(new_text)
    if not new_blocks:
        return False, 0.0
    prior_blocks: list[str] = []
    for h in prior_history:
        prior_blocks.extend(_extract_challenge_blocks(h))
    if not prior_blocks:
        return False, 0.0
    max_sim = 0.0
    for nb in new_blocks:
        for pb in prior_blocks:
            sim = _jaccard_similarity(nb, pb)
            if sim > max_sim:
                max_sim = sim
    return max_sim >= threshold, max_sim


def _llm_oneshot(model: str, system: str, user: str, config: dict, max_chunks: int = 4000) -> str:
    """Single-turn LLM call with no tools, returns concatenated text.

    Used by the lead helpers below — opening / probe / synthesis all want
    a clean prose response with no tool plumbing involved. Failures are
    silent (return ""); callers fall back to skipping the stage so a
    flaky lead model never breaks the brainstorm flow.
    """
    from providers import stream, TextChunk
    internal = config.copy()
    internal["no_tools"] = True
    chunks: list[str] = []
    try:
        for ev in stream(model, system, [{"role": "user", "content": user}], [], internal):
            if isinstance(ev, TextChunk):
                chunks.append(ev.text)
                if len(chunks) >= max_chunks:
                    break
    except Exception:
        return ""
    return "".join(chunks).strip()


def _lead_opening(topic: str, snapshot: str, lead_model: str, config: dict) -> str:
    """Lead opens the debate: defines what success looks like, what to
    REJECT (no platitudes, no 'consult an advisor', demand specifics),
    and warns the experts that round 2+ is adversarial cross-examination
    — politeness is forbidden.

    Empty string on failure — the caller continues without an opening,
    which degrades to the previous behavior, not a crash."""
    sys = (
        "You are the LEAD MODERATOR of an expert debate. You are not one "
        "of the experts — you set the agenda and you enforce quality. "
        "You will be judged on whether the final debate is ACTIONABLE "
        "and whether the experts ACTUALLY challenged each other instead "
        "of taking turns being polite."
    )
    has_grounding = "### GROUNDING DATA" in (snapshot or "")
    grounding_note = (
        "\n\n**Real /research data is attached in the context above as a "
        "`### GROUNDING DATA` block.** Anchor the debate to what the data "
        "ACTUALLY shows — when you set the agenda, point experts at the "
        "specific results numbered `[1]`, `[2]`, …, and forbid any claim "
        "that contradicts the grounding data without citing it.\n"
        if has_grounding else ""
    )
    user = f"""TOPIC: {topic}

PROJECT CONTEXT (truncated):
{snapshot[:3500]}
{grounding_note}
Your job NOW (the opening): write a tight 8-12 line briefing that the
debate will be anchored to. The briefing MUST contain, in this order:

1. What concrete artifact would make this debate USEFUL — name the unit
   of an answer the user actually needs (e.g. "specific tickers with a
   thesis, not 'consider semiconductors'"; "concrete config keys with
   defaults, not 'add observability'"; "named refactors with file paths,
   not 'improve modularity'"). Be very specific to THIS topic.
2. The 2-3 cheap escape hatches we will NOT accept. Examples by category:
   - Generic disclaimers ("consult a financial advisor", "do your own
     research", "this is not legal advice").
   - "Consider diversification / consult experts / monitor regularly"
     style filler.
   - Restating the question as the answer.
3. The single hardest question the experts must answer to make their
   contribution worth reading.
4. **Cross-examination rule for round 2+**: in any round after the first,
   each expert MUST quote a specific claim from another expert (by letter)
   and either attack it with a counter-claim OR explicitly accept it.
   Polite agreement counts as a dodge. The lead will probe any expert
   who fails to engage adversarially.

Output ONLY the briefing as plain Markdown. No preamble. No "here is
your briefing:". Start with `### Lead Opening — Debate Anchor`."""
    return _llm_oneshot(lead_model, sys, user, config)


def _lead_probe(topic: str, persona_role: str, persona_letter: str,
                persona_text: str, lead_model: str, config: dict,
                round_num: int = 1) -> str:
    """Lead reads the latest persona's contribution and decides if it's
    too vague / dodging / non-adversarial. Returns a short pointed
    follow-up question if so, or empty string if the persona was
    concrete and (in round 2+) actually challenged someone.

    Round 1: concrete vs vague check.
    Round 2+: also requires explicit `[CHALLENGE → Agent X]`-style attack
    on a specific named claim from another agent. A polite "I agree and
    would add" reply in round 2+ counts as DODGING and earns a probe."""
    if round_num >= 2:
        sys = (
            "You are the LEAD MODERATOR of an ADVERSARIAL debate. Your job "
            "in this cross-examination round is to make sure no expert dodges "
            "by being polite. A round-2+ contribution that restates the "
            "agent's own view, agrees with others, or summarizes the debate "
            "is a DODGE — even if it's well-written and concrete. Real "
            "engagement requires quoting another agent's specific claim and "
            "attacking it."
        )
        user = f"""TOPIC: {topic}

LATEST CONTRIBUTION (Agent {persona_letter} — {persona_role}, in a
ROUND-2+ CROSS-EXAMINATION round):
{persona_text[:2500]}

Decide which case applies:

A. The contribution contains at least one CHALLENGE block where the
   agent quotes a specific claim from another agent (by letter) and
   attacks it with a counter-claim. → respond with:
       NO_PROBE

B. The contribution is a polite agreement, a synthesis, a defense-only
   reply, restates the agent's own round-1 ideas, or doesn't quote anyone
   else. → respond with one short question (≤30 words) that names a
   specific agent and claim they should challenge:
       `> Lead to Agent {persona_letter}: Agent X said "...". Attack it
       or accept it — your call, but commit. Quote and refute, don't dodge.`

Do NOT explain your decision. Output exactly one of the two forms."""
    else:
        sys = (
            "You are the LEAD MODERATOR. Your job is to keep experts honest. "
            "If the latest contribution is concrete and answers the anchor, "
            "you stay silent. If it's vague, generic, or dodges the question, "
            "you ask ONE pointed follow-up that demands a specific commitment."
        )
        user = f"""TOPIC: {topic}

LATEST CONTRIBUTION (Agent {persona_letter} — {persona_role}):
{persona_text[:2500]}

Decide: is this concrete and useful, or is it filler?

If CONCRETE (names specific things, takes a position, gives numbers /
file paths / tickers / config keys / commands), respond with the single
literal token:
    NO_PROBE

If VAGUE (uses 'consider', 'evaluate', 'should', 'monitor regularly',
'consult experts', 'diversify' without naming what, etc.), respond
with one short question (≤25 words) that demands a specific commitment.
Format: `> Lead to Agent {persona_letter}: <question>`

Do NOT explain your decision. Output exactly one of the two forms."""
    out = _llm_oneshot(lead_model, sys, user, config).strip()
    if not out or out.upper().startswith("NO_PROBE"):
        return ""
    # Trim accidental wrapping fences / prefixes
    if out.startswith("```"):
        out = out.strip("`").strip()
    return out


def _lead_synthesis(topic: str, transcript: str, lead_model: str,
                    config: dict, opening: str = "",
                    grounding: str = "") -> str:
    """Lead produces the final structured synthesis. NO tool calls
    needed — the entire transcript is in the prompt context. This is
    what replaces the old "main agent reads file then synthesizes"
    flow that caused the duplicate-Read bug.

    `opening` is the lead's own debate-opening text (the agenda + ban
    list it set at the start). When provided, the synthesis prompt
    explicitly forces the lead to check its own action plan against
    its own ban list — closing the failure case where the synthesis
    listed "consult an advisor" as filler in the same document where
    its action plan included "discuss with a financial advisor"."""
    sys = (
        "You are the LEAD MODERATOR producing the final synthesis. The "
        "user will read THIS document and act on it. Filler is malpractice. "
        "You set the ban list yourself in the opening — DO NOT contradict "
        "yourself by recommending what you forbid."
    )
    opening_block = (
        f"YOUR OWN OPENING (the agenda + ban list YOU set at the start; "
        f"every action you propose below must obey this):\n\n{opening}\n\n---\n\n"
        if opening else ""
    )
    grounding_block = (
        f"GROUNDING DATA (real /research results — every consensus claim "
        f"and every action you write must trace to either one of these "
        f"`[N]` results OR to specific persona claims in the transcript "
        f"below; if a claim has no traceable source, DROP it):\n\n"
        f"{grounding}\n\n---\n\n"
        if grounding else ""
    )
    user = f"""TOPIC: {topic}

{opening_block}{grounding_block}FULL DEBATE TRANSCRIPT (each section
is one expert's contribution, in the order they spoke):

{transcript}

Produce the synthesis as Markdown with EXACTLY these four sections,
in this order:

## Ranked Consensus
A NUMBERED list of claims that ≥2 experts backed, **ranked from most
important / highest-priority to least** by whatever metric is implicit
in the user's topic (e.g. "highest expected return" for stocks; "lowest
risk × highest impact" for refactor proposals; "easiest to ship × most
user-visible" for feature ideas). The first line of this section MUST
state the ranking metric you used in **bold**, like:

    **Ranked by: <one-sentence metric extracted from the user's topic>**

Then for each numbered item:
  - Lead with the rank number (`1.`, `2.`, …) — this is mandatory.
  - State the claim, SPECIFIC (name what / how much / which file / ticker).
  - End with `(backed by: A, C)` listing the agent letters.
  - On a new indented line: `→ Why this rank: <one-sentence justification>`.

If experts only agreed at the abstract level ("we should diversify"),
do NOT list that — drop it instead.{
    " Where a consensus claim is supported by the GROUNDING DATA, also "
    "cite the relevant `[N]` after the agent letters."
    if grounding else ""
}

## Dissents
Bullet list of claims where experts disagreed, each phrased as
`X says A, Y says B — bottom line: <your call as moderator>`. If no
real disagreement, write a single line: `No substantive dissents.`

## Concrete Action Plan
A NUMBERED list of 5-10 actions the user can take TOMORROW. Each
action must have:
  - A specific noun (a ticker / file path / config key / command /
    person to call) — never "research X" without naming the next
    output of that research.
  - An owner if applicable (you, the user, an advisor).
  - A binary done/not-done acceptance criterion.

**SELF-CHECK BEFORE WRITING THIS SECTION**: re-read the ban list in
your opening above. If any action you're about to write matches a
banned escape hatch (e.g. "consult an advisor", "diversify",
"monitor regularly", "research X" without naming what), REWRITE the
action to be specific — or DELETE it. The contradiction of banning
something then recommending it is unacceptable.

## What Was Filler
1-3 bullets calling out the cheap escape hatches the experts tried
(if any). Be blunt. If none, omit the section entirely.

Output ONLY the four sections. No preamble. No "here is the synthesis"."""
    return _llm_oneshot(lead_model, sys, user, config)


def _parse_models_flag(args: str) -> tuple[list[str], str]:
    """Pull `--models a,b,c` out of `args`, return (models, remaining_args).

    Supports both `--models a,b,c` and `--models=a,b,c`. Models are not
    validated here — providers.detect_provider does that lazily on first
    use. The remaining args (with the flag stripped) become the topic.

    Why this matters: a single-model brainstorm is an echo chamber — every
    persona shares the same training data and blind spots. Letting each
    persona run a different model (Claude critic + GPT optimist + DeepSeek
    pragmatist) buys real epistemic diversity. Borrowed in spirit from
    Dulus's RoundtableAgent (webchat_server.py).
    """
    import re as _re_models
    out_models: list[str] = []
    pattern = _re_models.compile(r"--models(?:=|\s+)([^\s]+)")
    m = pattern.search(args)
    if not m:
        return [], args
    raw = m.group(1)
    out_models = [tok.strip() for tok in raw.split(",") if tok.strip()]
    remaining = (args[:m.start()] + args[m.end():]).strip()
    return out_models, remaining


_TECH_PERSONAS = {
    "architect":   {"icon": "🏗️", "role": "Principal Software Architect",       "desc": "Focus on modularity, clear boundaries, patterns, and long-term maintainability."},
    "innovator":   {"icon": "💡", "role": "Pragmatic Product Innovator",          "desc": "Focus on bold, technically feasible ideas that add high user value and differentiation."},
    "security":    {"icon": "🛡️", "role": "Security & Risk Engineer",            "desc": "Focus on vulnerabilities, data integrity, secrets handling, and project robustness."},
    "refactor":    {"icon": "🔧", "role": "Senior Code Quality Lead",             "desc": "Focus on code smells, complexity reduction, DRY principles, and readability."},
    "performance": {"icon": "⚡", "role": "Performance & Optimization Specialist","desc": "Focus on I/O bottlenecks, resource efficiency, latency, and scalability."},
}


def _generate_personas(topic: str, curr_model: str, config: dict, count: int = 5) -> dict | None:
    from providers import stream, TextChunk
    import json

    example_entries = "\n".join(
        f'  "p{i+1}": {{"icon": "emoji", "role": "Expert Title", "desc": "One sentence describing their analytical angle."}}'
        for i in range(count)
    )
    user_msg = f"""Generate {count} expert personas for a multi-perspective brainstorming debate on: "{topic}"

Return ONLY a valid JSON object — no markdown fences, no extra text — like this:
{{
{example_entries}
}}

Choose experts whose domains are most relevant to analyzing "{topic}" from different angles."""

    internal_config = config.copy()
    internal_config["no_tools"] = True
    chunks = []
    try:
        for event in stream(curr_model, "You are a debate facilitator. Return only valid JSON.", [{"role": "user", "content": user_msg}], [], internal_config):
            if isinstance(event, TextChunk):
                chunks.append(event.text)
    except Exception:
        return None

    raw = "".join(chunks).strip()
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip().lstrip("json").strip()
            try:
                return json.loads(part)
            except Exception:
                continue
    try:
        return json.loads(raw)
    except Exception:
        return None


def cmd_brainstorm(args: str, state, config) -> bool:
    """Run a multi-persona iterative brainstorming session on the project."""
    from providers import stream, TextChunk
    from tools import ask_input_interactive

    # ── /brainstorm status — list active background brainstorms and exit. ─
    if args.strip().lower() == "status":
        snap = _bg_snapshot()
        if not snap:
            info("  No background brainstorms (none running, none finished in the last hour).")
            return True
        ok(f"Background brainstorms ({len(snap)}):")
        for s in snap:
            elapsed = int(time.time() - s["started"])
            mins = elapsed // 60
            secs = elapsed % 60
            status_color = "ok" if not s["error"] else "err" if s["done"] else "yellow"
            status_label = (
                "✓ done" if s["done"] and not s["error"]
                else "✗ failed" if s["error"]
                else f"⟳ {s['stage']}"
            )
            info(
                f"  {clr(s['id'], 'cyan')}  {clr(status_label, status_color)}  "
                f"({mins}m{secs:02d}s)  →  {s['output']}"
            )
            info(f"    topic: {s['topic'][:80]}")
            if s["error"]:
                info(clr(f"    error: {s['error'][:200]}", "dim"))
        return True

    # ── Parse the `--bg` background flag FIRST so it can branch the whole
    #    function. When set: interactive prompts still run synchronously
    #    (they need stdin), but the actual brainstorm work runs in a
    #    daemon thread so the REPL is freed.
    bg, args = _parse_bg_flag(args)

    readme_path = Path("README.md")
    readme_content = readme_path.read_text("utf-8", errors="replace") if readme_path.exists() else ""
    claude_md = Path("CLAUDE.md")
    claude_content = claude_md.read_text("utf-8", errors="replace") if claude_md.exists() else ""
    project_files = "\n".join([f.name for f in Path(".").glob("*") if f.is_file() and not f.name.startswith(".")])

    # Pull optional flags before treating the remainder as topic.
    # `--models a,b,c` distributes models round-robin across personas.
    # `--lead <model>` picks who runs the moderator role (opening, probes,
    # synthesis). `--rounds N` controls how many times each persona speaks.
    # `--ground` (or `--ground=N`) pre-fetches a /research brief and
    # inlines top results so personas debate against real data instead
    # of training-time priors.
    rounds_override, args_remaining = _parse_rounds_flag(args)
    lead_model_override, args_remaining = _parse_lead_flag(args_remaining)
    persona_models, args_remaining = _parse_models_flag(args_remaining)
    ground_top_n, args_remaining = _parse_ground_flag(args_remaining)
    user_topic = args_remaining.strip() or "general project improvement and architectural evolution"

    if config.get("_bg_recursion"):
        # Re-entered from the bg thread — values are pre-resolved by the
        # parent invocation. No prompts (the user has the REPL stdin now).
        agent_count = config["_bg_agent_count"]
        n_rounds = config["_bg_n_rounds"]
        ground_top_n = config["_bg_ground_top_n"]
    elif _is_in_tg_turn(config) or _is_in_web_turn(config):
        # No interactive prompts in bridge / web mode — pick safe defaults.
        agent_count = 5
        n_rounds = rounds_override if rounds_override is not None else 2
        # Grounding stays at whatever --ground arg said (default off).
    else:
        try:
            ans = ask_input_interactive(clr("  How many agents? (2-100, default 5) > ", "cyan"), config).strip()
            agent_count = int(ans) if ans else 5
            agent_count = max(2, min(agent_count, 100))
        except (ValueError, KeyboardInterrupt, EOFError):
            agent_count = 5
        # Rounds prompt — only when --rounds was NOT passed via args.
        # 1 = monologues (one shot per persona), 2 = initial + critique
        # (recommended default), 3+ = converges harder but costs more.
        if rounds_override is not None:
            n_rounds = rounds_override
        else:
            try:
                rans = ask_input_interactive(
                    clr("  Rounds [1=monologues, 2=critique (default), 3-6=more debate] > ", "cyan"),
                    config,
                ).strip()
                n_rounds = int(rans) if rans else 2
                n_rounds = max(1, min(n_rounds, 6))
            except (ValueError, KeyboardInterrupt, EOFError):
                n_rounds = 2
        # Grounding prompt — only when --ground was NOT passed via args.
        # Default off because some topics don't need real-data grounding
        # (architecture, refactor, design) and the fetch costs 10-30s.
        if ground_top_n == 0:
            try:
                gans = ask_input_interactive(
                    clr(
                        "  Ground in /research data first? (recommended for "
                        "stocks/news/current events) [y/N] > ",
                        "cyan",
                    ),
                    config,
                ).strip().lower()
                if gans in ("y", "yes", "1", "true"):
                    ground_top_n = 15
            except (KeyboardInterrupt, EOFError):
                pass

    # ── Background fork: when --bg was passed and we're not already
    # inside a bg recursion, spawn a daemon thread that re-enters
    # cmd_brainstorm with the same resolved args (interactive prompts
    # already done) plus markers so it (1) skips re-prompting and
    # (2) doesn't return the TODO sentinel (no REPL is listening).
    # Stage progress prints from the thread interleave with the user's
    # input — that's the trade-off for keeping the REPL free.
    outputs_dir = Path("brainstorm_outputs")
    outputs_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = outputs_dir / f"brainstorm_{ts}.md"
    out_file_abs = out_file.resolve()

    if bg and not config.get("_bg_recursion"):
        bg_id = f"bs-{ts}"
        _bg_register(bg_id, user_topic, str(out_file_abs))

        ok(f"Brainstorm started in background — id={clr(bg_id, 'cyan')}")
        info(f"  Output: {clr(str(out_file_abs), 'bold')}")
        info(clr("  Stage progress will print as it happens. The REPL is yours.", "dim"))
        info(clr("  Check status: /brainstorm status", "dim"))
        info(clr("  Tip: `tail -f` the output file for live transcript building.", "dim"))

        # Re-enter cmd_brainstorm in a daemon thread, bypassing the
        # interactive prompts via the _bg_recursion markers below.
        bg_config = {
            **config,
            "_bg_recursion":      True,
            "_bg_id":             bg_id,
            "_bg_agent_count":    agent_count,
            "_bg_n_rounds":       n_rounds,
            "_bg_ground_top_n":   ground_top_n,
            "_bg_out_file":       str(out_file),
            "_bg_out_file_abs":   str(out_file_abs),
        }
        bg_args = user_topic   # topic only — flags already extracted

        def _bg_runner():
            try:
                cmd_brainstorm(bg_args, state, bg_config)
                _bg_complete(bg_id)
                ok(f"\n[brainstorm bg id={bg_id}] complete → {out_file_abs}")
            except Exception as e:
                _bg_complete(bg_id, error=f"{type(e).__name__}: {e}")
                err(f"\n[brainstorm bg id={bg_id}] FAILED: {type(e).__name__}: {str(e)[:200]}")

        threading.Thread(target=_bg_runner, daemon=True,
                          name=f"brainstorm-{bg_id}").start()
        return True

    # If we ARE inside a bg recursion, swap in the pre-resolved values
    # and use the parent's output file path so the bg ID's announced
    # path matches what gets written.
    if config.get("_bg_recursion"):
        out_file = Path(config["_bg_out_file"])
        out_file_abs = Path(config["_bg_out_file_abs"])
        bg_id = config.get("_bg_id", "")
        if bg_id:
            _bg_set_stage(bg_id, "starting")

    # ── Synchronous path (no --bg, OR inside the bg thread). Continue
    # inline below.
    # Optional grounding fetch — must happen BEFORE snapshot is built so
    # personas / lead all see the same grounding data inline. Cheap when
    # cached (24h SQLite) so re-runs on the same topic are basically free.
    grounding_block = ""
    if ground_top_n > 0:
        info(clr(f"Fetching grounding data via /research (top {ground_top_n})...", "dim"))
        _start_tool_spinner()
        grounding_block = _fetch_grounding(user_topic, ground_top_n, config)
        _stop_tool_spinner()
        if grounding_block:
            print(clr(f"  └─ Grounding attached ({len(grounding_block)} chars).", "dim"))

    snapshot = f"""PROJECT CONTEXT:
README:
{readme_content[:3000]}

CLAUDE.MD:
{claude_content[:1000]}

ROOT FILES:
{project_files}

USER FOCUS: {user_topic}
"""
    if grounding_block:
        snapshot = grounding_block + "\n\n---\n\n" + snapshot
    curr_model = config["model"]

    info(clr(f"Generating {agent_count} topic-appropriate expert personas...", "dim"))
    personas = _generate_personas(user_topic, curr_model, config, count=agent_count)
    if not personas:
        info(clr("(persona generation failed, using default tech personas)", "dim"))
        personas = dict(list(_TECH_PERSONAS.items())[:agent_count])

    def _make_identity(letter: str) -> tuple[str, str]:
        """Build (letter, name) for one persona. Faker is preferred but falls
        back to a small hand-picked pool if the package is missing."""
        try:
            from faker import Faker
            return letter, Faker().name()
        except Exception:
            import random
            first = ["Alex", "Sam", "Taylor", "Jordan", "Casey", "Riley", "Drew", "Avery"]
            last = ["Garcia", "Martinez", "Lopez", "Hernandez", "Gonzalez", "Sanchez", "Ramirez", "Torres"]
            return letter, f"{random.choice(first)} {random.choice(last)}"

    # Pre-assign a stable (letter, name) for each persona ONCE, by the
    # persona's index in `personas`. Two prior bugs this kills:
    #   (1) `persona_name[0].upper()` for letter — every persona key was
    #       `p1/p2/…` so every Agent ended up labeled `P`, breaking the
    #       cross-examination's clarity (`Agent P quoting Agent P attacking
    #       Agent P`). Letters are now A, B, C, … (capped at Z if you ever
    #       need >26 personas, which is extreme).
    #   (2) `get_identity` was re-rolled every persona invocation, so the
    #       SAME persona's "name" changed across rounds (round 1 = Riley
    #       Torres, round 2 defense = Alex Lopez, round 3 = Taylor Gonzalez
    #       — all the same agent). That made the transcript impossible to
    #       follow. Identity is now sealed before the rounds loop.
    _persona_keys = list(personas.keys())
    _LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    persona_identity: dict[str, tuple[str, str]] = {
        k: _make_identity(_LETTERS[i] if i < len(_LETTERS) else f"X{i+1}")
        for i, k in enumerate(_persona_keys)
    }

    # outputs_dir / out_file / out_file_abs / ts are already computed
    # above at the bg-fork point so the bg path can announce the output
    # path before spawning. Reuse those — don't redefine here.

    # Lead model defaults to the current session model unless --lead overrode.
    lead_model = lead_model_override or curr_model

    brainstorm_history = []
    ok(f"Starting {agent_count}-Agent Brainstorming Session on: {clr(user_topic, 'bold')}")
    if persona_models:
        info(clr(f"Multi-model debate across: {', '.join(persona_models)}", "dim"))
    if lead_model != curr_model:
        info(clr(f"Lead moderator: {lead_model}", "dim"))

    # Heads-up if the lead model is in a known weak family — the lead
    # role (opening / probes / synthesis) carries most of the quality
    # weight; a weak lead leaves the personas un-moderated and the
    # synthesis flat. We never override silently — just inform.
    if _is_weak_lead_model(lead_model):
        warn(
            f"  Lead model `{lead_model}` is a small/weak family — "
            f"opening + probes + synthesis quality will suffer."
        )
        info(clr(
            "  Tip: pass `--lead claude-opus-4-7` (or any strong model) "
            "to keep weak personas but get a strong moderator. "
            "Free option: `--lead nim/deepseek-ai/deepseek-r1` "
            "(NIM free tier, no payment).",
            "dim",
        ))

    # ── Stage 1: Lead opening — set the agenda + reject filler. ──────────
    if config.get("_bg_id"):
        _bg_set_stage(config["_bg_id"], "lead opening")
    info(clr("Lead moderator framing the debate...", "dim"))
    _start_tool_spinner()
    lead_opening = _lead_opening(user_topic, snapshot, lead_model, config)
    _stop_tool_spinner()
    if lead_opening:
        print(clr("  └─ Anchor set.", "dim"))
    else:
        info(clr("  (lead opening failed — personas will run without an anchor)", "dim"))

    info(clr("Generating diverse perspectives...", "dim"))

    def _model_for_index(i: int) -> str:
        """Round-robin model assignment across personas. Falls back to the
        session's current model when no `--models` was given."""
        if persona_models:
            return persona_models[i % len(persona_models)]
        return curr_model

    def call_persona(persona_name, p_data, history, persona_model: str,
                     anchor: str, round_num: int = 1, total_rounds: int = 1,
                     follow_up: str = ""):
        # Pull the pre-assigned, stable (letter, name) for this persona.
        # Sealed once before the rounds loop — see persona_identity above.
        letter, name = persona_identity[persona_name]
        anchor_block = (
            f"\nDEBATE ANCHOR (set by the lead moderator — adhere to this):\n{anchor}\n"
            if anchor else ""
        )

        # Round-aware instructions. Round 1 is "stake your position";
        # round 2+ is "engage with what others said, do NOT repeat".
        # Without this distinction, multi-round just produces N copies
        # of each persona's first take, which is not a brainstorm.
        if round_num == 1:
            instructions = (
                "1. Provide 3-5 concrete, actionable insights or ideas from "
                "your expert perspective on the topic. Adhere to the debate "
                "anchor — concrete artifacts only, no filler.\n"
                "2. If there are prior ideas from other agents in this round, "
                "briefly acknowledge them and build upon or challenge them.\n"
                "3. Be specific, well-reasoned, and professional. Stay in "
                "character as your role.\n"
                f"4. Prefix each of your points with: [Agent {letter} — {name}]\n"
                "5. **If a `### GROUNDING DATA` section appears in your "
                "context above, you MUST cite specific results by `[N]` "
                "when your claim relates to one. If you make a claim that "
                "the grounding data does NOT support, say so explicitly — "
                "do not invent figures, prices, or statistics that don't "
                "appear there.**\n"
                "6. Output your response in clean Markdown."
            )
        else:
            instructions = (
                f"This is ROUND {round_num} of {total_rounds} — an "
                "**ADVERSARIAL CROSS-EXAMINATION** round. You are NOT here "
                "to politely reinforce other agents. You are here to find "
                "the WEAKEST CLAIM made by someone else and ATTACK it.\n\n"
                "MANDATORY (failure to do all of these = wasted round):\n\n"
                "1. **Quote a specific claim from another agent VERBATIM** "
                "(not yourself, not a summary — pick one named claim with a "
                "specific noun in it: a ticker, a number, a file path, a "
                "command). Identify them by letter, e.g. "
                "`Agent A claimed: \"...\"`.\n"
                "2. **Attack at least ONE specific weakness** in that claim. "
                "Pick from:\n"
                "   - Their data is wrong / outdated / misinterpreted\n"
                "   - Their proposed mechanism doesn't produce the outcome "
                "they predict\n"
                "   - There's a confounder, contrary case, or base rate they "
                "ignored\n"
                "   - The claim is too vague to be tested or falsified\n"
                "   - The claim contradicts a stronger claim already in the "
                "debate (cite which one)\n"
                "3. **Propose a falsifiable counter-claim** with at least one "
                "specific (a number, a date, a named entity, a measurable "
                "outcome that would prove you wrong if it didn't happen).\n"
                "4. (Optional) Defend your own round-1 position against any "
                "attacks already lodged against you — but this counts SEPARATELY "
                "from the required challenge above. You can't skip the "
                "challenge by only defending yourself.\n\n"
                "FORMAT — use this exact structure for each challenge:\n"
                "```\n"
                "### [CHALLENGE → Agent X]\n"
                "> \"<quoted claim from Agent X, verbatim or near-verbatim>\"\n"
                "**Why this fails:** <one or two sentences with the specific weakness>\n"
                "**Counter:** <your falsifiable counter-claim with a specific number/name/date>\n"
                "```\n\n"
                "FORBIDDEN: \"great point\", \"I agree, and would add\", "
                "\"building on what Agent X said\", restating someone's claim "
                "without attacking it, vague approval, asking the user to "
                "decide. Synthesis is the LEAD's job in the final stage — "
                "your job in this round is to stress-test.\n\n"
                f"Prefix any defense-of-your-own-position section with: "
                f"[Agent {letter} — {name}, round {round_num} defense]\n\n"
                "Total response: 8-15 lines. Concise, specific, adversarial."
            )

        system_prompt = f"""You are {name}, the {p_data['role']}. Identity: Agent {letter}.
{p_data['desc']}

TOPIC UNDER DISCUSSION: {user_topic}
{anchor_block}
PROJECT CONTEXT (if relevant to the topic):
{snapshot}

INSTRUCTIONS:
{instructions}
"""
        if follow_up:
            user_msg = (
                f"TOPIC: {user_topic}\n\n"
                f"PRIOR DEBATE:\n{history}\n\n"
                f"FOLLOW-UP FROM LEAD MODERATOR (you must answer this directly, in 4-8 lines, with the specifics it asks for):\n{follow_up}"
            )
        elif round_num == 1:
            user_msg = (
                f"TOPIC: {user_topic}\n\n"
                f"PRIOR IDEAS FROM DEBATE:\n{history or 'No previous ideas yet. You are the first to speak.'}"
            )
        else:
            user_msg = (
                f"TOPIC: {user_topic}\n\n"
                f"FULL PRIOR DEBATE (rounds 1..{round_num - 1} — do NOT repeat any of this; engage with it):\n{history}"
            )
        full_response = []
        internal_config = config.copy()
        internal_config["no_tools"] = True
        try:
            for event in stream(persona_model, system_prompt, [{"role": "user", "content": user_msg}], [], internal_config):
                if isinstance(event, TextChunk):
                    full_response.append(event.text)
        except Exception as e:
            return f"Error from Agent {letter} (model {persona_model}): {e}"
        return "".join(full_response).strip()

    if persona_models:
        _model_summary = ", ".join(persona_models)
    else:
        _model_summary = curr_model
    full_log = [
        f"# Brainstorming Session: {user_topic}",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Personas via:** {_model_summary}",
        f"**Lead moderator:** {lead_model}",
        f"**Rounds:** {n_rounds}",
        "---",
    ]
    if lead_opening:
        full_log.append(f"## 🎯 Lead Opening\n{lead_opening}")

    # ── Stage 2: Personas — N rounds of debate. ──────────────────────────
    # Round 1 = initial positions. Round 2+ = explicit critique/revise:
    # personas see the full prior transcript and are required to engage
    # with what others said (the round-aware prompt in call_persona
    # enforces this). Lead probe may run after each persona in any round.
    for round_num in range(1, n_rounds + 1):
        if n_rounds > 1:
            label = (
                "initial positions" if round_num == 1
                else "adversarial cross-examination — agents must attack each other's claims"
            )
            if config.get("_bg_id"):
                _bg_set_stage(config["_bg_id"], f"round {round_num}/{n_rounds}")
            print(clr(f"\n  ── Round {round_num}/{n_rounds} ({label}) ──", "cyan"))
            full_log.append(f"\n---\n### Round {round_num}/{n_rounds}")

        for i, (p_name, p_data) in enumerate(personas.items()):
            icon = p_data.get("icon", "🤖")
            p_model = _model_for_index(i)
            # Pull from the pre-assigned identity map (A, B, C, …) — see
            # persona_identity build at the top of this function. Don't use
            # `p_name[0].upper()` — every persona dict key is `p1/p2/…` so
            # that always returns 'P'.
            letter = persona_identity[p_name][0]
            label = (
                f"{icon} {clr(p_data['role'], 'yellow')}"
                + (f" ({clr(p_model, 'dim')})" if persona_models else "")
            )
            info(f"{label} is thinking..." if round_num == 1
                 else f"{label} is responding to round {round_num - 1}...")
            _start_tool_spinner()
            hist_text = "\n\n".join(brainstorm_history) if brainstorm_history else ""
            content = call_persona(p_name, p_data, hist_text, p_model, lead_opening,
                                    round_num=round_num, total_rounds=n_rounds)
            _stop_tool_spinner()
            if not content:
                err(f"  └─ Failed to capture {p_name} perspective.")
                continue

            # Anti-copy-paste: in round 2+, weak models (qwen2.5 + vLLM
            # is the canonical case) sometimes spot the first persona's
            # CHALLENGE block in history and clone it verbatim with maybe
            # one word changed. Detect Jaccard similarity ≥ 0.7 against
            # any prior CHALLENGE block in the transcript and force ONE
            # regeneration with an explicit "pick a different target /
            # different angle" nudge. If it's still redundant after the
            # retry, accept it but flag it in the log so the synthesizer
            # can ignore it.
            if round_num >= 2:
                redundant, sim = _is_redundant_challenge(content, brainstorm_history)
                if redundant:
                    info(clr(
                        f"  └─ Lead flag: {p_data['role']}'s challenge is "
                        f"~{int(sim * 100)}% identical to a prior one — "
                        f"asking for a different angle.",
                        "yellow",
                    ))
                    nudge = (
                        f"Your previous attempt copied another agent's "
                        f"CHALLENGE block almost verbatim ({int(sim * 100)}% "
                        f"token overlap). That doesn't add anything to the "
                        f"debate. Pick a DIFFERENT target persona this time — "
                        f"and a DIFFERENT angle of attack. Specifically: do "
                        f"NOT challenge any of the same claims that were "
                        f"already attacked above; find a fresh weakness in "
                        f"someone else's contribution."
                    )
                    _start_tool_spinner()
                    retry = call_persona(p_name, p_data, hist_text, p_model,
                                          lead_opening, round_num=round_num,
                                          total_rounds=n_rounds, follow_up=nudge)
                    _stop_tool_spinner()
                    # Use the retry only if it's actually less redundant.
                    if retry:
                        retry_redundant, retry_sim = _is_redundant_challenge(
                            retry, brainstorm_history)
                        if not retry_redundant or retry_sim < sim:
                            content = retry
                            print(clr("  └─ Re-engaged on a different angle.", "dim"))
                        else:
                            content = (
                                f"_[lead note: contribution flagged as "
                                f"redundant — {int(retry_sim * 100)}% overlap "
                                f"with prior challenge]_\n\n{retry}"
                            )
                            print(clr("  └─ Still redundant — kept with flag.", "yellow"))

            brainstorm_history.append(content)
            heading_suffix = "" if round_num == 1 else f" — round {round_num}"
            full_log.append(f"## {icon} {p_data['role']}{heading_suffix} _(via {p_model})_\n{content}")
            print(clr(
                "  └─ Perspective captured." if round_num == 1
                else "  └─ Engagement captured.", "dim",
            ))

            # Lead probe — gives the persona one more swing if vague (in
            # round 1) or if they dodged the cross-examination (in round
            # 2+, the probe demands an actual challenge to a named agent).
            # Skipped on the very last round (no time for the persona to
            # revise after the final round anyway).
            if round_num < n_rounds:
                _start_tool_spinner()
                probe = _lead_probe(user_topic, p_data["role"], letter, content,
                                     lead_model, config, round_num=round_num)
                _stop_tool_spinner()
                if probe:
                    info(clr(f"  └─ Lead probe: {probe[:120]}", "yellow"))
                    _start_tool_spinner()
                    follow = call_persona(p_name, p_data, hist_text, p_model,
                                           lead_opening, round_num=round_num,
                                           total_rounds=n_rounds, follow_up=probe)
                    _stop_tool_spinner()
                    if follow:
                        brainstorm_history.append(
                            f"_(follow-up to lead probe, round {round_num})_\n{follow}"
                        )
                        full_log.append(
                            f"### 🔍 Lead probe + Agent {letter} reply (round {round_num})\n"
                            f"{probe}\n\n{follow}"
                        )
                        print(clr("  └─ Follow-up captured.", "dim"))

    # ── Stage 3: Lead synthesis — done HERE (not via main agent). ────────
    if config.get("_bg_id"):
        _bg_set_stage(config["_bg_id"], "synthesis")
    info(clr("Lead moderator producing final synthesis...", "dim"))
    _start_tool_spinner()
    # Pass the raw debate history (every persona turn + follow-up) directly,
    # rather than slicing full_log by index — the index drifts every time
    # the header layout changes.
    transcript_for_synth = "\n\n".join(brainstorm_history)
    lead_master_plan = _lead_synthesis(
        user_topic, transcript_for_synth, lead_model, config,
        opening=lead_opening,
        grounding=grounding_block,
    )

    # Programmatic backstops — both deterministic, both run AFTER the
    # lead's synthesis is back. They don't replace the prompt-side
    # SELF-CHECK; they catch the cases where the lead model (especially
    # weak ones like qwen2.5) read the SELF-CHECK instruction and
    # ignored it. See "When NOT to use" in docs/guides/brainstorm.md.
    if lead_master_plan:
        # 1) Ensure the Consensus section is ranked. If the lead skipped
        #    the rank requirement, do ONE fallback LLM call to add it.
        before_rank = lead_master_plan
        lead_master_plan = _ensure_consensus_is_ranked(
            lead_master_plan, user_topic, lead_model, config,
        )
        if lead_master_plan != before_rank:
            print(clr("  └─ Consensus re-ranked programmatically.", "dim"))

        # 2) Filter the Action Plan against the ban list. Drop any item
        #    whose text contains a banned keyword (default set + topic-
        #    specific extracted from opening). Deterministic — runs
        #    regardless of what the lead said it would do.
        ban_kws = _extract_ban_keywords(lead_opening)
        lead_master_plan, removed_items = _filter_action_plan(
            lead_master_plan, ban_kws,
        )
        if removed_items:
            warn(
                f"  └─ Programmatic self-check removed "
                f"{len(removed_items)} action(s) for matching the ban list."
            )
            for r in removed_items[:3]:
                info(clr(f"      • {r[:140]}", "dim"))
    _stop_tool_spinner()
    if lead_master_plan:
        full_log.append("---\n## 📋 Lead Synthesis — Master Plan\n" + lead_master_plan)
        print(clr("  └─ Synthesis complete.", "dim"))
    else:
        warn("  └─ Lead synthesis failed — falling back to bare debate transcript.")

    final_output = "\n\n".join(full_log)
    out_file.write_text(final_output, encoding="utf-8")
    ok(f"Brainstorming complete! Results saved to {clr(str(out_file_abs), 'bold')}")

    # The TODO prompt INLINES the master plan, so the main agent does NOT
    # need to Read the file — eliminates the duplicate-Read pattern that
    # weak models (qwen2.5 etc.) were prone to. If the lead failed to
    # produce a plan, fall back to telling the agent to read the file.
    if lead_master_plan:
        todo_payload = (
            "I just ran a multi-persona brainstorming session moderated by a "
            f"lead model. Here is the lead's final master plan for: '{user_topic}'.\n\n"
            f"--- BEGIN MASTER PLAN ---\n{lead_master_plan}\n--- END MASTER PLAN ---\n\n"
            "There is NOTHING to read — the plan is above, in your context. "
            "Use the master plan above directly."
        )
    else:
        todo_payload = (
            f"A brainstorming session ran but the lead synthesis failed. "
            f"Read {out_file_abs} (use this absolute path verbatim) and "
            "produce a master plan from the raw debate."
        )

    # In bg-recursion the file is already written and there's no REPL
    # waiting for the sentinel — just return True so the bg thread exits
    # cleanly and the daemon-thread runner can announce completion.
    if config.get("_bg_recursion"):
        return True
    return ("__brainstorm__", todo_payload, str(out_file_abs))


def _save_synthesis(state, out_file: str) -> None:
    """Append the last assistant response as the synthesis section of the brainstorm file."""
    for msg in reversed(state.messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            return
        text = text.strip()
        if not text:
            return
        try:
            with Path(out_file).open("a", encoding="utf-8") as f:
                f.write("\n\n---\n\n## 🧠 Synthesis — Master Plan\n\n")
                f.write(text)
                f.write("\n")
            ok(f"Synthesis appended to {clr(out_file, 'bold')}")
        except Exception as e:
            err(f"Failed to save synthesis: {e}")
        return


# ── Draft (semi-automatic reply suggestion) ────────────────────────────────

def cmd_draft(args: str, _state, config) -> bool:
    """Draft 3 candidate replies for a message someone sent you.

    Usage:
        /draft <对方刚发的消息>
        /draft @<uid_or_label> <对方刚发的消息>

    The auxiliary cheap model (config.auxiliary_model) drafts 3 candidates.
    With @<uid_or_label>, contact relationship/notes from
    ~/.cheetahclaws/wx_contacts.json are used to tune tone and language.
    Past confirmed sends from the smart-reply history (if any) feed style
    mimicking automatically.
    """
    raw = args.strip()
    if not raw:
        err("Usage: /draft <对方的消息>")
        info("       /draft @<uid_or_label> <对方的消息>")
        info("Example: /draft 周末有空吗")
        info("         /draft @wxid_alice 周末有空吗")
        return True

    contact = None
    sender_label = "对方"
    if raw.startswith("@"):
        head, _, rest = raw.partition(" ")
        if rest.strip():
            key = head[1:]
            try:
                from bridges.wechat_smart_reply import ContactsStore
                store = ContactsStore()
                c = store.get(key)
                if c is None:
                    # Fallback: match by label (case-insensitive) in the JSON
                    for uid, entry in store._data.items():
                        if (entry.get("label") or "").lower() == key.lower():
                            c = store.get(uid)
                            break
                if c:
                    contact = c
                    sender_label = c.label or key
                else:
                    warn(f"No contact '{key}' in ~/.cheetahclaws/wx_contacts.json — using generic tone")
                    sender_label = key
            except Exception as _e:
                warn(f"Contact lookup failed ({_e}); using generic tone")
                sender_label = key
            raw = rest.strip()

    msg_preview = raw if len(raw) <= 80 else raw[:77] + "…"
    info(f"  Drafting 3 replies for {clr(sender_label, 'bold')} → 「{msg_preview}」")

    history = []
    try:
        from bridges.wechat_smart_reply_store import make_store
        store = make_store(timeout_s=300)
        if hasattr(store, "recent_replies"):
            # Mirror the smart-reply path: exclude this contact so we don't
            # feed earlier drafts to the same thread back as "style examples".
            history = store.recent_replies(
                n=10,
                exclude_uid=contact.uid if contact else None,
            )
    except Exception:
        pass  # history is purely a quality boost, never required

    _start_tool_spinner()
    try:
        from bridges.wechat_smart_reply import generate_candidates
        candidates = generate_candidates(
            raw, sender_label, config,
            contact=contact, history=history,
        )
    finally:
        _stop_tool_spinner()

    if not candidates:
        err("Drafting failed — auxiliary model returned nothing usable.")
        info("Check `auxiliary_model` in /config or fall back by setting it to your main model.")
        return True

    print()
    for i, c in enumerate(candidates, 1):
        print(clr(f"  [{i}] ", "cyan", "bold") + c)
    print()
    info(clr("Copy one and paste it into WeChat (or wherever) — this is fully manual.", "dim"))

    # If we're inside a bridge turn (the user typed /draft from WeChat /
    # Telegram / Slack), echo the candidates back to that channel too so
    # they can see and copy them on their phone.  Best-effort: failures
    # never break the terminal output.
    try:
        import runtime
        ctx = runtime.get_ctx(config)
        bridge_text = (
            f"💬 Drafts for 「{msg_preview}」\n\n"
            + "\n".join(f"[{i}] {c}" for i, c in enumerate(candidates, 1))
            + "\n\n回 1/2/3 取那条 · 复制粘贴给对方"
        )
        wx_uid = getattr(ctx, "wx_current_user_id", "") or ""
        if wx_uid:
            from bridges.wechat import _wx_send
            from bridges.draft_cache import put as _draft_put
            # Stash so a follow-up "1"/"2"/"3" from this uid resolves to
            # the chosen candidate (handled in bridges/wechat.py inbound).
            _draft_put(wx_uid, candidates)
            _wx_send(wx_uid, bridge_text, config)
        if getattr(ctx, "in_telegram_turn", False):
            chat_id = config.get("telegram_chat_id")
            token = config.get("telegram_token")
            if chat_id and token:
                from bridges.telegram import _tg_send
                _tg_send(token, chat_id, bridge_text)
        slack_chan = getattr(ctx, "slack_current_channel", "") or ""
        if slack_chan:
            slack_token = config.get("slack_bot_token") or config.get("slack_token")
            if slack_token:
                try:
                    from bridges.slack import _slack_send
                    _slack_send(slack_token, slack_chan, bridge_text)
                except Exception:
                    pass
    except Exception:
        pass

    return True


# ── Worker ─────────────────────────────────────────────────────────────────

def cmd_worker(args: str, state, config) -> bool:
    """Auto-implement pending tasks from a todo_list.txt file."""
    raw = args.strip()
    todo_path_override = None
    task_nums_str      = None
    max_workers        = None

    tokens = raw.split() if raw else []
    remaining = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--path" and i + 1 < len(tokens):
            todo_path_override = tokens[i + 1]; i += 2
        elif tok.startswith("--path="):
            todo_path_override = tok[len("--path="):]; i += 1
        elif tok == "--tasks" and i + 1 < len(tokens):
            task_nums_str = tokens[i + 1]; i += 2
        elif tok.startswith("--tasks="):
            task_nums_str = tok[len("--tasks="):]; i += 1
        elif tok == "--workers" and i + 1 < len(tokens):
            max_workers = tokens[i + 1]; i += 2
        elif tok.startswith("--workers="):
            max_workers = tok[len("--workers="):]; i += 1
        else:
            remaining.append(tok); i += 1

    if remaining:
        leftover = " ".join(remaining)
        if todo_path_override is None and (
            "/" in leftover or "\\" in leftover
            or leftover.endswith(".txt") or leftover.endswith(".md")
        ):
            todo_path_override = leftover
        elif task_nums_str is None:
            task_nums_str = leftover

    todo_path = Path(todo_path_override) if todo_path_override else Path("brainstorm_outputs") / "todo_list.txt"

    if not todo_path.exists():
        err(f"No todo file found at {todo_path}.")
        if not todo_path_override:
            info("Run /brainstorm first, or specify a path with --path /your/todo.txt")
        return True

    content = todo_path.read_text(encoding="utf-8", errors="replace")
    lines   = content.splitlines()
    pending = [(i, ln) for i, ln in enumerate(lines) if ln.strip().startswith("- [ ]")]

    if not pending:
        any_tasks = any(ln.strip().startswith("- [") for ln in lines)
        if any_tasks:
            ok(f"All tasks completed! No pending items in {todo_path}.")
        else:
            err(f"No task lines found in {todo_path}.")
            info("Worker expects lines like:  - [ ] task description")
        return True

    if task_nums_str:
        try:
            nums = [int(x.strip()) for x in task_nums_str.split(",") if x.strip()]
            selected = []
            for n in nums:
                if 1 <= n <= len(pending):
                    selected.append(pending[n - 1])
                else:
                    err(f"Task #{n} out of range (1-{len(pending)}).")
                    return True
            pending = selected
        except ValueError:
            err(f"Invalid task number(s): '{task_nums_str}'. Use e.g. 1,4,6")
            return True

    worker_count = len(pending)
    if max_workers is not None:
        try:
            worker_count = max(1, int(max_workers))
        except ValueError:
            err(f"Invalid --workers value: '{max_workers}'. Must be a positive integer.")
            return True
    if worker_count < len(pending):
        info(f"Workers: {worker_count} — running first {worker_count} of {len(pending)} pending task(s) this session.")
        pending = pending[:worker_count]

    ok(f"Worker starting — {len(pending)} task(s) | file: {todo_path}")
    info("Pending tasks:")
    for n, (_, ln) in enumerate(pending, 1):
        print(f"  {n}. {ln.strip()}")

    worker_prompts = []
    for line_idx, task_line in pending:
        task_text = task_line.strip().replace("- [ ] ", "", 1)
        prompt = (
            f"You are the Worker. Your job is to implement this task:\n\n"
            f"  {task_text}\n\n"
            f"Instructions:\n"
            f"1. Read the relevant files, understand the codebase.\n"
            f"2. Implement the task — write code, edit files, run tests.\n"
            f"3. When DONE, use the Edit tool to mark this exact line in {todo_path}:\n"
            f'   Change "- [ ] {task_text}" to "- [x] {task_text}"\n'
            f"4. If you CANNOT complete it, leave it as - [ ] and explain why.\n"
            f"5. Be concise. Act, don't explain."
        )
        worker_prompts.append((line_idx, task_text, prompt))

    return ("__worker__", worker_prompts)


# ── SSJ Trading Sub-menu ───────────────────────────────────────────────────

def _ssj_trading_submenu(config, state):
    """Interactive trading sub-menu for SSJ mode."""
    from tools import ask_input_interactive

    _TRADING_SUBMENU = (
        clr("\n╭─ 📈 Trading Agent ", "dim") + clr("━━━━━━━━━━━━━━━━━━━━━━━━━", "dim")
        + "\n│"
        + "\n│  " + clr("a.", "bold") + " 🔍  Quick Analyze — Full multi-agent analysis (Bull/Bear + Risk + PM)"
        + "\n│  " + clr("b.", "bold") + " 📊  Backtest     — Test a strategy on historical data"
        + "\n│  " + clr("c.", "bold") + " 💰  Price Check  — Current price & key metrics"
        + "\n│  " + clr("d.", "bold") + " 📉  Indicators   — Technical indicators report"
        + "\n│  " + clr("e.", "bold") + " 🤖  Trading Bot  — Launch autonomous trading agent"
        + "\n│  " + clr("f.", "bold") + " 📜  History      — Past trading decisions"
        + "\n│  " + clr("g.", "bold") + " 🧠  Memory       — Trading memory status"
        + "\n│  " + clr("0.", "bold") + " ↩   Back to SSJ"
        + "\n│"
        + "\n" + clr("╰━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "dim")
    )

    print(_TRADING_SUBMENU)
    try:
        choice = ask_input_interactive(clr("\n  📈 Trading » ", "cyan", "bold"), config, _TRADING_SUBMENU).strip().lower()
    except (KeyboardInterrupt, EOFError):
        return True

    if choice in ("0", "q", ""):
        return True

    elif choice == "a":
        symbol = ask_input_interactive(
            clr("  Symbol (e.g. AAPL, BTC, NVDA): ", "cyan"), config
        ).strip().upper()
        if not symbol:
            err("Symbol required.")
            return True
        return ("__ssj_passthrough__", f"/trading analyze {symbol}")

    elif choice == "b":
        symbol = ask_input_interactive(
            clr("  Symbol (e.g. AAPL, BTC): ", "cyan"), config
        ).strip().upper()
        if not symbol:
            err("Symbol required.")
            return True

        _STRAT_MENU = (
            clr("\n  Strategies:", "cyan")
            + "\n    1. dual_ma          — SMA(20/50) crossover (trend following)"
            + "\n    2. rsi_mean_reversion — RSI 30/70 (mean reversion)"
            + "\n    3. bollinger_breakout — Bollinger Band breakout"
            + "\n    4. macd_crossover   — MACD histogram (momentum)"
            + "\n    5. all              — Run all & compare"
        )
        print(_STRAT_MENU)
        strat_choice = ask_input_interactive(
            clr("  Strategy [1-5, default=5]: ", "cyan"), config, _STRAT_MENU
        ).strip()

        strat_map = {"1": "dual_ma", "2": "rsi_mean_reversion",
                     "3": "bollinger_breakout", "4": "macd_crossover"}

        if strat_choice in ("5", ""):
            # Run all strategies and compare
            return ("__ssj_query__",
                    f"Run backtests on {symbol} using all 4 built-in strategies "
                    f"(dual_ma, rsi_mean_reversion, bollinger_breakout, macd_crossover). "
                    f"Use the RunBacktest tool for each. Present results in a comparison table "
                    f"and recommend the best strategy based on Sharpe ratio.")
        elif strat_choice in strat_map:
            strategy = strat_map[strat_choice]
            return ("__ssj_passthrough__", f"/trading backtest {symbol} {strategy}")
        else:
            err(f"Invalid choice: {strat_choice}")
            return True

    elif choice == "c":
        symbol = ask_input_interactive(
            clr("  Symbol (e.g. AAPL, BTC, ETH): ", "cyan"), config
        ).strip().upper()
        if symbol:
            return ("__ssj_passthrough__", f"/trading price {symbol}")
        err("Symbol required.")
        return True

    elif choice == "d":
        symbol = ask_input_interactive(
            clr("  Symbol (e.g. AAPL, NVDA): ", "cyan"), config
        ).strip().upper()
        if symbol:
            return ("__ssj_passthrough__", f"/trading indicators {symbol}")
        err("Symbol required.")
        return True

    elif choice == "e":
        watchlist = ask_input_interactive(
            clr("  Watchlist (comma-separated, default: AAPL,MSFT,GOOGL,NVDA,BTC,ETH): ", "cyan"), config
        ).strip()
        if not watchlist:
            watchlist = "AAPL,MSFT,GOOGL,NVDA,BTC,ETH"
        return ("__ssj_query__",
                f"You are the CheetahClaws Trading Agent. Analyze each symbol in this watchlist: {watchlist}. "
                f"For each symbol:\n"
                f"1. Use GetPrice and GetTechnicalIndicators to gather data\n"
                f"2. Run the full multi-agent analysis pipeline:\n"
                f"   - Bull Researcher: build bullish case with data\n"
                f"   - Bear Researcher: build bearish case with data\n"
                f"   - Research Judge: decisive BUY/SELL/HOLD recommendation\n"
                f"   - Risk Panel: aggressive/conservative/neutral perspectives\n"
                f"   - Portfolio Manager: final RATING (BUY/OVERWEIGHT/HOLD/UNDERWEIGHT/SELL)\n"
                f"3. Present a summary table at the end with all symbols and ratings.\n"
                f"Be specific — cite actual indicator values and fundamentals.")

    elif choice == "f":
        return ("__ssj_passthrough__", "/trading history")

    elif choice == "g":
        return ("__ssj_passthrough__", "/trading status")

    else:
        err(f"Invalid choice: {choice}")
    return True


# ── SSJ ────────────────────────────────────────────────────────────────────

def cmd_ssj(args: str, state, config) -> bool:
    """SSJ Developer Mode — Interactive power menu for project workflows."""
    try:
        import modular
        _all_cmds = modular.load_all_commands()
        _VIDEO_AVAILABLE    = "video"   in _all_cmds
        _VOICE_MODULAR      = "voice"   in _all_cmds
        _TRADING_AVAILABLE  = "trading" in _all_cmds
    except Exception:
        _VIDEO_AVAILABLE   = False
        _VOICE_MODULAR     = False
        _TRADING_AVAILABLE = False

    from tools import ask_input_interactive

    _SSJ_MENU = (
        clr("\n╭─ SSJ Developer Mode ", "dim") + clr("⚡", "yellow") + clr(" ─────────────────────────", "dim")
        + "\n│"
        + "\n│  " + clr(" 1.", "bold") + " 💡  Brainstorm — Multi-persona AI debate"
        + "\n│  " + clr(" 2.", "bold") + " 📋  Show TODO — View todo_list.txt"
        + "\n│  " + clr(" 3.", "bold") + " 👷  Worker — Auto-implement pending tasks"
        + "\n│  " + clr(" 4.", "bold") + " 🧠  Debate — Expert debate on a file"
        + "\n│  " + clr(" 5.", "bold") + " ✨  Propose — AI improvement for a file"
        + "\n│  " + clr(" 6.", "bold") + " 🔎  Review — Quick file analysis"
        + "\n│  " + clr(" 7.", "bold") + " 📘  Readme — Auto-generate README.md"
        + "\n│  " + clr(" 8.", "bold") + " 💬  Commit — AI-suggested commit message"
        + "\n│  " + clr(" 9.", "bold") + " 🧪  Scan — Analyze git diff"
        + "\n│  " + clr("10.", "bold") + " 📝  Promote — Idea to tasks"
        + ("\n│  " + clr("11.", "bold") + " 🎬  Video — AI video content factory" if _VIDEO_AVAILABLE else "")
        + ("\n│  " + clr("12.", "bold") + " 🎙  TTS   — AI voice generation (any style)" if _VOICE_MODULAR else "")
        + "\n│  " + clr("13.", "bold") + " 📡  Monitor — AI subscriptions & alerts"
        + ("\n│  " + clr("14.", "bold") + " 📈  Trading — Market analysis, backtest & trading agent" if _TRADING_AVAILABLE else "")
        + "\n│  " + clr("15.", "bold") + " 🤖  Agent  — Autonomous task agents (research / bug-fix / code / write)"
        + "\n│  " + clr("16.", "bold") + " 🔍  Research — 20-source topic search (arXiv · HN · GitHub · Zhihu · B站 · Weibo · 小红书 · …)"
        + "\n│  " + clr("17.", "bold") + " 📊  Trend Track — Auto-rerun /research weekly on a topic (/monitor hookup)"
        + "\n│  " + clr("18.", "bold") + " 📁  Reports   — Browse saved research briefs"
        + "\n│  " + clr(" 0.", "bold") + " 🚪  Exit SSJ Mode  (or type q)"
        + "\n│"
        + "\n" + clr("╰──────────────────────────────────────────────", "dim")
    )

    def _pick_file(prompt_text="  Select file #: ", exts=None):
        files = sorted([
            f for f in Path(".").iterdir()
            if f.is_file() and not f.name.startswith(".")
            and (exts is None or f.suffix in exts)
        ])
        if not files:
            err("No matching files found in current directory.")
            return None
        menu_text = clr(f"\n  📂 Files in {Path.cwd().name}/", "cyan")
        for i, f in enumerate(files, 1):
            menu_text += ("\n" + f"  {i:3d}. {f.name}")
        sel = ask_input_interactive(clr(prompt_text, "cyan"), config, menu_text).strip()
        if sel.isdigit() and 1 <= int(sel) <= len(files):
            return str(files[int(sel) - 1])
        elif sel:
            return sel
        err("Invalid selection.")
        return None

    print(_SSJ_MENU)

    while True:
        try:
            choice = ask_input_interactive(clr("\n  ⚡ SSJ » ", "yellow", "bold"), config, _SSJ_MENU).strip()
        except (KeyboardInterrupt, EOFError):
            break

        if choice.startswith("/"):
            return ("__ssj_passthrough__", choice)

        if choice == "0" or choice.lower() in ("exit", "q"):
            ok("Exiting SSJ Mode.")
            break

        elif choice == "1":
            topic = ask_input_interactive(clr("  Topic (Enter for general): ", "cyan"), config).strip()
            return ("__ssj_cmd__", "brainstorm", topic)

        elif choice == "2":
            todo_path = Path("brainstorm_outputs") / "todo_list.txt"
            if todo_path.exists():
                content = todo_path.read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()
                task_lines = [(i, l) for i, l in enumerate(lines) if l.strip().startswith("- [")]
                pending_lines = [(i, l) for i, l in task_lines if l.strip().startswith("- [ ]")]
                done_lines = [(i, l) for i, l in task_lines if l.strip().startswith("- [x]")]
                pending = len(pending_lines)
                done = len(done_lines)
                print(clr(f"\n  📋 TODO List ({done} done / {pending} pending):", "cyan"))
                print(clr("  " + "─" * 46, "dim"))
                for _, ln in done_lines:
                    label = ln.strip()[5:].strip()
                    print(clr(f"       ✓ {label}", "green"))
                for num, (_, ln) in enumerate(pending_lines, 1):
                    label = ln.strip()[5:].strip()
                    print(f"  {num:3d}. ○ {label}")
                print(clr("  " + "─" * 46, "dim"))
                print(clr("  Tip: use Worker (3) with pending task #s e.g. 1,4,6", "dim"))
            else:
                err("No todo_list.txt found. Run Brainstorm (1) first.")
            print(_SSJ_MENU)
            continue

        elif choice == "3":
            _default_todo = Path("brainstorm_outputs") / "todo_list.txt"
            pending_count = 0
            if _default_todo.exists():
                _lines = _default_todo.read_text(encoding="utf-8", errors="replace").splitlines()
                pending_count = sum(1 for l in _lines if l.strip().startswith("- [ ]"))
                if pending_count:
                    print(clr(f"\n  👷 Worker — {pending_count} pending task(s) in {_default_todo}", "cyan"))
                else:
                    print(clr(f"\n  ✓ All tasks completed in {_default_todo}", "green"))
            task_sel = ask_input_interactive(
                clr("  Task #s to run (e.g. 1,3,5), or Enter for all: ", "cyan"), config
            ).strip()
            return ("__ssj_cmd__", "worker", task_sel)

        elif choice == "4":
            fpath = _pick_file("  Select file for debate: ")
            if fpath:
                return ("__ssj_query__", f"Act as a panel of 3 expert engineers. Each gives 2-3 critical insights on this file: {fpath}. Be specific and constructive.")

        elif choice == "5":
            fpath = _pick_file("  Select file to improve: ")
            if fpath:
                return ("__ssj_query__", f"Analyze {fpath} and propose 3 high-impact improvements with code examples. Focus on correctness, performance, or maintainability.")

        elif choice == "6":
            fpath = _pick_file("  Select file to review: ")
            if fpath:
                return ("__ssj_query__", f"Give a quick code review of {fpath}: identify bugs, code smells, or missing edge cases. Be concise.")

        elif choice == "7":
            return ("__ssj_query__", "Generate a comprehensive README.md for this project. Include: project description, features, installation, usage examples, and contributing guidelines. Use the project files and CLAUDE.md for context.")

        elif choice == "8":
            return ("__ssj_query__", "Review the git diff (git diff HEAD) and suggest a concise, descriptive commit message following conventional commits format. Also list files changed.")

        elif choice == "9":
            return ("__ssj_query__", "Run git diff HEAD and analyze the changes. Summarize what was changed, why it might have been changed, and flag any potential issues or regressions.")

        elif choice == "10":
            idea = ask_input_interactive(clr("  Describe your idea or feature: ", "cyan"), config).strip()
            if idea:
                return ("__ssj_promote_worker__", idea)

        elif choice == "11" and _VIDEO_AVAILABLE:
            return ("__ssj_passthrough__", "/video")

        elif choice == "12" and _VOICE_MODULAR:
            return ("__ssj_passthrough__", "/tts")

        elif choice == "13":
            return ("__ssj_passthrough__", "/monitor")

        elif choice == "14" and _TRADING_AVAILABLE:
            return _ssj_trading_submenu(config, state)

        elif choice == "15":
            return ("__ssj_passthrough__", "/agent")

        elif choice == "16":
            # 傻瓜式研究向导:问 topic → 可选 range + citations → 直接跑 /research
            topic = ask_input_interactive(
                clr("  Topic to research: ", "cyan"), config
            ).strip()
            if not topic:
                err("No topic given.")
                print(_SSJ_MENU)
                continue
            range_menu = clr(
                "\n  Time range:"
                "\n    1. Last 7 days"
                "\n    2. Last 30 days  (default)"
                "\n    3. Last 6 months"
                "\n    4. Last 1 year"
                "\n    5. All time",
                "cyan",
            )
            range_choice = ask_input_interactive(
                clr("  Range [1-5, Enter for 2]: ", "cyan"), config, range_menu
            ).strip() or "2"
            range_map = {"1": "7d", "2": "30d", "3": "6m", "4": "1y", "5": "all"}
            r = range_map.get(range_choice, "30d")
            cite_choice = ask_input_interactive(
                clr("  Include notable-citer analysis? (y/N): ", "cyan"), config
            ).strip().lower()
            cmd_line = f"/research --range {r}"
            if cite_choice.startswith("y"):
                cmd_line += " --citations"
            cmd_line += f" {topic}"
            return ("__ssj_passthrough__", cmd_line)

        elif choice == "17":
            topic = ask_input_interactive(
                clr("  Topic to track weekly: ", "cyan"), config
            ).strip()
            if not topic:
                err("No topic given.")
                print(_SSJ_MENU)
                continue
            range_menu = clr(
                "\n  Tracking window (what each weekly run pulls):"
                "\n    1. Last 7 days    (default — matches the weekly schedule)"
                "\n    2. Last 30 days"
                "\n    3. Last 3 months",
                "cyan",
            )
            rc = ask_input_interactive(
                clr("  Window [1-3, Enter for 1]: ", "cyan"), config, range_menu
            ).strip() or "1"
            rmap = {"1": "7d", "2": "30d", "3": "90d"}
            rng = rmap.get(rc, "7d")
            freq_menu = clr(
                "\n  Frequency:"
                "\n    1. Daily"
                "\n    2. Weekly  (default — recommended for trend tracking)"
                "\n    3. Every 12 hours",
                "cyan",
            )
            fc = ask_input_interactive(
                clr("  Frequency [1-3, Enter for 2]: ", "cyan"), config, freq_menu
            ).strip() or "2"
            freq_map = {"1": "daily", "2": "weekly", "3": "12h"}
            freq = freq_map.get(fc, "weekly")
            topic_id = f"research:{rng}:{topic}"
            cmd_line = f"/subscribe {topic_id} {freq}"
            ok(f"  Subscribing to {topic_id}  ({freq})")
            return ("__ssj_passthrough__", cmd_line)

        elif choice == "18":
            return ("__ssj_passthrough__", "/reports")

        else:
            err(f"Invalid choice: {choice}")

        print(_SSJ_MENU)

    return True


# ── Summarize (multi-agent map-reduce) ────────────────────────────────────


def cmd_summarize(args: str, _state, config) -> bool:
    """Summarize a (potentially large) file via multi-agent map-reduce.

    Usage:
        /summarize <abs-path> [focus phrase]

    Calls the same SummarizeLargeFile tool the /agent flows use, but
    inline so the user gets the summary printed directly. Number of
    chunks is adaptive to file size — works on files of any size."""
    parts = args.strip().split(maxsplit=1)
    if not parts:
        info("Usage: /summarize <absolute-path> [focus phrase]")
        info("Reads any size file (PDF / txt / md / code), chunks adaptively,")
        info("summarizes each chunk in parallel via sub-LLM calls, merges into")
        info("one unified summary. No context-window limit.")
        return True

    file_path = parts[0]
    focus = parts[1] if len(parts) > 1 else ""

    p = Path(file_path)
    if not p.is_absolute():
        # Resolve relative to cwd for convenience
        p = Path.cwd() / file_path
    if not p.exists():
        err(f"File not found: {p}")
        return True

    info(clr(f"Summarizing {p.name} via multi-agent map-reduce…", "dim"))
    if focus:
        info(clr(f"  Focus: {focus}", "dim"))
    _start_tool_spinner()
    from tools.files import _summarize_large_file
    try:
        result = _summarize_large_file(
            {"file_path": str(p), "focus": focus}, config,
        )
    finally:
        _stop_tool_spinner()

    print()
    print(result)
    print()
    return True


# ── Memory ─────────────────────────────────────────────────────────────────

def cmd_memory(args: str, _state, config) -> bool:
    from memory import search_memory, load_index
    from memory.scan import scan_all_memories, format_memory_manifest, memory_freshness_text

    stripped = args.strip()

    if stripped == "consolidate":
        from memory import consolidate_session
        msgs = _state.get("messages", []) if hasattr(_state, 'get') else getattr(_state, 'messages', [])
        info("  Analyzing session for long-term memories…")
        saved = consolidate_session(msgs, config)
        if saved:
            info(f"  ✓ Consolidated {len(saved)} memory/memories: {', '.join(saved)}")
        else:
            info("  Nothing new worth saving (session too short, or nothing extractable).")
        return True

    if stripped:
        results = search_memory(stripped)
        if not results:
            info(f"No memories matching '{stripped}'")
            return True
        info(f"  {len(results)} result(s) for '{stripped}':")
        for m in results:
            conf_tag = f" conf:{m.confidence:.0%}" if m.confidence < 1.0 else ""
            src_tag = f" src:{m.source}" if m.source and m.source != "user" else ""
            info(f"  [{m.type:9s}|{m.scope:7s}] {m.name}{conf_tag}{src_tag}: {m.description}")
            info(f"    {m.content[:120]}{'...' if len(m.content) > 120 else ''}")
        return True

    headers = scan_all_memories()
    if not headers:
        info("No memories stored. The model saves memories via MemorySave.")
        return True
    info(f"  {len(headers)} memory/memories (newest first):")
    for h in headers:
        fresh_warn = "  ⚠ stale" if memory_freshness_text(h.mtime_s) else ""
        tag = f"[{h.type or '?':9s}|{h.scope:7s}]"
        info(f"  {tag} {h.filename}{fresh_warn}")
        if h.description:
            info(f"    {h.description}")
    return True


# ── Agents ─────────────────────────────────────────────────────────────────

def cmd_agents(_args: str, _state, config) -> bool:
    try:
        from multi_agent.tools import get_agent_manager
        mgr = get_agent_manager()
        tasks = mgr.list_tasks()
        if not tasks:
            info("No sub-agent tasks.")
            return True
        info(f"  {len(tasks)} sub-agent task(s):")
        for t in tasks:
            preview = t.prompt[:50] + ("..." if len(t.prompt) > 50 else "")
            wt_info = f"  branch:{t.worktree_branch}" if t.worktree_branch else ""
            info(f"  {t.id} [{t.status:9s}] name={t.name}{wt_info}  {preview}")
    except Exception:
        info("Sub-agent system not initialized.")
    return True


def _print_background_notifications():
    """Print notifications for newly completed background agent tasks."""
    try:
        from multi_agent.tools import get_agent_manager
        mgr = get_agent_manager()
    except Exception:
        return

    if not hasattr(_print_background_notifications, "_seen"):
        _print_background_notifications._seen = set()

    for task in mgr.list_tasks():
        if task.id in _print_background_notifications._seen:
            continue
        if task.status in ("completed", "failed", "cancelled"):
            _print_background_notifications._seen.add(task.id)
            icon = "✓" if task.status == "completed" else "✗"
            color = "green" if task.status == "completed" else "red"
            branch_info = f" [branch: {task.worktree_branch}]" if task.worktree_branch else ""
            print(clr(
                f"\n  {icon} Background agent '{task.name}' {task.status}{branch_info}",
                color, "bold"
            ))
            if task.result:
                preview = task.result[:200] + ("..." if len(task.result) > 200 else "")
                print(clr(f"    {preview}", "dim"))
            print()


# ── Skills ─────────────────────────────────────────────────────────────────

def cmd_skills(_args: str, _state, config) -> bool:
    from skill import load_skills
    skills = load_skills()
    if not skills:
        info("No skills found.")
        return True
    info(f"Available skills ({len(skills)}):")
    for s in skills:
        triggers = ", ".join(s.triggers)
        source_label = f"[{s.source}]" if s.source != "builtin" else ""
        hint = f"  args: {s.argument_hint}" if s.argument_hint else ""
        print(f"  {clr(s.name, 'cyan'):24s} {s.description}  {clr(triggers, 'dim')}{hint} {clr(source_label, 'yellow')}")
        if s.when_to_use:
            print(f"    {clr(s.when_to_use[:80], 'dim')}")
    return True


# ── MCP ────────────────────────────────────────────────────────────────────

def cmd_mcp(args: str, _state, config) -> bool:
    """Show MCP server status, or manage servers."""
    from cc_mcp.client import get_mcp_manager
    from cc_mcp.config import (load_mcp_configs, add_server_to_user_config,
                                remove_server_from_user_config, list_config_files)
    from cc_mcp.tools import initialize_mcp, reload_mcp, refresh_server

    parts = args.split() if args.strip() else []
    subcmd = parts[0].lower() if parts else ""

    if subcmd == "reload":
        target = parts[1] if len(parts) > 1 else ""
        if target:
            mcp_err = refresh_server(target)
            if mcp_err:
                err(f"Failed to reload '{target}': {mcp_err}")
            else:
                ok(f"Reloaded MCP server: {target}")
        else:
            errors = reload_mcp()
            for name, e in errors.items():
                if e:
                    print(f"  {clr('✗', 'red')} {name}: {e}")
                else:
                    print(f"  {clr('✓', 'green')} {name}: connected")
        return True

    if subcmd == "add":
        # HTTP/SSE transport: /mcp add <name> --transport http <url>
        if "--transport" in parts:
            ti = parts.index("--transport")
            if ti + 2 >= len(parts):
                err("Usage: /mcp add <name> --transport http <url>")
                return True
            transport_type = parts[ti + 1].lower()
            if transport_type not in ("http", "sse"):
                err(f"Unsupported transport '{transport_type}'. Use: http, sse")
                return True
            name = parts[1]
            url = parts[ti + 2]
            add_server_to_user_config(name, {"type": transport_type, "url": url})
            ok(f"Added MCP server '{name}' ({transport_type}: {url}) → /mcp reload to connect")
            return True
        # Stdio: /mcp add <name> <command> [args...]
        if len(parts) < 3:
            err("Usage: /mcp add <name> <command> [arg1 arg2 ...]\n"
                "       /mcp add <name> --transport http <url>")
            return True
        name = parts[1]
        command = parts[2]
        cmd_args = parts[3:]
        raw = {"type": "stdio", "command": command}
        if cmd_args:
            raw["args"] = cmd_args
        add_server_to_user_config(name, raw)
        ok(f"Added MCP server '{name}' → restart or /mcp reload to connect")
        return True

    if subcmd == "remove":
        if len(parts) < 2:
            err("Usage: /mcp remove <name>")
            return True
        name = parts[1]
        removed = remove_server_from_user_config(name)
        if removed:
            ok(f"Removed MCP server '{name}' from user config")
        else:
            err(f"Server '{name}' not found in user config")
        return True

    if subcmd not in ("", "list"):
        err(f"Unknown /mcp subcommand '{subcmd}'. Use: reload, add, remove, list")
        return True

    mgr = get_mcp_manager()
    servers = mgr.list_servers()
    config_files = list_config_files()
    if config_files:
        info(f"Config files: {', '.join(str(f) for f in config_files)}")

    if not servers:
        configs = load_mcp_configs()
        if not configs:
            info("No MCP servers configured.")
            info("Add servers in ~/.cheetahclaws/mcp.json or .mcp.json")
            info("Example: /mcp add my-git uvx mcp-server-git")
        else:
            info("MCP servers configured but not yet connected. Run /mcp reload")
        return True

    info(f"MCP servers ({len(servers)}):")
    total_tools = 0
    for client in servers:
        status_color = {
            "connected":    "green",
            "connecting":   "yellow",
            "disconnected": "dim",
            "error":        "red",
        }.get(client.state.value, "dim")
        print(f"  {clr(client.status_line(), status_color)}")
        for tool in client._tools:
            import textwrap
            desc_lines = textwrap.wrap(tool.description, width=72, subsequent_indent=" " * 8)
            desc = ("\n" + " " * 8).join(desc_lines) if desc_lines else ""
            print(f"      {clr(tool.qualified_name, 'cyan')}  {desc}")
            total_tools += 1

    if total_tools:
        info(f"Total: {total_tools} MCP tool(s) available to Claude")
    return True


# ── Plugin ─────────────────────────────────────────────────────────────────

def cmd_plugin(args: str, _state, config) -> bool:
    """Manage plugins."""
    from plugin import (
        install_plugin, uninstall_plugin, enable_plugin, disable_plugin,
        disable_all_plugins, update_plugin, list_plugins, get_plugin,
        PluginScope, recommend_plugins, format_recommendations,
    )

    parts = args.split(None, 1)
    subcmd = parts[0].lower() if parts else ""
    rest   = parts[1].strip() if len(parts) > 1 else ""

    if not subcmd:
        plugins = list_plugins()
        if not plugins:
            info("No plugins installed.")
            info("Install: /plugin install name@git_url")
            info("Recommend: /plugin recommend")
            return True
        info(f"Installed plugins ({len(plugins)}):")
        for p in plugins:
            state_color = "green" if p.enabled else "dim"
            state_str   = "enabled" if p.enabled else "disabled"
            desc = p.manifest.description if p.manifest else ""
            print(f"  {clr(p.name, state_color)} [{p.scope.value}] {state_str}  {desc[:60]}")
        return True

    if subcmd == "install":
        if not rest:
            err("Usage: /plugin install name@git_url")
            return True
        scope_str = "user"
        if " --project" in rest:
            scope_str = "project"
            rest = rest.replace("--project", "").strip()
        scope = PluginScope(scope_str)
        success, msg = install_plugin(rest, scope=scope)
        (ok if success else err)(msg)
        return True

    if subcmd == "uninstall":
        if not rest:
            err("Usage: /plugin uninstall name")
            return True
        success, msg = uninstall_plugin(rest)
        (ok if success else err)(msg)
        return True

    if subcmd == "enable":
        if not rest:
            err("Usage: /plugin enable name")
            return True
        success, msg = enable_plugin(rest)
        (ok if success else err)(msg)
        return True

    if subcmd == "disable":
        if not rest:
            err("Usage: /plugin disable name")
            return True
        success, msg = disable_plugin(rest)
        (ok if success else err)(msg)
        return True

    if subcmd == "disable-all":
        success, msg = disable_all_plugins()
        (ok if success else err)(msg)
        return True

    if subcmd == "update":
        if not rest:
            err("Usage: /plugin update name")
            return True
        success, msg = update_plugin(rest)
        (ok if success else err)(msg)
        return True

    if subcmd == "recommend":
        context = rest
        if not context:
            from plugin.recommend import recommend_from_files
            files = list(Path.cwd().glob("**/*"))[:200]
            recs = recommend_from_files(files)
        else:
            recs = recommend_plugins(context)
        print(format_recommendations(recs))
        return True

    if subcmd == "info":
        if not rest:
            err("Usage: /plugin info name")
            return True
        entry = get_plugin(rest)
        if entry is None:
            err(f"Plugin '{rest}' not found.")
            return True
        m = entry.manifest
        print(f"Name:    {entry.name}")
        print(f"Scope:   {entry.scope.value}")
        print(f"Source:  {entry.source}")
        print(f"Dir:     {entry.install_dir}")
        print(f"Enabled: {entry.enabled}")
        if m:
            print(f"Version: {m.version}")
            print(f"Author:  {m.author}")
            print(f"Desc:    {m.description}")
            if m.tags:
                print(f"Tags:    {', '.join(m.tags)}")
            if m.tools:
                print(f"Tools:   {', '.join(m.tools)}")
            if m.skills:
                print(f"Skills:  {', '.join(m.skills)}")
        return True

    err(f"Unknown plugin subcommand: {subcmd}  (try /plugin or /help)")
    return True


# ── Tasks ──────────────────────────────────────────────────────────────────

def cmd_tasks(args: str, _state, config) -> bool:
    """Show and manage tasks."""
    from task import list_tasks, get_task, create_task, update_task, delete_task, clear_all_tasks
    from task.types import TaskStatus

    parts = args.split(None, 1)
    subcmd = parts[0].lower() if parts else ""
    rest   = parts[1].strip() if len(parts) > 1 else ""

    STATUS_MAP = {
        "done":   "completed",
        "start":  "in_progress",
        "cancel": "cancelled",
    }

    if not subcmd:
        tasks = list_tasks()
        if not tasks:
            info("No tasks. Use TaskCreate tool or /tasks create <subject>.")
            return True
        resolved = {t.id for t in tasks if t.status == TaskStatus.COMPLETED}
        total = len(tasks)
        done  = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
        info(f"Tasks ({done}/{total} completed):")
        for t in tasks:
            pending_blockers = [b for b in t.blocked_by if b not in resolved]
            owner_str   = f" {clr(f'({t.owner})', 'dim')}" if t.owner else ""
            blocked_str = clr(f" [blocked by #{', #'.join(pending_blockers)}]", "yellow") if pending_blockers else ""
            status_color = {
                TaskStatus.PENDING:     "dim",
                TaskStatus.IN_PROGRESS: "cyan",
                TaskStatus.COMPLETED:   "green",
                TaskStatus.CANCELLED:   "red",
            }.get(t.status, "dim")
            icon = t.status_icon()
            print(f"  #{t.id} {clr(icon + ' ' + t.status.value, status_color)} {t.subject}{owner_str}{blocked_str}")
        return True

    if subcmd == "create":
        if not rest:
            err("Usage: /tasks create <subject>")
            return True
        t = create_task(rest, description="(created via REPL)")
        ok(f"Task #{t.id} created: {t.subject}")
        return True

    if subcmd in STATUS_MAP:
        new_status = STATUS_MAP[subcmd]
        if not rest:
            err(f"Usage: /tasks {subcmd} <task_id>")
            return True
        task, fields = update_task(rest, status=new_status)
        if task is None:
            err(f"Task #{rest} not found.")
        else:
            ok(f"Task #{task.id} → {new_status}: {task.subject}")
        return True

    if subcmd == "delete":
        if not rest:
            err("Usage: /tasks delete <task_id>")
            return True
        removed = delete_task(rest)
        if removed:
            ok(f"Task #{rest} deleted.")
        else:
            err(f"Task #{rest} not found.")
        return True

    if subcmd == "get":
        if not rest:
            err("Usage: /tasks get <task_id>")
            return True
        t = get_task(rest)
        if t is None:
            err(f"Task #{rest} not found.")
            return True
        print(f"  #{t.id} [{t.status.value}] {t.subject}")
        print(f"  Description: {t.description}")
        if t.owner:         print(f"  Owner:       {t.owner}")
        if t.active_form:   print(f"  Active form: {t.active_form}")
        if t.blocked_by:    print(f"  Blocked by:  #{', #'.join(t.blocked_by)}")
        if t.blocks:        print(f"  Blocks:      #{', #'.join(t.blocks)}")
        if t.metadata:      print(f"  Metadata:    {t.metadata}")
        print(f"  Created: {t.created_at[:19]}  Updated: {t.updated_at[:19]}")
        return True

    if subcmd == "clear":
        clear_all_tasks()
        ok("All tasks deleted.")
        return True

    err(f"Unknown tasks subcommand: {subcmd}  (try /tasks or /help)")
    return True
