"""Entity extraction — mine frequent models / benchmarks / orgs / people
from result titles and snippets.

Purpose: answer "what is everyone talking about right now" for this topic
by showing a ranked table of the most-mentioned named entities across
all pulled sources. Works offline, no LLM call.

Categories:
    models      — GPT-5, Claude-Opus-5, Llama-4, GLM-5.1, Qwen-3, DeepSeek-V3…
    benchmarks  — MMLU, GSM8K, HumanEval, HumanEval+, MATH, MMMU, SWE-bench, …
    orgs        — OpenAI, Anthropic, Google DeepMind, Meta, xAI, NVIDIA, …
    people      — from academic results' `author` field (safe; no free-text NER)

All patterns are literal-token based with word-boundary anchoring. Case-
insensitive match, case-preserved display. Counts are frequency across
the (title + snippet) haystack of each result, capped at 1 per result
per entity to avoid one spammy abstract dominating.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .types import Result


# ─── Curated entity vocab (updated Apr 2026) ───────────────────────────────

_MODEL_FAMILIES = [
    # Anthropic
    r"Claude(?:[\s\-](?:Opus|Sonnet|Haiku))?[\s\-]?\d+(?:\.\d+)?",
    # OpenAI
    r"GPT[\s\-]?\d+(?:\.\d+)?(?:[\s\-](?:Turbo|Mini|Nano|o|Ultra))?",
    r"o\d+(?:[\s\-]Mini|[\s\-]Pro)?",                 # o1, o3-mini, o4-pro
    # Google
    r"Gemini(?:[\s\-](?:Flash|Pro|Ultra|Nano))?[\s\-]?\d+(?:\.\d+)?",
    r"Gemma[\s\-]?\d+(?:\.\d+)?",
    r"PaLM[\s\-]?\d+(?:\.\d+)?",
    # Meta — both `Llama` and `LLaMA` covered via case-insensitive flag
    r"Llama[\s\-]?\d+(?:\.\d+)?(?:[\s\-]?(?:\d+B|Instruct|Chat))?",
    # xAI
    r"Grok[\s\-]?\d+(?:\.\d+)?",
    # Mistral
    r"Mistral[\s\-]?\d*(?:\.\d+)?",
    r"Mixtral[\s\-]?(?:\d+x\d+B)?",
    # DeepSeek
    r"DeepSeek(?:[\s\-](?:V|R|Coder|Chat|Math))?[\s\-]?\d+(?:\.\d+)?",
    # Alibaba
    r"Qwen[\s\-]?\d+(?:\.\d+)?(?:[\s\-](?:VL|Coder|Math|Omni))?",
    r"QwQ[\s\-]?\d*(?:\.\d+)?",
    # Zhipu
    r"GLM[\s\-]?\d+(?:\.\d+)?",
    r"ChatGLM[\s\-]?\d*",
    # Moonshot
    r"Moonshot[\s\-]?v?\d+(?:\.\d+)?",
    r"Kimi(?:[\s\-](?:Latest|K1|K2))?",
    # Microsoft
    r"Phi[\s\-]?\d+(?:\.\d+)?",
    # Others
    r"Yi[\s\-]?\d+(?:\.\d+)?",           # 01.AI Yi
    r"Baichuan[\s\-]?\d+(?:\.\d+)?",
    r"MiniMax[\s\-]?(?:Text|abab)?[\s\-]?\d*(?:\.\d+)?",
    r"Nova[\s\-]?(?:Pro|Lite|Micro)?",    # Amazon Nova
    r"Command[\s\-]?R\+?",                # Cohere
    r"OLMo[\s\-]?\d*(?:\.\d+)?",          # AI2
    r"Falcon[\s\-]?\d+(?:\.\d+)?",        # TII
    r"StableLM[\s\-]?\d*",
    r"WizardLM[\s\-]?\d*",
    r"Vicuna[\s\-]?\d*",
    r"Alpaca",
]

_BENCHMARKS = [
    # Classic LM eval
    "MMLU", "MMLU-Pro", "MMLU-Redux", "BBH", "HellaSwag", "ARC", "AGIEval",
    "GSM8K", "MATH", "MATH-500", "AIME", "OlympiadBench",
    "HumanEval", "HumanEval+", "MBPP", "MBPP+", "BigCodeBench", "APPS",
    "SWE-bench", "SWE-bench Verified", "LiveCodeBench", "CRUX",
    "TruthfulQA", "MuSR", "DROP", "TriviaQA", "NaturalQuestions",
    # Multimodal
    "MMMU", "MMMU-Pro", "MathVista", "MathVerse", "ChartQA", "DocVQA",
    "VQAv2", "OKVQA", "MMBench", "ScienceQA", "MME", "SEED-Bench",
    "ViCo", "PaperBench", "WebArena", "WebCompass",
    # Agent
    "GAIA", "AgentBench", "WebArena", "VisualWebArena", "OSWorld",
    "Cybench", "AgentSafetyBench",
    # Safety / alignment
    "HarmBench", "MT-Bench", "Chatbot Arena", "LMSys Arena", "Arena Hard",
    "Do-Not-Answer", "AdvBench", "RealToxicityPrompts",
    # Chinese
    "C-Eval", "CMMLU", "GaoKao-Bench", "CLUE", "SuperCLUE",
    # Long context
    "RULER", "LongBench", "Needle-in-a-Haystack", "InfiniteBench",
    # Reasoning / frontier
    "FrontierMath", "ARC-AGI", "ARC-AGI-2", "HLE",  # Humanity's Last Exam
    "GPQA", "GPQA-Diamond",
]

_ORGS = [
    "OpenAI", "Anthropic", "Google DeepMind", "DeepMind", "Google Research",
    "Meta AI", "Meta", "FAIR", "Microsoft Research", "Microsoft",
    "xAI", "Mistral AI", "Mistral", "Cohere", "Stability AI",
    "DeepSeek", "Moonshot AI", "Moonshot", "Alibaba", "Alibaba DAMO",
    "Qwen Team", "Baidu", "Tencent", "ByteDance", "Zhipu AI", "Zhipu",
    "Hugging Face", "HuggingFace", "Together AI", "Groq", "Perplexity",
    "NVIDIA", "AMD", "Intel", "Apple", "Tesla", "Amazon", "AWS",
    "Databricks", "MosaicML", "01.AI", "AI2", "Allen Institute",
    "Salesforce", "IBM", "SenseTime", "MiniMax",
    "Stanford", "MIT", "Berkeley", "CMU", "Tsinghua", "Peking University",
    "Princeton", "Cambridge", "Oxford", "Mila", "Vector Institute",
]

# Compile
_MODEL_RE = re.compile(
    r"\b(?:" + "|".join(_MODEL_FAMILIES) + r")\b",
    re.IGNORECASE,
)
_BENCH_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(b) for b in _BENCHMARKS) + r")\b",
    re.IGNORECASE,
)
_ORG_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(o) for o in _ORGS) + r")\b",
)


@dataclass
class Entities:
    models:     list[tuple[str, int]] = field(default_factory=list)
    benchmarks: list[tuple[str, int]] = field(default_factory=list)
    orgs:       list[tuple[str, int]] = field(default_factory=list)
    people:     list[tuple[str, int]] = field(default_factory=list)


def extract(results: list[Result]) -> Entities:
    """Scan all results once, returning ranked (name, count) tuples per category."""
    model_c: Counter = Counter()
    bench_c: Counter = Counter()
    org_c:   Counter = Counter()
    people_c: Counter = Counter()

    for r in results:
        # Haystack: title + snippet. Dedupe per-result so one spammy
        # abstract doesn't dominate the count for any single entity.
        hay = f"{r.title}  {r.snippet}"
        for m in set(_normalize(x) for x in _MODEL_RE.findall(hay)):
            if m:
                model_c[m] += 1
        for m in set(x.upper() if x.isupper() or len(x) <= 5 else x
                     for x in _BENCH_RE.findall(hay)):
            if m:
                bench_c[m] += 1
        for m in set(_ORG_RE.findall(hay)):
            if m:
                org_c[m] += 1

        # People: use author field (from academic results); split on commas
        if r.author:
            # Skip obviously non-human fields (usernames, emails)
            raw = r.author
            if raw.startswith("@") or "@" in raw.split(",")[0]:
                continue
            for name in raw.split(","):
                n = name.strip().split("+")[0].strip()  # "+2 more" suffix
                if 3 <= len(n) <= 50 and n.count(" ") <= 4:
                    people_c[n] += 1

    return Entities(
        models=model_c.most_common(12),
        benchmarks=bench_c.most_common(10),
        orgs=org_c.most_common(10),
        people=[(n, c) for n, c in people_c.most_common(10) if c >= 2],
    )


def _normalize(match: str) -> str:
    """Canonicalize whitespace/case in a matched model name for merging."""
    m = re.sub(r"\s+", "-", match.strip())
    # Title-case major families while keeping all-caps ones
    parts = m.split("-")
    out = []
    for p in parts:
        if p.upper() in ("GPT", "LLM", "GLM", "MMLU"):
            out.append(p.upper())
        elif p.lower() in ("mini", "nano", "opus", "sonnet", "haiku", "turbo",
                           "pro", "ultra", "flash"):
            out.append(p.title())
        elif p.isdigit() or re.match(r"\d+(\.\d+)?$", p):
            out.append(p)
        else:
            out.append(p[:1].upper() + p[1:])
    return "-".join(out)


def render_entities_table(e: Entities, title_prefix: str = "") -> str:
    """Render as a compact markdown block with 4 subsections.

    Returns empty string if no entities of any category were found (so the
    caller can skip the whole section cleanly).
    """
    if not (e.models or e.benchmarks or e.orgs or e.people):
        return ""

    pfx = f"{title_prefix} " if title_prefix else ""
    out = [f"## {pfx}Top mentioned entities",
           ""]

    def _col(name: str, items: list[tuple[str, int]]) -> list[str]:
        if not items:
            return [f"**{name}**: —", ""]
        parts = [f"**{name}** ({len(items)}):"]
        parts.append(" · ".join(f"{n} ×{c}" for n, c in items))
        parts.append("")
        return parts

    out += _col("Models", e.models)
    out += _col("Benchmarks", e.benchmarks)
    out += _col("Orgs / Labs", e.orgs)
    out += _col("People", e.people)
    return "\n".join(out).strip()
