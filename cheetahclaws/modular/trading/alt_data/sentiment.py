"""
sentiment.py — LLM-based news sentiment for the analyze prompt.

We don't ship a transformer model. Instead we ride on cheetahclaws's
existing auxiliary-model infrastructure: the trading agent calls a
cheap model (e.g. gpt-5-nano, claude-haiku) with a tight 5-line prompt
asking for a -10..+10 score per headline, then aggregates.

Soft-fail to empty string if the auxiliary model isn't reachable or if
yfinance returned no headlines.
"""
from __future__ import annotations

import json
import re
from typing import Any


def fetch_recent_headlines(symbol: str, max_items: int = 8) -> list[dict[str, Any]]:
    """Pull recent news headlines from yfinance. Soft-fail to []."""
    try:
        import yfinance as yf
    except ImportError:
        return []
    try:
        items = yf.Ticker(symbol).news or []
    except Exception:
        return []

    out: list[dict[str, Any]] = []
    for it in items[:max_items]:
        if not isinstance(it, dict):
            continue
        # yfinance shape changed over versions — try both.
        title = it.get("title") or it.get("content", {}).get("title")
        publisher = it.get("publisher") or it.get("content", {}).get("provider", {}).get("displayName")
        link = it.get("link") or it.get("content", {}).get("canonicalUrl", {}).get("url")
        ts = it.get("providerPublishTime") or it.get("content", {}).get("pubDate")
        if title:
            out.append({"title": title, "publisher": publisher or "", "link": link or "", "ts": ts})
    return out


def _score_with_aux_model(symbol: str, headlines: list[dict[str, Any]]) -> dict[str, Any]:
    """Ask the auxiliary model for sentiment scores. Returns {} on failure."""
    if not headlines:
        return {}

    # Build a tight prompt — small models choke on long ones.
    items = [f"{i+1}. {h['title']}" for i, h in enumerate(headlines)]
    prompt = (
        f"Score the sentiment of each {symbol} headline below as an integer "
        f"from -10 (very bearish) to +10 (very bullish). 0 = neutral.\n\n"
        + "\n".join(items)
        + "\n\nRespond ONLY with valid JSON: "
        '{"scores": [-3, 5, ...], "reasoning": "one sentence overall"}'
    )

    try:
        from cheetahclaws.auxiliary import stream_auxiliary
    except ImportError:
        return {}

    try:
        raw = stream_auxiliary(
            system="You are a financial analyst. Respond ONLY with valid JSON.",
            messages=[{"role": "user", "content": prompt}],
            config={},
        )
    except Exception:
        return {}
    if not raw:
        return {}

    # Extract JSON even if the model wrapped it in markdown fences.
    m = re.search(r"\{[\s\S]*?\}", raw)
    if not m:
        return {}
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}

    scores = parsed.get("scores", [])
    if not isinstance(scores, list) or len(scores) != len(headlines):
        return {}

    # Clamp + coerce
    clean: list[int] = []
    for s in scores:
        try:
            clean.append(max(-10, min(10, int(s))))
        except (TypeError, ValueError):
            clean.append(0)

    return {
        "scores": clean,
        "mean":   sum(clean) / len(clean) if clean else 0.0,
        "n_pos":  sum(1 for s in clean if s > 1),
        "n_neg":  sum(1 for s in clean if s < -1),
        "reasoning": parsed.get("reasoning", "")[:200],
    }


def render_sentiment_block(symbol: str, max_items: int = 8) -> str:
    """Markdown block summarising recent news sentiment. Empty if unavailable."""
    headlines = fetch_recent_headlines(symbol, max_items=max_items)
    if not headlines:
        return ""

    scored = _score_with_aux_model(symbol, headlines)

    lines = [f"## News Sentiment ({symbol})"]
    if scored:
        mean = scored["mean"]
        regime = ("BULLISH" if mean > 2 else
                  "BEARISH" if mean < -2 else "MIXED")
        lines.append(f"- Headlines analysed: {len(headlines)}")
        lines.append(
            f"- Aggregate score: **{mean:+.1f}/10** → **{regime}** "
            f"({scored['n_pos']} bullish, {scored['n_neg']} bearish)"
        )
        if scored.get("reasoning"):
            lines.append(f"- Auxiliary model read: _{scored['reasoning']}_")

    lines.append("- Headlines:")
    score_iter = iter(scored.get("scores", []))
    for h in headlines:
        score = next(score_iter, None)
        score_str = f" `[{score:+d}]`" if score is not None else ""
        pub = f" ({h['publisher']})" if h["publisher"] else ""
        lines.append(f"  - {h['title']}{pub}{score_str}")
    return "\n".join(lines)
