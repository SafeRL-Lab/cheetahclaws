"""
monitor/summarizer.py — AI-powered report generation.

Uses providers.stream() to summarize raw fetched data into a concise,
actionable report. Falls back to plain formatting if AI is unavailable.
"""
from __future__ import annotations

import json
from datetime import datetime


def _build_prompt(raw: dict) -> str:
    topic = raw["topic"]
    source = raw["source"]
    items = raw.get("items", [])
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [f"Date/Time: {now}", f"Topic: {topic}", f"Source: {source}", ""]

    for i, item in enumerate(items[:20], 1):
        title = item.get("title", "")
        url = item.get("url", "")
        summary = item.get("summary", "")
        date = item.get("date", "")
        lines.append(f"[{i}] {title}")
        if date:
            lines.append(f"    Date: {date}")
        if url:
            lines.append(f"    URL: {url}")
        if summary:
            lines.append(f"    {summary}")
        lines.append("")

    return "\n".join(lines)


def _system_prompt_for(topic: str) -> str:
    if topic == "ai_research":
        return (
            "You are an AI research analyst. Given a list of recent arxiv papers, "
            "write a concise daily briefing (under 400 words). "
            "Highlight 3-5 most significant papers, explain why they matter, "
            "identify emerging trends, and give a 1-sentence actionable takeaway. "
            "Use clear section headers. Be direct and informative."
        )
    if topic.startswith("stock_"):
        ticker = topic[6:]
        return (
            f"You are a financial analyst monitoring {ticker}. "
            "Given current price data and recent history, write a brief market update (under 200 words). "
            "State current price clearly, assess momentum (up/down/sideways), "
            "note any notable moves, and give a clear sentiment (bullish/neutral/bearish) with brief reasoning. "
            "Keep it factual and concise."
        )
    if topic.startswith("crypto_"):
        symbol = topic[7:]
        return (
            f"You are a crypto market analyst monitoring {symbol}. "
            "Given the price data, write a brief market update (under 200 words). "
            "State the price clearly, assess 24h and 7d momentum, "
            "give market sentiment (bullish/neutral/bearish), and one key observation. "
            "Be concise and direct."
        )
    if topic == "world_news":
        return (
            "You are a world news analyst. Given today's top headlines, "
            "write a brief world news digest (under 400 words). "
            "Group related stories, highlight the 3-4 most significant events, "
            "and give a 2-sentence overall world situation summary. "
            "Use clear headers. Be objective and informative."
        )
    return (
        "You are a research analyst. Summarize the following data into a concise, "
        "actionable briefing (under 300 words). Highlight key findings and trends. "
        "End with one clear recommendation or insight."
    )


def summarize(raw: dict, config: dict) -> str:
    """Generate an AI summary of raw fetched data. Returns report string."""
    if not raw.get("items"):
        err = raw.get("error", "No data available")
        return f"[{raw['topic']}] No data to summarize. Error: {err}"

    topic = raw["topic"]
    data_text = _build_prompt(raw)

    # Try AI summarization
    try:
        from cheetahclaws.providers import stream, AssistantTurn, TextChunk
        system = _system_prompt_for(topic)
        messages = [{"role": "user", "content": data_text}]

        chunks = []
        for event in stream(config["model"], system, messages, [], config):
            if isinstance(event, TextChunk):
                chunks.append(event.text)
            elif isinstance(event, AssistantTurn):
                break

        ai_text = "".join(chunks).strip()
        if ai_text:
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            header = f"[{topic}] Monitor Report — {now}\n{'='*50}\n"
            return header + ai_text
    except Exception:
        pass

    # Fallback: plain text formatting
    return _plain_format(raw)


def _plain_format(raw: dict) -> str:
    """Plain text report when AI is unavailable."""
    topic = raw["topic"]
    source = raw["source"]
    items = raw.get("items", [])
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"[{topic}] Monitor Report — {now}",
        "=" * 50,
        f"Source: {source}",
        "",
    ]
    for i, item in enumerate(items[:10], 1):
        lines.append(f"{i}. {item.get('title', 'N/A')}")
        if item.get("summary"):
            lines.append(f"   {item['summary'][:150]}")
        if item.get("url"):
            lines.append(f"   {item['url']}")
        lines.append("")

    return "\n".join(lines)
