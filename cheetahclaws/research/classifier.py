"""Heuristic topic → domain classifier.

Fast, offline, zero-cost. Keyword matching with weighted signals. If no
strong signal, returns a broad set so the fan-out still covers the topic.

Designed to be good enough that users rarely need --domain; explicit
--domain always overrides.
"""
from __future__ import annotations

import re

from .types import Domain

_ACADEMIC_KEYWORDS = {
    "paper", "papers", "arxiv", "preprint", "citation", "benchmark",
    "dataset", "abstract", "conference", "proceedings", "thesis",
    "study", "experiment", "methodology", "algorithm", "theorem",
    "proof", "neurips", "icml", "iclr", "cvpr", "acl", "emnlp", "siggraph",
    "ablation", "state-of-the-art", "sota", "transformer", "diffusion",
    "reinforcement learning", "self-supervised", "few-shot", "zero-shot",
    "fine-tuning", "pretraining", "embedding", "attention mechanism",
}

_TECH_KEYWORDS = {
    "api", "sdk", "cli", "library", "framework", "compiler", "runtime",
    "kubernetes", "docker", "rust", "python", "typescript", "react",
    "nextjs", "webpack", "vite", "postgres", "redis", "kafka", "grpc",
    "github", "pull request", "pr ", "commit", "bug", "regression",
    "memory leak", "race condition", "latency", "throughput", "benchmark",
    "observability", "prometheus", "opentelemetry", "tracing",
    "microservice", "monorepo", "build system", "ci/cd", "pipeline",
    "rate limit", "webhook", "oauth", "tls", "ssl", "http2", "http3",
    "llm", "rag", "vector db", "embedding", "agent",
}

_FINANCE_KEYWORDS = {
    "stock", "stocks", "earnings", "revenue", "ipo", "merger", "acquisition",
    "hedge fund", "etf", "bond", "yield", "fed", "fomc", "interest rate",
    "inflation", "cpi", "ppi", "gdp", "recession", "bull", "bear",
    "ticker", "nasdaq", "s&p", "dow", "crypto", "bitcoin", "btc", "ethereum",
    "eth", "defi", "nft", "tokenomics", "sec filing", "10-k", "10-q",
    "8-k", "13f", "prospectus", "valuation", "p/e", "eps", "market cap",
    "polymarket", "prediction market", "odds",
}

_NEWS_KEYWORDS = {
    "news", "today", "this week", "breaking", "announced", "launches",
    "launched", "reaction", "opinion", "analysis", "explainer",
    "yesterday", "recent", "latest",
}

_SOCIAL_KEYWORDS = {
    "reddit", "twitter", "x.com", "hackernews", "hn ", "community",
    "discussion", "thread", "meme", "viral", "trending", "upvotes",
    "controversy", "debate", "consensus",
}

_TICKER_RE = re.compile(r"\b([A-Z]{2,5})\b")
_CRYPTO_RE = re.compile(r"\b(BTC|ETH|SOL|XRP|DOGE|ADA|AVAX|DOT|LINK|MATIC)\b", re.I)


def classify(topic: str) -> list[Domain]:
    """Return a list of domains, highest-signal first.

    Returns at most 3 domains. Always returns at least one — falls back
    to ['web', 'news'] when nothing matches.
    """
    t = topic.lower().strip()
    if not t:
        return ["web"]

    scores: dict[Domain, int] = {
        "academic": _keyword_hits(t, _ACADEMIC_KEYWORDS),
        "tech":     _keyword_hits(t, _TECH_KEYWORDS),
        "finance":  _keyword_hits(t, _FINANCE_KEYWORDS),
        "news":     _keyword_hits(t, _NEWS_KEYWORDS),
        "social":   _keyword_hits(t, _SOCIAL_KEYWORDS),
    }

    if _CRYPTO_RE.search(topic):
        scores["finance"] += 3
    # Uppercase ticker-like tokens in the ORIGINAL topic
    tickers = [m for m in _TICKER_RE.findall(topic)
               if m not in ("AI", "ML", "API", "SDK", "CPU", "GPU", "LLM", "RAG",
                            "URL", "URI", "JSON", "YAML", "HTML", "CSS", "OS")]
    if tickers:
        scores["finance"] += 2

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top = [d for d, s in ranked if s > 0]

    if not top:
        return ["web", "news"]

    if "academic" in top[:2]:
        _push_unique(top, "news")
    if "finance" in top[:2]:
        _push_unique(top, "news")
    if "tech" in top[:2] and "social" not in top:
        _push_unique(top, "social")

    return top[:3]


def _keyword_hits(text: str, bag: set[str]) -> int:
    return sum(1 for kw in bag if kw in text)


def _push_unique(lst: list[Domain], item: Domain) -> None:
    if item not in lst:
        lst.append(item)
