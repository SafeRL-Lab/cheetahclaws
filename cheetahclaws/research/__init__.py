"""CheetahClaws Research — multi-source topic research with engagement scoring.

Public API:
    from cheetahclaws.research import research, Brief, SourceStatus

    brief = research(
        topic="transformer inference efficiency",
        domains=["academic", "tech"],     # optional; classifier picks if omitted
        sources=None,                     # optional explicit source list
        limit=30,                         # max results per source
        use_cache=True,
        synthesize=True,                  # run LLM synthesis
        config=None,                      # cheetahclaws config (for model + API keys)
    )

Sources are organized under research.sources.* and each expose a
`search(query: str, limit: int) -> list[Result]` function. The aggregator
fans out to sources in parallel, dedupes by URL, ranks by engagement, and
optionally asks the current model to synthesize a brief.
"""
from __future__ import annotations

from .types import Result, Brief, SourceStatus, Domain
from .time_range import TimeRange, parse_range, build as build_time_range
from .aggregator import research, compare

__all__ = ["research", "compare", "Result", "Brief", "SourceStatus", "Domain",
           "TimeRange", "parse_range", "build_time_range"]
