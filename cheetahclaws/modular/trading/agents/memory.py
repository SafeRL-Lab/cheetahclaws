"""
BM25-based financial memory system.

Inspired by TradingAgents' FinancialSituationMemory. Enables agents to
learn from past trading decisions without API calls or token limits.

Each memory stores a (situation, recommendation, outcome) tuple.
Retrieval uses BM25 similarity matching to find relevant past decisions.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class MemoryEntry:
    """A single memory: situation + recommendation + outcome."""
    situation: str      # market context when decision was made
    recommendation: str # what was decided (BUY/SELL/HOLD + reasoning)
    outcome: str = ""   # what happened (profit/loss, was decision correct?)
    date: str = ""      # when the decision was made
    symbol: str = ""    # which instrument


class TradingMemory:
    """BM25-based memory for trading decisions.

    Uses rank_bm25 if available, falls back to simple TF matching.
    Persists memories to JSON file for cross-session learning.
    """

    def __init__(self, memory_file: str | Path | None = None):
        self.entries: list[MemoryEntry] = []
        self._index = None
        self._tokenized: list[list[str]] = []
        self._memory_file = Path(memory_file) if memory_file else None
        if self._memory_file and self._memory_file.exists():
            self._load()

    def add(
        self,
        situation: str,
        recommendation: str,
        outcome: str = "",
        date: str = "",
        symbol: str = "",
    ) -> None:
        """Add a new memory entry and rebuild index."""
        self.entries.append(MemoryEntry(
            situation=situation,
            recommendation=recommendation,
            outcome=outcome,
            date=date,
            symbol=symbol,
        ))
        self._rebuild_index()
        if self._memory_file:
            self._save()

    def add_batch(self, entries: list[tuple[str, str, str]]) -> None:
        """Add multiple (situation, recommendation, outcome) tuples."""
        for sit, rec, out in entries:
            self.entries.append(MemoryEntry(situation=sit, recommendation=rec, outcome=out))
        self._rebuild_index()
        if self._memory_file:
            self._save()

    def get_memories(
        self,
        current_situation: str,
        n_matches: int = 3,
        symbol: str | None = None,
    ) -> list[dict]:
        """Retrieve most relevant past memories for current situation.

        Args:
            current_situation: description of current market context
            n_matches: maximum number of results
            symbol: optional filter by symbol

        Returns:
            List of {"situation", "recommendation", "outcome", "similarity"} dicts
        """
        candidates = self.entries
        if symbol:
            symbol_candidates = [e for e in candidates if e.symbol == symbol]
            if symbol_candidates:
                candidates = symbol_candidates

        if not candidates:
            return []

        query_tokens = _tokenize(current_situation)
        if not query_tokens:
            return []

        # Try BM25 first
        try:
            return self._bm25_search(candidates, query_tokens, n_matches)
        except Exception:
            pass

        # Fallback: simple term-frequency matching
        return self._simple_search(candidates, query_tokens, n_matches)

    def _bm25_search(
        self, candidates: list[MemoryEntry], query_tokens: list[str], n: int
    ) -> list[dict]:
        """Search using BM25 algorithm (requires rank_bm25)."""
        from rank_bm25 import BM25Okapi  # type: ignore

        corpus = [_tokenize(e.situation + " " + e.recommendation) for e in candidates]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query_tokens)

        # Sort by score descending
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in ranked[:n]:
            if score <= 0:
                break
            e = candidates[idx]
            max_score = max(scores) if max(scores) > 0 else 1
            results.append({
                "situation": e.situation,
                "recommendation": e.recommendation,
                "outcome": e.outcome,
                "date": e.date,
                "symbol": e.symbol,
                "similarity": round(score / max_score, 3),
            })
        return results

    def _simple_search(
        self, candidates: list[MemoryEntry], query_tokens: list[str], n: int
    ) -> list[dict]:
        """Fallback: simple term overlap scoring."""
        query_set = set(query_tokens)
        scored = []
        for e in candidates:
            doc_tokens = set(_tokenize(e.situation + " " + e.recommendation))
            overlap = len(query_set & doc_tokens)
            total = len(query_set | doc_tokens)
            score = overlap / total if total > 0 else 0
            scored.append((e, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for e, score in scored[:n]:
            if score <= 0:
                break
            results.append({
                "situation": e.situation,
                "recommendation": e.recommendation,
                "outcome": e.outcome,
                "date": e.date,
                "symbol": e.symbol,
                "similarity": round(score, 3),
            })
        return results

    def _rebuild_index(self) -> None:
        """Rebuild the tokenized corpus."""
        self._tokenized = [
            _tokenize(e.situation + " " + e.recommendation)
            for e in self.entries
        ]

    def _save(self) -> None:
        """Persist memories to JSON file."""
        if not self._memory_file:
            return
        self._memory_file.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "situation": e.situation,
                "recommendation": e.recommendation,
                "outcome": e.outcome,
                "date": e.date,
                "symbol": e.symbol,
            }
            for e in self.entries
        ]
        self._memory_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def _load(self) -> None:
        """Load memories from JSON file."""
        try:
            data = json.loads(self._memory_file.read_text())
            for item in data:
                self.entries.append(MemoryEntry(
                    situation=item.get("situation", ""),
                    recommendation=item.get("recommendation", ""),
                    outcome=item.get("outcome", ""),
                    date=item.get("date", ""),
                    symbol=item.get("symbol", ""),
                ))
            self._rebuild_index()
        except Exception:
            pass

    def __len__(self) -> int:
        return len(self.entries)

    def clear(self) -> None:
        """Clear all memories."""
        self.entries.clear()
        self._tokenized.clear()
        self._index = None
        if self._memory_file and self._memory_file.exists():
            self._memory_file.unlink()


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer with lowercasing."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


# ── Memory Manager (singleton-like) ───────────────────────────────────────

_MEMORY_DIR = Path.home() / ".cheetahclaws" / "trading" / "memory"


def get_memory(component: str) -> TradingMemory:
    """Get or create a memory store for a specific agent component.

    Components: bull_researcher, bear_researcher, trader, risk_judge, portfolio_manager
    """
    return TradingMemory(_MEMORY_DIR / f"{component}.json")


def get_all_memories() -> dict[str, TradingMemory]:
    """Load all component memories."""
    components = [
        "bull_researcher", "bear_researcher", "trader",
        "risk_judge", "portfolio_manager",
    ]
    return {c: get_memory(c) for c in components}
