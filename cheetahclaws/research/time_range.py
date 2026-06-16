"""TimeRange — parse user-specified time windows and convert to each source's
native date filter.

Supported inputs:
    Preset tokens: 1d, 3d, 7d, 14d, 30d, 60d, 90d, 6m, 1y, 2y, 5y, all
                   1day, 3days, 1week, 1month, 6months, 1year (natural)
    ISO dates:     2024-01-01  (date only — time defaults to midnight UTC)
    ISO datetimes: 2024-01-01T00:00:00Z

A TimeRange has `since` and `until`; either may be None. A nil range
(both None) means "all time" — sources SHOULD NOT apply any date filter.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass
class TimeRange:
    since: datetime | None = None
    until: datetime | None = None
    label: str = ""   # human-readable, e.g. "last 30 days"

    @property
    def is_bounded(self) -> bool:
        return self.since is not None or self.until is not None

    def to_iso_date(self, which: str) -> str | None:
        dt = self.since if which == "since" else self.until
        return dt.strftime("%Y-%m-%d") if dt else None

    def to_iso_datetime(self, which: str) -> str | None:
        dt = self.since if which == "since" else self.until
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ") if dt else None

    def to_unix_ts(self, which: str) -> int | None:
        dt = self.since if which == "since" else self.until
        return int(dt.timestamp()) if dt else None


_PRESET_DAYS: dict[str, int] = {
    "1d": 1, "3d": 3, "7d": 7, "14d": 14,
    "30d": 30, "60d": 60, "90d": 90, "180d": 180,
    "6m": 180, "1m": 30, "3m": 90,
    "1y": 365, "2y": 730, "5y": 1825,
}

_NATURAL_RE = re.compile(
    r"^\s*(\d+)\s*(day|days|week|weeks|month|months|year|years)\s*$",
    re.IGNORECASE,
)


def parse_range(token: str) -> TimeRange:
    """Parse a single --range token into a TimeRange relative to now.

    Raises ValueError on unrecognized input.
    """
    t = (token or "").strip().lower()
    if not t or t == "all":
        return TimeRange(label="all time")

    if t in _PRESET_DAYS:
        days = _PRESET_DAYS[t]
        now = datetime.now(timezone.utc)
        return TimeRange(since=now - timedelta(days=days),
                         label=_label_for_days(days))

    m = _NATURAL_RE.match(t)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower().rstrip("s")
        days = {
            "day": n, "week": n * 7, "month": n * 30, "year": n * 365,
        }[unit]
        now = datetime.now(timezone.utc)
        return TimeRange(since=now - timedelta(days=days),
                         label=_label_for_days(days))

    raise ValueError(
        f"Unrecognized range: {token!r}. "
        f"Use a preset (1d/3d/7d/30d/90d/6m/1y/2y/5y/all) or "
        f"a natural form (30days, 6months, 2years)."
    )


def parse_iso(s: str) -> datetime:
    """Parse an ISO date (YYYY-MM-DD) or datetime into UTC-aware datetime."""
    s = (s or "").strip()
    if not s:
        raise ValueError("empty date")
    # Date-only
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        dt = datetime.strptime(s, "%Y-%m-%d")
        return dt.replace(tzinfo=timezone.utc)
    # Full ISO (trailing Z → UTC)
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError as e:
        raise ValueError(f"Unrecognized date {s!r}: expected YYYY-MM-DD "
                         f"or ISO 8601 datetime") from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def build(range_token: str | None = None,
          since: str | None = None,
          until: str | None = None) -> TimeRange:
    """Build a TimeRange from CLI-style flags.

    Precedence: explicit since/until wins over a --range preset.
    If all three are None, returns an unbounded TimeRange.
    """
    base = parse_range(range_token) if range_token else TimeRange()
    if since:
        base.since = parse_iso(since)
    if until:
        base.until = parse_iso(until)
    if since or until:
        parts = []
        if base.since:
            parts.append(f"since {base.since.strftime('%Y-%m-%d')}")
        if base.until:
            parts.append(f"until {base.until.strftime('%Y-%m-%d')}")
        base.label = ", ".join(parts)
    return base


def _label_for_days(days: int) -> str:
    if days < 7:
        return f"last {days} days"
    if days < 30:
        return f"last {days // 7} weeks"
    if days < 365:
        return f"last {days // 30} months"
    years = days / 365
    return f"last {int(years)} years" if years >= 1 else f"last {days} days"
