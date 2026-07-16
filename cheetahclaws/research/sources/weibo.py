"""微博 Weibo — via the mobile (m.weibo.cn) PWA JSON endpoint.

Anonymous access returns `ok: -100` (login-required). User supplies
`WEIBO_COOKIE` — paste the `SUB` / `SUBP` cookie values from a logged-in
browser session. Without it, the source skips gracefully.

Engagement = attitudes (点赞) + 2×reposts + comments.
"""
from __future__ import annotations

import html as _html
import os
import re
import urllib.parse

from ..http import get
from ..types import Result
from . import SourceSkipped, SourceSpec, register

_ENDPOINT = "https://m.weibo.cn/api/container/getIndex"


def search(query: str, limit: int = 20, config: dict | None = None,
           time_range=None) -> list[Result]:
    cookie = (
        (config or {}).get("weibo_cookie")
        or os.environ.get("WEIBO_COOKIE")
    )
    if not cookie:
        raise SourceSkipped(
            "WEIBO_COOKIE not set (Weibo's mobile API requires login)"
        )

    # containerid = 100103type=1&q=<query>  (type=1 = 综合, all-content search)
    containerid = urllib.parse.quote(f"100103type=1&q={query}", safe="")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148"
        ),
        "Accept": "application/json",
        "Referer": f"https://m.weibo.cn/search?containerid={containerid}",
        "X-Requested-With": "XMLHttpRequest",
        "MWeibo-Pwa": "1",
        "Cookie": cookie,
    }
    data = get(
        _ENDPOINT,
        params={"containerid": containerid, "page_type": "searchall"},
        headers=headers,
    )

    if data.get("ok") != 1:
        return []

    cards = (data.get("data") or {}).get("cards") or []
    out: list[Result] = []

    for card in cards:
        # card_type 9 → single post; 11 → group of posts
        if card.get("card_type") == 11 and card.get("card_group"):
            for c in card["card_group"]:
                _append_card(c, out, time_range, limit)
                if len(out) >= limit:
                    return out
        elif card.get("card_type") == 9:
            _append_card(card, out, time_range, limit)
            if len(out) >= limit:
                return out
    return out


def _append_card(card, out, time_range, limit):
    mblog = card.get("mblog") or {}
    if not mblog:
        return
    mid = mblog.get("id") or mblog.get("mid")
    user = (mblog.get("user") or {}).get("screen_name") or ""
    screen_id = (mblog.get("user") or {}).get("id")
    if not mid:
        return

    text = _strip_html(mblog.get("text") or "")
    # Long posts may have a "longText" field with the full body
    long_text = (mblog.get("longText") or {}).get("longTextContent", "")
    snippet = _strip_html(long_text)[:500] if long_text else text[:500]

    url = f"https://m.weibo.cn/status/{mid}"
    attitudes = int(mblog.get("attitudes_count") or 0)
    reposts = int(mblog.get("reposts_count") or 0)
    comments = int(mblog.get("comments_count") or 0)
    created_at = mblog.get("created_at") or ""
    published = _parse_weibo_date(created_at)

    if time_range and time_range.is_bounded and published:
        from datetime import datetime
        try:
            pdt = datetime.fromisoformat(published.replace("Z", "+00:00"))
            if time_range.since and pdt < time_range.since:
                return
            if time_range.until and pdt > time_range.until:
                return
        except ValueError:
            pass

    engagement = attitudes + 2 * reposts + comments
    first_line = text.split("\n", 1)[0][:120]

    out.append(Result(
        source="weibo",
        title=first_line or f"@{user} 微博",
        url=url,
        snippet=snippet,
        author=f"@{user}" if user else "",
        published=published,
        engagement_raw=engagement,
        engagement_label=f"{attitudes:,} 赞 · {reposts} 转 · {comments} 评",
        domain="social",
        extra={"user_id": screen_id, "mid": mid,
               "attitudes": attitudes, "reposts": reposts, "comments": comments},
    ))


def _strip_html(s: str) -> str:
    s = _html.unescape(s or "")
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _parse_weibo_date(s: str) -> str:
    """Weibo returns human strings like '刚刚', '1分钟前', '2025-04-18',
    'Sun Apr 18 15:04:00 +0800 2025'. Return ISO UTC when possible."""
    if not s:
        return ""
    from datetime import datetime, timedelta, timezone
    s = s.strip()

    # Relative Chinese forms
    m = re.match(r"(\d+)\s*分钟前", s)
    if m:
        return (datetime.now(timezone.utc) - timedelta(minutes=int(m.group(1)))
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
    m = re.match(r"(\d+)\s*小时前", s)
    if m:
        return (datetime.now(timezone.utc) - timedelta(hours=int(m.group(1)))
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
    if s in ("刚刚", "just now"):
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # "今天 HH:MM"
    m = re.match(r"今天\s*(\d{1,2}):(\d{1,2})", s)
    if m:
        now = datetime.now(timezone.utc)
        return now.replace(hour=int(m.group(1)), minute=int(m.group(2)),
                           second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")

    # "MM-DD" (this year)
    m = re.match(r"(\d{1,2})-(\d{1,2})", s)
    if m:
        y = datetime.now(timezone.utc).year
        return f"{y}-{int(m.group(1)):02d}-{int(m.group(2)):02d}T00:00:00Z"

    # Weibo full: "Sun Apr 18 15:04:00 +0800 2025"
    try:
        dt = datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y")
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        pass

    # "YYYY-MM-DD ..."
    try:
        dt = datetime.fromisoformat(s[:10])
        return dt.replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return ""


register(SourceSpec(
    name="weibo",
    domains=["social", "news"],
    tier="optional",
    search=search,
    requires_env=["WEIBO_COOKIE"],
    description="微博 Weibo — text posts with likes/reposts · needs WEIBO_COOKIE",
))
