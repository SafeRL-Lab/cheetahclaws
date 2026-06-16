"""Zhihu (知乎) — Chinese-language Q&A platform via the v4 search API.

Zhihu's anti-bot blocks anonymous API access (HTTP 401). To enable this
source, the user supplies `ZHIHU_COOKIE` — the `d_c0` + `z_c0` cookie
values copy-pasted from a browser session (Zhihu's OAuth flow is closed
to third-party apps). Without it, the source silently skips.

This gives high-signal access to Chinese-language takes on any topic —
technical, financial, academic, cultural. Zhihu's upvote/comment counts
are the engagement signal.
"""
from __future__ import annotations

import html
import os
import re

from ..http import get
from ..types import Result
from . import SourceSkipped, SourceSpec, register

_ENDPOINT = "https://www.zhihu.com/api/v4/search_v3"


def search(query: str, limit: int = 20, config: dict | None = None,
           time_range=None) -> list[Result]:
    cookie = (
        (config or {}).get("zhihu_cookie")
        or os.environ.get("ZHIHU_COOKIE")
    )
    if not cookie:
        raise SourceSkipped("ZHIHU_COOKIE not set (Zhihu blocks anonymous API access)")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/121.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.zhihu.com/search",
        "Cookie": cookie,
        "x-requested-with": "fetch",
    }

    data = get(_ENDPOINT, params={
        "t": "general",
        "q": query,
        "correction": 1,
        "offset": 0,
        "limit": min(limit, 50),
        "lc_idx": 0,
        "show_all_topics": 0,
    }, headers=headers)

    out: list[Result] = []
    for item in data.get("data") or []:
        obj = item.get("object") or {}
        obj_type = obj.get("type") or item.get("type") or ""
        if obj_type not in ("answer", "article", "question"):
            continue

        if obj_type == "answer":
            question = obj.get("question") or {}
            title = _strip_html(question.get("name") or question.get("title") or "")
            qid = question.get("id")
            aid = obj.get("id")
            url = f"https://www.zhihu.com/question/{qid}/answer/{aid}" if qid and aid else ""
            excerpt = _strip_html(obj.get("excerpt") or "")
            author = ((obj.get("author") or {}).get("name")) or ""
            votes = int(obj.get("voteup_count") or 0)
            comments = int(obj.get("comment_count") or 0)
            published = _fmt_ts(obj.get("created_time") or obj.get("updated_time"))
        elif obj_type == "article":
            title = _strip_html(obj.get("title") or "")
            aid = obj.get("id")
            url = f"https://zhuanlan.zhihu.com/p/{aid}" if aid else ""
            excerpt = _strip_html(obj.get("excerpt") or "")
            author = ((obj.get("author") or {}).get("name")) or ""
            votes = int(obj.get("voteup_count") or 0)
            comments = int(obj.get("comment_count") or 0)
            published = _fmt_ts(obj.get("created") or obj.get("updated"))
        else:  # question
            title = _strip_html(obj.get("name") or obj.get("title") or "")
            qid = obj.get("id")
            url = f"https://www.zhihu.com/question/{qid}" if qid else ""
            excerpt = _strip_html(obj.get("excerpt") or obj.get("detail") or "")
            author = ((obj.get("author") or {}).get("name")) or ""
            votes = int(obj.get("follower_count") or 0)
            comments = int(obj.get("answer_count") or 0)
            published = _fmt_ts(obj.get("created") or obj.get("updated_time"))

        if not title or not url:
            continue

        label = {
            "answer":   f"{votes} 赞 · {comments} 评论",
            "article":  f"{votes} 赞 · {comments} 评论 · 专栏",
            "question": f"{votes} 关注 · {comments} 回答",
        }.get(obj_type, f"{votes}")

        out.append(Result(
            source="zhihu",
            title=f"[{obj_type}] {title}",
            url=url,
            snippet=excerpt[:500],
            author=author,
            published=published,
            engagement_raw=votes + (comments // 2),
            engagement_label=label,
            domain="social",
            extra={"zhihu_type": obj_type, "votes": votes, "comments": comments},
        ))
    return out


def _strip_html(s: str) -> str:
    s = html.unescape(s or "")
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _fmt_ts(ts) -> str:
    if not ts:
        return ""
    from datetime import datetime, timezone
    try:
        return datetime.fromtimestamp(
            float(ts), tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, OSError, TypeError):
        return ""


register(SourceSpec(
    name="zhihu",
    domains=["social", "tech", "finance", "news"],
    tier="optional",
    search=search,
    requires_env=["ZHIHU_COOKIE"],
    description="知乎 — Chinese Q&A (answers, articles, questions) · needs ZHIHU_COOKIE",
))
