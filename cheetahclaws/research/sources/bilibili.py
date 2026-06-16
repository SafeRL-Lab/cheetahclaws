"""Bilibili (B站) — Chinese video + community search.

Uses the public web-interface search-all endpoint:
    https://api.bilibili.com/x/web-interface/search/all/v2

Returns mixed groups (video/bangumi/user/live/article…). We extract the
`video` and `article` groups — those carry the engagement signals most
users actually care about. Works without auth for plain queries; some
queries may need a `buvid3` cookie (we send one iff present in env).

Engagement blend: plays + 2×likes + coins + 0.5×comments.
"""
from __future__ import annotations

import html as _html
import os
import re

from ..http import get
from ..types import Result
from . import SourceSpec, register

_ENDPOINT = "https://api.bilibili.com/x/web-interface/search/all/v2"


def search(query: str, limit: int = 20, config: dict | None = None,
           time_range=None) -> list[Result]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Referer": "https://www.bilibili.com/",
        "Accept": "application/json",
    }
    cookie = (
        (config or {}).get("bilibili_cookie")
        or os.environ.get("BILIBILI_COOKIE")
    )
    if cookie:
        headers["Cookie"] = cookie

    params = {"keyword": query, "page": 1}
    data = get(_ENDPOINT, params=params, headers=headers)

    if data.get("code") != 0:
        # code != 0 usually means anti-bot; skipped via `data` key missing
        return []

    groups = ((data.get("data") or {}).get("result")) or []
    out: list[Result] = []

    for grp in groups:
        gtype = grp.get("result_type")
        if gtype not in ("video", "bili_user", "article", "media_ft", "media_bangumi"):
            continue
        items = grp.get("data") or []
        for it in items:
            if gtype == "video":
                title = _clean(it.get("title", ""))
                bvid = it.get("bvid") or ""
                aid = it.get("aid")
                url = f"https://www.bilibili.com/video/{bvid}" if bvid else (
                    f"https://www.bilibili.com/video/av{aid}" if aid else ""
                )
                if not url or not title:
                    continue
                plays = int(it.get("play") or 0)
                likes = int(it.get("like") or 0)
                danmu = int(it.get("video_review") or 0)
                comments = int(it.get("review") or 0)
                author = it.get("author") or ""
                desc = _clean(it.get("description", ""))[:400]
                duration = it.get("duration") or ""
                pubdate = it.get("pubdate")  # unix ts
                if pubdate:
                    from datetime import datetime, timezone
                    try:
                        published = datetime.fromtimestamp(
                            int(pubdate), tz=timezone.utc
                        ).strftime("%Y-%m-%dT%H:%M:%SZ")
                    except (ValueError, OSError):
                        published = ""
                else:
                    published = ""

                # Client-side time filter since B-search API doesn't expose one
                if time_range and time_range.is_bounded and published:
                    from datetime import datetime, timezone as _tz
                    try:
                        pdt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                        if time_range.since and pdt < time_range.since:
                            continue
                        if time_range.until and pdt > time_range.until:
                            continue
                    except ValueError:
                        pass

                eng = plays // 100 + 2 * likes + (danmu // 2) + (comments // 2)
                out.append(Result(
                    source="bilibili",
                    title=f"[video · {duration}] {title}",
                    url=url,
                    snippet=desc,
                    author=f"@{author}" if author else "",
                    published=published,
                    engagement_raw=eng,
                    engagement_label=(
                        f"{plays:,} 播放 · {likes:,} 赞 · {danmu} 弹幕"
                    ),
                    domain="social",
                    extra={"type": "video", "bvid": bvid, "plays": plays,
                           "likes": likes, "danmaku": danmu},
                ))

            elif gtype == "article":
                title = _clean(it.get("title", ""))
                art_id = it.get("id")
                url = f"https://www.bilibili.com/read/cv{art_id}" if art_id else ""
                if not url or not title:
                    continue
                view = int(it.get("view") or 0)
                likes = int(it.get("like") or 0)
                comments = int(it.get("reply") or 0)
                author = (it.get("up") or {}).get("name", "")
                desc = _clean(it.get("desc", ""))[:400]
                pubdate = it.get("pub_time")
                if pubdate:
                    from datetime import datetime, timezone
                    try:
                        published = datetime.fromtimestamp(
                            int(pubdate), tz=timezone.utc
                        ).strftime("%Y-%m-%dT%H:%M:%SZ")
                    except (ValueError, OSError):
                        published = ""
                else:
                    published = ""

                if time_range and time_range.is_bounded and published:
                    from datetime import datetime, timezone as _tz
                    try:
                        pdt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                        if time_range.since and pdt < time_range.since:
                            continue
                        if time_range.until and pdt > time_range.until:
                            continue
                    except ValueError:
                        pass

                eng = view // 50 + 3 * likes + comments
                out.append(Result(
                    source="bilibili",
                    title=f"[article] {title}",
                    url=url,
                    snippet=desc,
                    author=f"@{author}" if author else "",
                    published=published,
                    engagement_raw=eng,
                    engagement_label=f"{view:,} 阅读 · {likes:,} 赞 · {comments} 评论",
                    domain="social",
                    extra={"type": "article", "views": view, "likes": likes},
                ))

            if len(out) >= limit:
                return out
    return out


def _clean(s: str) -> str:
    s = _html.unescape(s or "")
    # Bilibili wraps matched query terms in <em class="keyword">...</em>
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"\s+", " ", s).strip()


register(SourceSpec(
    name="bilibili",
    domains=["social", "tech", "news"],
    tier="free",
    search=search,
    description="Bilibili (B站) — video + article search with plays/likes/bullet counts",
))
