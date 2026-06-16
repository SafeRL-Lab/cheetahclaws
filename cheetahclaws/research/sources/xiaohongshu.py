"""小红书 Xiaohongshu (RED) — web search via edith.xiaohongshu.com.

Xiaohongshu's API uses a signed `x-s` request header that changes per
call. Without a browser-extracted cookie + signing material, the
anonymous API is fully locked down. We therefore require the user to
supply both `XHS_COOKIE` and optionally `XHS_X_S_COMMON` (extracted from
a browser session); without these, the source skips.

Engagement = likes + comments + collects.

NOTE: Xiaohongshu's anti-bot is aggressive. Even with cookies, requests
may fail with CAPTCHAs or "too-many-requests" errors. If that happens,
the source surfaces the error as a normal graceful skip rather than
crashing the research run.
"""
from __future__ import annotations

import os

from ..http import post_json
from ..types import Result
from . import SourceSkipped, SourceSpec, register

_ENDPOINT = "https://edith.xiaohongshu.com/api/sns/web/v1/search/notes"


def search(query: str, limit: int = 20, config: dict | None = None,
           time_range=None) -> list[Result]:
    cookie = (
        (config or {}).get("xhs_cookie")
        or os.environ.get("XHS_COOKIE")
        or os.environ.get("XIAOHONGSHU_COOKIE")
    )
    if not cookie:
        raise SourceSkipped(
            "XHS_COOKIE not set — Xiaohongshu requires a browser-extracted "
            "cookie; as an alternative, pass `--sources tavily` with "
            "`<query> site:xiaohongshu.com`"
        )

    # x-s is an HMAC-based signature Xiaohongshu computes client-side;
    # without a runtime signer, we rely on a user-provided static value.
    # (This is typically extracted once per session from devtools.)
    x_s = (
        (config or {}).get("xhs_x_s")
        or os.environ.get("XHS_X_S")
    )

    headers = {
        "Cookie": cookie,
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Referer": "https://www.xiaohongshu.com/",
        "Origin": "https://www.xiaohongshu.com",
        "Accept": "application/json, text/plain, */*",
    }
    if x_s:
        headers["X-S"] = x_s

    payload = {
        "keyword": query,
        "page": 1,
        "page_size": min(limit, 20),
        "search_id": "cheetahclaws-research",
        "sort": "general",     # general | popularity | time
        "note_type": 0,        # 0 = all
    }

    try:
        data = post_json(_ENDPOINT, payload, headers=headers)
    except Exception as e:
        raise SourceSkipped(
            f"Xiaohongshu blocked the request ({type(e).__name__}: "
            f"{str(e)[:100]}). Cookie may be stale; re-extract from browser."
        )

    if not data.get("success"):
        msg = data.get("msg", "")
        raise SourceSkipped(f"Xiaohongshu rejected: {msg}")

    items = ((data.get("data") or {}).get("items")) or []
    out: list[Result] = []

    for item in items:
        note = item.get("note_card") or item.get("note") or {}
        note_id = note.get("id") or item.get("id") or ""
        title = (note.get("display_title") or note.get("title") or "").strip()
        if not title or not note_id:
            continue

        user = note.get("user") or {}
        author = user.get("nickname") or user.get("nick_name") or ""

        interact = note.get("interact_info") or {}
        likes = _parse_count(interact.get("liked_count"))
        comments = _parse_count(interact.get("comment_count"))
        collects = _parse_count(interact.get("collected_count"))
        shares = _parse_count(interact.get("share_count"))

        cover = (note.get("cover") or {}).get("url", "")

        url = f"https://www.xiaohongshu.com/explore/{note_id}"
        engagement = likes + comments + collects + shares // 2

        desc = (note.get("desc") or "")[:400]

        out.append(Result(
            source="xiaohongshu",
            title=title,
            url=url,
            snippet=desc,
            author=f"@{author}" if author else "",
            published="",   # XHS rarely surfaces publish time in search
            engagement_raw=engagement,
            engagement_label=(
                f"{likes:,} 赞 · {comments:,} 评 · {collects:,} 收藏"
            ),
            domain="social",
            extra={"note_id": note_id, "cover": cover,
                   "likes": likes, "comments": comments,
                   "collects": collects},
        ))
        if len(out) >= limit:
            break

    return out


def _parse_count(v) -> int:
    """XHS often returns counts as localized strings: '1.2w' → 12000, '500' → 500."""
    if isinstance(v, int):
        return v
    if not v:
        return 0
    s = str(v).strip()
    try:
        if s.endswith("w") or s.endswith("万"):
            return int(float(s.rstrip("w万")) * 10000)
        if s.endswith("k"):
            return int(float(s.rstrip("k")) * 1000)
        return int(float(s))
    except (ValueError, TypeError):
        return 0


register(SourceSpec(
    name="xiaohongshu",
    domains=["social", "news"],
    tier="optional",
    search=search,
    requires_env=["XHS_COOKIE"],
    description="小红书 Xiaohongshu — note search · needs XHS_COOKIE + often XHS_X_S",
))
