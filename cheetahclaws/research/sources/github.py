"""GitHub Search — repos + issues.

Unauthenticated: 10 req/min. With GITHUB_TOKEN or CHEETAHCLAWS_GITHUB_TOKEN:
60 req/min (for search) and 5000/hr (for other endpoints).
"""
from __future__ import annotations

import os

from ..http import get
from ..types import Result
from . import SourceSpec, register

_REPO_ENDPOINT  = "https://api.github.com/search/repositories"
_ISSUE_ENDPOINT = "https://api.github.com/search/issues"


def search(query: str, limit: int = 20, config: dict | None = None,
           time_range=None) -> list[Result]:
    headers = {"Accept": "application/vnd.github+json"}
    token = (
        (config or {}).get("github_token")
        or os.environ.get("CHEETAHCLAWS_GITHUB_TOKEN")
        or os.environ.get("GITHUB_TOKEN")
    )
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # GitHub's search syntax allows date qualifiers in the query string.
    repo_q = query
    issue_q = f"{query} is:issue"
    if time_range and time_range.is_bounded:
        if time_range.since:
            repo_q += f" pushed:>={time_range.since.strftime('%Y-%m-%d')}"
            issue_q += f" updated:>={time_range.since.strftime('%Y-%m-%d')}"
        if time_range.until:
            repo_q += f" pushed:<={time_range.until.strftime('%Y-%m-%d')}"
            issue_q += f" updated:<={time_range.until.strftime('%Y-%m-%d')}"

    # Split budget: 60% repos, 40% issues — repos are usually what people want
    repo_limit   = max(1, int(limit * 0.6))
    issue_limit  = max(1, limit - repo_limit)

    out: list[Result] = []

    try:
        data = get(_REPO_ENDPOINT, params={
            "q": repo_q,
            "sort": "stars",
            "order": "desc",
            "per_page": min(repo_limit, 30),
        }, headers=headers)
        for repo in data.get("items") or []:
            stars = int(repo.get("stargazers_count") or 0)
            forks = int(repo.get("forks_count") or 0)
            desc = (repo.get("description") or "")[:400]
            url = repo.get("html_url") or ""
            name = repo.get("full_name") or ""
            updated = repo.get("pushed_at") or repo.get("updated_at") or ""
            if not url:
                continue
            out.append(Result(
                source="github",
                title=f"{name} — {(repo.get('description') or '')[:80]}".rstrip(" —"),
                url=url,
                snippet=desc,
                author=(repo.get("owner") or {}).get("login", ""),
                published=updated,
                engagement_raw=stars,
                engagement_label=f"{stars:,} ⭐ · {forks:,} forks",
                domain="tech",
                extra={"repo": name, "language": repo.get("language", "")},
            ))
    except Exception:
        # Repo search failures shouldn't kill issue search
        pass

    try:
        data = get(_ISSUE_ENDPOINT, params={
            "q": issue_q,
            "sort": "reactions",
            "order": "desc",
            "per_page": min(issue_limit, 30),
        }, headers=headers)
        for item in data.get("items") or []:
            url = item.get("html_url") or ""
            title = item.get("title") or ""
            if not url or not title:
                continue
            reactions = (item.get("reactions") or {}).get("total_count", 0)
            comments = int(item.get("comments") or 0)
            body = (item.get("body") or "")[:400]
            # Pull repo name out of issue URL: /{owner}/{repo}/issues/N
            repo_name = ""
            rurl = item.get("repository_url") or ""
            if rurl.startswith("https://api.github.com/repos/"):
                repo_name = rurl.split("https://api.github.com/repos/", 1)[1]

            out.append(Result(
                source="github",
                title=f"[issue] {title}",
                url=url,
                snippet=body,
                author=(item.get("user") or {}).get("login", ""),
                published=item.get("updated_at") or item.get("created_at") or "",
                engagement_raw=int(reactions) + comments,
                engagement_label=f"{reactions} reactions · {comments} comments",
                domain="tech",
                extra={"repo": repo_name, "state": item.get("state", "")},
            ))
    except Exception:
        pass

    return out


register(SourceSpec(
    name="github",
    domains=["tech"],
    tier="free",
    search=search,
    description="GitHub repos + issues search (stars, forks, reactions)",
))
