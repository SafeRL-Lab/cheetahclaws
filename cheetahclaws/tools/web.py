"""tools_web.py — Web tool implementations: WebFetch, WebSearch."""
from __future__ import annotations

import re


_DEFAULT_WEB_FETCH_MAX_BYTES = 512 * 1024


def _read_response_bytes(response, max_bytes: int) -> tuple[bytes, bool]:
    """Consume at most ``max_bytes`` from a streamed HTTP response."""
    data = bytearray()
    truncated = False
    for chunk in response.iter_bytes(chunk_size=64 * 1024):
        remaining = max_bytes - len(data)
        if remaining <= 0:
            truncated = True
            break
        if len(chunk) > remaining:
            data.extend(chunk[:remaining])
            truncated = True
            break
        data.extend(chunk)

    # Content-Length lets us report truncation even when the response happens
    # to end exactly at the cap without reading an extra network chunk.
    try:
        content_length = int(response.headers.get("content-length", "0"))
        truncated = truncated or content_length > len(data)
    except (TypeError, ValueError):
        pass
    return bytes(data), truncated


def _webfetch(
    url: str,
    prompt: str = None,
    max_bytes: int = _DEFAULT_WEB_FETCH_MAX_BYTES,
) -> str:
    try:
        import httpx
        byte_limit = max(1, int(max_bytes or _DEFAULT_WEB_FETCH_MAX_BYTES))
        with httpx.stream(
            "GET", url,
            headers={"User-Agent": "NanoClaude/1.0"},
            timeout=30,
            follow_redirects=True,
        ) as response:
            response.raise_for_status()
            raw, source_truncated = _read_response_bytes(response, byte_limit)
            content_type = response.headers.get("content-type", "")
            encoding = response.encoding or "utf-8"

        text = raw.decode(encoding, errors="replace")
        if "html" in content_type.lower():
            text = re.sub(r"<script[^>]*>.*?</script>", "", text,
                          flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text,
                          flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

        output = text[:25000]
        if source_truncated:
            output += (
                f"\n\n[... WebFetch stopped after {byte_limit:,} response bytes "
                "to keep memory and latency bounded ...]"
            )
        return output
    except ImportError:
        return "Error: httpx not installed — run: pip install httpx"
    except Exception as e:
        return f"Error: {e}"


def _websearch(query: str) -> str:
    try:
        import httpx
        url = "https://html.duckduckgo.com/html/"
        r = httpx.get(url, params={"q": query},
                      headers={"User-Agent": "Mozilla/5.0 (compatible)"},
                      timeout=30, follow_redirects=True)
        titles   = re.findall(
            r'class="result__title"[^>]*>.*?<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            r.text, re.DOTALL,
        )
        snippets = re.findall(
            r'class="result__snippet"[^>]*>(.*?)</div>', r.text, re.DOTALL,
        )
        results = []
        for i, (link, title) in enumerate(titles[:8]):
            t = re.sub(r"<[^>]+>", "", title).strip()
            s = re.sub(r"<[^>]+>", "", snippets[i]).strip() if i < len(snippets) else ""
            results.append(f"**{t}**\n{link}\n{s}")
        return "\n\n".join(results) if results else "No results found"
    except ImportError:
        return "Error: httpx not installed — run: pip install httpx"
    except Exception as e:
        return f"Error: {e}"
