"""tools_web.py — Web tool implementations: WebFetch, WebSearch."""
from __future__ import annotations

import asyncio
import time
from html.parser import HTMLParser
from urllib.parse import urljoin


_DEFAULT_WEB_FETCH_MAX_BYTES = 512 * 1024
_DEFAULT_WEB_MAX_SECONDS = 30

# HTML void elements emit no matching end tag (when written bare, e.g.
# ``<img>``/``<br>`` as DuckDuckGo serves them), so depth counters that
# increment on every start tag and decrement on every end tag would drift.
_VOID_ELEMENTS = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
})


def _coerce_max_seconds(value: int | float) -> float:
    try:
        return float(max(1, float(value)))
    except (TypeError, ValueError):
        return float(_DEFAULT_WEB_MAX_SECONDS)


class _HTMLTextExtractor(HTMLParser):
    """Extract visible HTML text without regex backtracking hazards."""

    def __init__(self, char_cap: int):
        super().__init__(convert_charrefs=True)
        self._char_cap = char_cap
        self._parts: list[str] = []
        self._chars = 0
        self._ignored_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"script", "style", "noscript", "template"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag):
        if tag.lower() in {"script", "style", "noscript", "template"}:
            self._ignored_depth = max(0, self._ignored_depth - 1)

    def handle_data(self, data):
        if self._ignored_depth or not data or self._chars >= self._char_cap:
            return
        text = " ".join(data.split())
        if not text:
            return
        remaining = self._char_cap - self._chars
        self._parts.append(text[:remaining])
        self._chars += min(len(text), remaining)

    def get_text(self) -> str:
        return " ".join(self._parts).strip()


class _DuckDuckGoResultParser(HTMLParser):
    """Small, bounded parser for DuckDuckGo's HTML-only result cards."""

    def __init__(self, result_cap: int = 8, field_cap: int = 4_000):
        super().__init__(convert_charrefs=True)
        self._result_cap = result_cap
        self._field_cap = field_cap
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._result_div_depth = 0
        self._title_depth = 0
        self._snippet_depth = 0

    @staticmethod
    def _classes(attrs) -> set[str]:
        # A valueless attribute (e.g. ``<div class>``) yields ('class', None),
        # so ``.get("class", "")`` returns None, not the default — guard with
        # ``or ""`` so a single bare attribute can't crash the whole parse.
        return set((dict(attrs).get("class") or "").split())

    def _finish_current(self) -> None:
        if self._current and (self._current["title"] or self._current["link"]):
            self.results.append(self._current)
        self._current = None
        self._result_div_depth = 0
        self._title_depth = 0
        self._snippet_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in _VOID_ELEMENTS:
            # No end tag will follow, so touching the depth counters here would
            # leave title/snippet depth permanently offset and misattribute the
            # rest of the result's text.
            return
        classes = self._classes(attrs)
        if tag.lower() == "div" and "result" in classes:
            self._finish_current()
            if len(self.results) < self._result_cap:
                self._current = {"title": "", "link": "", "snippet": ""}
                self._result_div_depth = 1
        if self._current is None:
            return
        if tag.lower() == "div" and not ("result" in classes and self._result_div_depth == 1):
            self._result_div_depth += 1
        if "result__title" in classes:
            self._title_depth += 1
        elif self._title_depth:
            self._title_depth += 1
        if "result__snippet" in classes:
            self._snippet_depth += 1
        elif self._snippet_depth:
            self._snippet_depth += 1
        if tag.lower() == "a" and self._title_depth:
            href = dict(attrs).get("href")
            if href:
                self._current["link"] = href[:self._field_cap]

    def handle_endtag(self, tag):
        if tag.lower() in _VOID_ELEMENTS:
            # Mirror handle_starttag: a self-closed void tag (``<br/>``) drives
            # the parser's default start+end pair, so skip it on both sides to
            # keep the depth counters balanced.
            return
        if tag.lower() == "div" and self._current is not None:
            self._result_div_depth -= 1
            if self._result_div_depth <= 0:
                self._finish_current()
                return
        if self._title_depth:
            self._title_depth -= 1
        if self._snippet_depth:
            self._snippet_depth -= 1

    def handle_data(self, data):
        if self._current is None or not data:
            return
        field = "title" if self._title_depth else "snippet" if self._snippet_depth else None
        if not field:
            return
        text = " ".join(data.split())
        if not text:
            return
        existing = self._current[field]
        if len(existing) < self._field_cap:
            separator = " " if existing else ""
            self._current[field] = (existing + separator + text)[:self._field_cap]

    def close(self):
        super().close()
        self._finish_current()


def _read_response_bytes(
    response,
    max_bytes: int,
    max_seconds: int | float = _DEFAULT_WEB_MAX_SECONDS,
    deadline: float | None = None,
) -> tuple[bytes, bool]:
    """Consume at most ``max_bytes`` from a streamed HTTP response."""
    data = bytearray()
    truncated = False
    deadline = deadline if deadline is not None else time.monotonic() + _coerce_max_seconds(max_seconds)
    try:
        content_length = int(response.headers.get("content-length", ""))
    except (TypeError, ValueError):
        content_length = None

    if time.monotonic() >= deadline:
        return b"", True
    # With identity encoding, raw chunks avoid HTTPX's decoded-byte chunker
    # buffering a slow drip until it reaches 64 KiB before we can check time.
    iterator = response.iter_raw() if hasattr(response, "iter_raw") else response.iter_bytes(chunk_size=64 * 1024)
    for chunk in iterator:
        if time.monotonic() >= deadline:
            truncated = True
            break
        remaining = max_bytes - len(data)
        if remaining <= 0:
            truncated = True
            break
        if len(chunk) > remaining:
            data.extend(chunk[:remaining])
            truncated = True
            break
        data.extend(chunk)
        if len(data) == max_bytes:
            # Stop immediately: an unknown length is conservatively marked as
            # truncated rather than waiting for another possibly slow chunk.
            truncated = content_length is None or content_length > len(data)
            break

    if content_length is not None:
        truncated = truncated or content_length > len(data)
    return bytes(data), truncated


def _stream_identity_bytes(httpx, url: str, *, params: dict | None,
                           headers: dict[str, str], max_bytes: int,
                           deadline: float) -> tuple[bytes | None, bool, dict, str, str | None]:
    """Follow a small redirect chain without resetting the total deadline."""
    current_url = url
    current_params = params
    for redirect_count in range(6):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None, True, {}, "utf-8", "Error: web request exceeded its elapsed-time budget."
        with httpx.stream(
            "GET", current_url, params=current_params, headers=headers,
            timeout=remaining, follow_redirects=False,
        ) as response:
            response_headers = dict(response.headers)
            status = getattr(response, "status_code", 0)
            location = response_headers.get("location")
            if status in {301, 302, 303, 307, 308} and location:
                if redirect_count == 5:
                    return None, True, {}, "utf-8", "Error: web request exceeded 5 redirects."
                current_url = urljoin(str(getattr(response, "url", current_url)), location)
                current_params = None
                continue
            response.raise_for_status()
            content_encoding = response_headers.get("content-encoding", "identity").lower()
            if content_encoding not in {"", "identity"}:
                return (
                    None, True, {}, "utf-8",
                    "Error: compressed HTTP responses are not accepted; retry with identity encoding.",
                )
            raw, truncated = _read_response_bytes(
                response, max_bytes, deadline=deadline,
            )
            return raw, truncated, response_headers, response.encoding or "utf-8", None
    return None, True, {}, "utf-8", "Error: web request could not resolve redirects."


async def _read_response_bytes_async(
    response,
    max_bytes: int,
    deadline: float,
) -> tuple[bytes, bool]:
    """Async counterpart that permits cancellation while a body is stalled."""
    data = bytearray()
    truncated = False
    try:
        content_length = int(response.headers.get("content-length", ""))
    except (TypeError, ValueError):
        content_length = None

    async for chunk in response.aiter_raw():
        if time.monotonic() >= deadline:
            truncated = True
            break
        remaining = max_bytes - len(data)
        if remaining <= 0:
            truncated = True
            break
        if len(chunk) > remaining:
            data.extend(chunk[:remaining])
            truncated = True
            break
        data.extend(chunk)
        if len(data) == max_bytes:
            truncated = content_length is None or content_length > len(data)
            break

    if content_length is not None:
        truncated = truncated or content_length > len(data)
    return bytes(data), truncated


async def _stream_identity_bytes_async_impl(
    httpx,
    url: str,
    *,
    params: dict | None,
    headers: dict[str, str],
    max_bytes: int,
    deadline: float,
) -> tuple[bytes | None, bool, dict, str, str | None]:
    """Async transport path; outer ``wait_for`` owns the global deadline."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None, True, {}, "utf-8", "Error: web request exceeded its elapsed-time budget."

    async with httpx.AsyncClient(
        headers=headers,
        timeout=remaining,
        follow_redirects=False,
    ) as client:
        current_url = url
        current_params = params
        for redirect_count in range(6):
            request = client.build_request(
                "GET", current_url, params=current_params,
            )
            response = await client.send(request, stream=True)
            try:
                response_headers = dict(response.headers)
                status = getattr(response, "status_code", 0)
                location = response_headers.get("location")
                if status in {301, 302, 303, 307, 308} and location:
                    if redirect_count == 5:
                        return None, True, {}, "utf-8", "Error: web request exceeded 5 redirects."
                    current_url = urljoin(str(getattr(response, "url", current_url)), location)
                    current_params = None
                    continue
                response.raise_for_status()
                content_encoding = response_headers.get("content-encoding", "identity").lower()
                if content_encoding not in {"", "identity"}:
                    return (
                        None, True, {}, "utf-8",
                        "Error: compressed HTTP responses are not accepted; retry with identity encoding.",
                    )
                raw, truncated = await _read_response_bytes_async(
                    response, max_bytes, deadline,
                )
                return raw, truncated, response_headers, response.encoding or "utf-8", None
            finally:
                await response.aclose()
    return None, True, {}, "utf-8", "Error: web request could not resolve redirects."


async def _stream_identity_bytes_async(
    httpx,
    url: str,
    *,
    params: dict | None,
    headers: dict[str, str],
    max_bytes: int,
    deadline: float,
) -> tuple[bytes | None, bool, dict, str, str | None]:
    """Enforce one cancellable elapsed-time budget for the entire request."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None, True, {}, "utf-8", "Error: web request exceeded its elapsed-time budget."
    try:
        return await asyncio.wait_for(
            _stream_identity_bytes_async_impl(
                httpx, url, params=params, headers=headers,
                max_bytes=max_bytes, deadline=deadline,
            ),
            timeout=remaining,
        )
    except asyncio.TimeoutError:
        return None, True, {}, "utf-8", "Error: web request exceeded its elapsed-time budget."


def _stream_bounded_identity_bytes(httpx, url: str, *, params: dict | None,
                                   headers: dict[str, str], max_bytes: int,
                                   deadline: float) -> tuple[bytes | None, bool, dict, str, str | None]:
    """Use cancellable async I/O in production, retaining a minimal test fallback."""
    if not hasattr(httpx, "AsyncClient"):
        # Lightweight fake HTTP clients used by unit tests only implement the
        # synchronous ``stream`` API. Real httpx always exposes AsyncClient.
        return _stream_identity_bytes(
            httpx, url, params=params, headers=headers,
            max_bytes=max_bytes, deadline=deadline,
        )
    return asyncio.run(_stream_identity_bytes_async(
        httpx, url, params=params, headers=headers,
        max_bytes=max_bytes, deadline=deadline,
    ))


def _webfetch(
    url: str,
    prompt: str = None,
    max_bytes: int = _DEFAULT_WEB_FETCH_MAX_BYTES,
    max_seconds: int | float = _DEFAULT_WEB_MAX_SECONDS,
) -> str:
    try:
        import httpx
        byte_limit = max(1, int(max_bytes or _DEFAULT_WEB_FETCH_MAX_BYTES))
        seconds = _coerce_max_seconds(max_seconds)
        deadline = time.monotonic() + seconds
        raw, source_truncated, response_headers, encoding, error = _stream_bounded_identity_bytes(
            httpx, url, params=None,
            headers={"User-Agent": "NanoClaude/1.0", "Accept-Encoding": "identity"},
            max_bytes=byte_limit, deadline=deadline,
        )
        if error:
            return error
        assert raw is not None
        content_type = response_headers.get("content-type", "")

        text = raw.decode(encoding, errors="replace")
        if "html" in content_type.lower():
            extractor = _HTMLTextExtractor(char_cap=25_000)
            extractor.feed(text)
            extractor.close()
            text = extractor.get_text()

        output = text[:25000]
        if source_truncated:
            output += (
                f"\n\n[... WebFetch stopped after {len(raw):,} response bytes "
                "at its configured byte bound or elapsed collection budget ...]"
            )
        return output
    except ImportError:
        return "Error: httpx not installed — run: pip install httpx"
    except Exception as e:
        return f"Error: {e}"


def _websearch(
    query: str,
    max_bytes: int = _DEFAULT_WEB_FETCH_MAX_BYTES,
    max_seconds: int | float = _DEFAULT_WEB_MAX_SECONDS,
) -> str:
    try:
        import httpx
        byte_limit = max(1, int(max_bytes or _DEFAULT_WEB_FETCH_MAX_BYTES))
        seconds = _coerce_max_seconds(max_seconds)
        deadline = time.monotonic() + seconds
        url = "https://html.duckduckgo.com/html/"
        raw, source_truncated, _headers, encoding, error = _stream_bounded_identity_bytes(
            httpx, url, params={"q": query},
            headers={
                "User-Agent": "Mozilla/5.0 (compatible)",
                "Accept-Encoding": "identity",
            },
            max_bytes=byte_limit, deadline=deadline,
        )
        if error:
            return error
        assert raw is not None
        parser = _DuckDuckGoResultParser()
        parser.feed(raw.decode(encoding, errors="replace"))
        parser.close()
        output = "\n\n".join(
            f"**{result['title']}**\n{result['link']}\n{result['snippet']}"
            for result in parser.results
        ) or "No results found"
        if source_truncated:
            output += (
                f"\n\n[... WebSearch stopped after {len(raw):,} response bytes "
                "at its configured byte bound or elapsed collection budget ...]"
            )
        return output
    except ImportError:
        return "Error: httpx not installed — run: pip install httpx"
    except Exception as e:
        return f"Error: {e}"
