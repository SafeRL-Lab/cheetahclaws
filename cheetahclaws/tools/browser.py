"""Lightweight browser tool — render JS pages, extract content, take screenshots.

Requires: pip install cheetahclaws[browser]  (installs playwright)
Falls back gracefully when not installed.
"""
from __future__ import annotations

from cheetahclaws.tool_registry import ToolDef, register_tool

_INSTALL_HINT = (
    "Browser tool requires playwright. Install with:\n"
    "  pip install playwright && playwright install chromium\n"
    "Or: pip install cheetahclaws[browser]"
)


def _web_browse(params: dict, config: dict) -> str:
    """Browse a URL with a headless browser, rendering JavaScript."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return _INSTALL_HINT

    url = params["url"]
    action = params.get("action", "extract")
    selector = params.get("selector")
    wait_seconds = min(params.get("wait", 3), 30)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_default_timeout(30000)
            page.goto(url, wait_until="networkidle", timeout=30000)

            if wait_seconds > 0:
                page.wait_for_timeout(int(wait_seconds * 1000))

            if action == "screenshot":
                import base64
                screenshot = page.screenshot(full_page=False)
                b64 = base64.b64encode(screenshot).decode()
                browser.close()
                return f"Screenshot captured ({len(screenshot)} bytes, base64-encoded).\ndata:image/png;base64,{b64[:100]}..."

            elif action == "click":
                if not selector:
                    browser.close()
                    return "Error: 'selector' is required for click action."
                page.click(selector)
                page.wait_for_timeout(2000)
                content = page.content()
                browser.close()
                text = _html_to_text(content)
                return _truncate(text, 30000)

            else:  # extract (default)
                if selector:
                    elements = page.query_selector_all(selector)
                    texts = []
                    for el in elements[:50]:
                        t = el.inner_text()
                        if t.strip():
                            texts.append(t.strip())
                    browser.close()
                    if not texts:
                        return f"No elements found matching selector: {selector}"
                    return f"Found {len(texts)} element(s):\n\n" + "\n---\n".join(texts)
                else:
                    content = page.inner_text("body")
                    browser.close()
                    return _truncate(content, 30000)

    except Exception as e:
        return f"Browser error: {type(e).__name__}: {e}"


def _html_to_text(html: str) -> str:
    """Simple HTML to text extraction."""
    import re
    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[... truncated {len(text) - limit} chars ...]"


# ── Register ─────────────────────────────────────────────────────────────

register_tool(ToolDef(
    name="WebBrowse",
    schema={
        "name": "WebBrowse",
        "description": (
            "Browse a URL with a headless browser that renders JavaScript. "
            "Use this instead of WebFetch for dynamic/SPA pages that require JS rendering. "
            "Actions: 'extract' (get text content), 'screenshot' (capture page image), "
            "'click' (click an element then extract). "
            "Optionally filter content with a CSS selector."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to browse",
                },
                "action": {
                    "type": "string",
                    "enum": ["extract", "screenshot", "click"],
                    "description": "Action to perform (default: extract)",
                },
                "selector": {
                    "type": "string",
                    "description": "Optional CSS selector to target specific elements",
                },
                "wait": {
                    "type": "number",
                    "description": "Seconds to wait after page load for JS to render (default: 3, max: 30)",
                },
            },
            "required": ["url"],
        },
    },
    func=_web_browse,
    read_only=True,
    concurrent_safe=True,
))
