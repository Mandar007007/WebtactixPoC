"""Helpers to pick the correct Chrome tab when attaching via CDP."""
from __future__ import annotations

from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from playwright.async_api import BrowserContext, Page

_SKIP_PREFIXES = (
    "about:",
    "chrome://",
    "chrome-extension://",
    "devtools://",
)


def is_usable_page_url(url: str) -> bool:
    """Return True if the URL looks like a real web page."""
    u = (url or "").strip()
    if not u:
        return False
    if any(u.startswith(prefix) for prefix in _SKIP_PREFIXES):
        return False
    parsed = urlparse(u)
    return parsed.scheme in {"http", "https"}


def _is_usable_page_url(url: str) -> bool:
    """Backward-compatible alias."""
    return is_usable_page_url(url)


def _normalize_url(url: str) -> str:
    """Normalize URLs for loose equality checks."""
    u = (url or "").strip()
    if not u:
        return ""
    parsed = urlparse(u)
    path = parsed.path.rstrip("/")
    base = f"{parsed.scheme}://{parsed.netloc}{path}"
    if parsed.fragment:
        return f"{base}#{parsed.fragment}"
    return base


def _url_matches(a: str, b: str) -> bool:
    """Return True when two URLs likely refer to the same page."""
    na = _normalize_url(a)
    nb = _normalize_url(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    return na.rstrip("/") == nb.rstrip("/")


async def fetch_cdp_targets(cdp_url: str) -> list[dict[str, Any]]:
    """Fetch open CDP targets from Chrome's /json/list endpoint."""
    base = cdp_url.rstrip("/")
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(f"{base}/json/list")
        resp.raise_for_status()
        data = resp.json()
    if not isinstance(data, list):
        return []
    return [t for t in data if isinstance(t, dict)]


def pick_best_target_url(
    targets: list[dict[str, Any]],
    *,
    preferred_url: str = "",
) -> Optional[str]:
    """Pick the best page target URL from CDP /json/list data."""
    pages = [t for t in targets if t.get("type") == "page"]
    candidates = [
        t for t in pages if _is_usable_page_url(str(t.get("url", "")))
    ]
    if not candidates:
        return None

    if preferred_url:
        for target in candidates:
            url = str(target.get("url", ""))
            if _url_matches(url, preferred_url):
                return url

    return str(candidates[-1].get("url", ""))


async def resolve_preferred_page_url(
    cdp_url: str,
    *,
    preferred_url: str = "",
) -> Optional[str]:
    """Resolve the page URL the agent should attach to."""
    targets = await fetch_cdp_targets(cdp_url)
    return pick_best_target_url(
        targets,
        preferred_url=preferred_url,
    )


def select_page_from_context(
    context: BrowserContext,
    *,
    preferred_url: str = "",
) -> Optional[Page]:
    """Pick the Playwright page matching the preferred CDP target."""
    pages = list(context.pages)
    if not pages:
        return None

    if preferred_url:
        for page in pages:
            if _url_matches(page.url, preferred_url):
                return page

    for page in reversed(pages):
        if _is_usable_page_url(page.url):
            return page

    if preferred_url:
        for page in pages:
            if page.url and _url_matches(page.url, preferred_url):
                return page

    return None


def find_any_usable_page(context: BrowserContext) -> Optional[Page]:
    """Return any page with a real http(s) URL, if present."""
    for page in reversed(context.pages):
        if _is_usable_page_url(page.url):
            return page
    return None
