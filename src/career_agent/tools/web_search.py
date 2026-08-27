from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from bs4 import BeautifulSoup


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


def _unwrap_duckduckgo_url(href: str) -> str:
    parsed = urlparse(href)
    query = parse_qs(parsed.query)
    target = query.get("uddg", [None])[0]
    return unquote(target) if target else href


def search_public_web(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search the public web through DuckDuckGo's HTML endpoint.

    This keeps Prototype 1 API-key-free for web discovery. Search-engine markup can
    change, so callers must treat results as candidates and verify fetched pages.
    """
    if not query.strip():
        return []

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SimplyNextCareerAgent/0.1)"
    }
    response = httpx.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        headers=headers,
        timeout=12.0,
        follow_redirects=True,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    results: list[SearchResult] = []
    seen: set[str] = set()

    for result in soup.select(".result"):
        anchor = result.select_one(".result__a")
        if not anchor:
            continue
        href = anchor.get("href", "").strip()
        url = _unwrap_duckduckgo_url(href)
        if not url.startswith(("http://", "https://")) or url in seen:
            continue
        seen.add(url)
        snippet_node = result.select_one(".result__snippet")
        results.append(
            SearchResult(
                title=anchor.get_text(" ", strip=True),
                url=url,
                snippet=snippet_node.get_text(" ", strip=True) if snippet_node else "",
            )
        )
        if len(results) >= max_results:
            break

    return results
