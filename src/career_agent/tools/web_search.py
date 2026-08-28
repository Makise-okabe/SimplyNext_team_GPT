from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from bs4 import BeautifulSoup


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"
DUCKDUCKGO_LITE_URL = "https://lite.duckduckgo.com/lite/"


def _unwrap_duckduckgo_url(href: str) -> str:
    parsed = urlparse(href)
    query = parse_qs(parsed.query)
    target = query.get("uddg", [None])[0]
    return unquote(target) if target else href


def _simplify_query(query: str) -> str:
    """Relax brittle exact-search syntax after a zero-result attempt.

    Search engines can return no HTML results for otherwise valid roles when a
    query contains exact quotes, punctuation, parentheses, or forwarded-email
    formatting. The relaxed variant keeps all lexical information while removing
    operators that make discovery unnecessarily brittle.
    """
    value = query.replace('"', " ").replace("'", " ")
    value = re.sub(r"[(),|]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _append_result(
    results: list[SearchResult],
    seen: set[str],
    *,
    title: str,
    href: str,
    snippet: str = "",
) -> None:
    url = _unwrap_duckduckgo_url(href.strip())
    if not url.startswith(("http://", "https://")) or url in seen:
        return
    seen.add(url)
    results.append(
        SearchResult(
            title=" ".join(title.split()),
            url=url,
            snippet=" ".join(snippet.split()),
        )
    )


def _parse_html_results(html: str, max_results: int) -> list[SearchResult]:
    soup = BeautifulSoup(html, "html.parser")
    results: list[SearchResult] = []
    seen: set[str] = set()

    for result in soup.select(".result"):
        anchor = result.select_one(".result__a")
        if not anchor:
            continue
        snippet_node = result.select_one(".result__snippet")
        _append_result(
            results,
            seen,
            title=anchor.get_text(" ", strip=True),
            href=anchor.get("href", ""),
            snippet=(snippet_node.get_text(" ", strip=True) if snippet_node else ""),
        )
        if len(results) >= max_results:
            break

    return results


def _parse_lite_results(html: str, max_results: int) -> list[SearchResult]:
    """Parse DuckDuckGo Lite, whose markup differs from the HTML endpoint."""
    soup = BeautifulSoup(html, "html.parser")
    results: list[SearchResult] = []
    seen: set[str] = set()

    anchors = soup.select("a.result-link")
    if not anchors:
        # DuckDuckGo Lite has used both classed and unclassed outbound links over
        # time. This fallback stays conservative by requiring an HTTP(S) or DDG
        # redirect href and non-empty visible text.
        anchors = [
            anchor
            for anchor in soup.find_all("a", href=True)
            if anchor.get_text(" ", strip=True)
            and (
                anchor.get("href", "").startswith(("http://", "https://"))
                or "uddg=" in anchor.get("href", "")
            )
        ]

    for anchor in anchors:
        snippet = ""
        parent = anchor.parent
        if parent:
            snippet_node = parent.find_next(class_="result-snippet")
            if snippet_node:
                snippet = snippet_node.get_text(" ", strip=True)
        _append_result(
            results,
            seen,
            title=anchor.get_text(" ", strip=True),
            href=anchor.get("href", ""),
            snippet=snippet,
        )
        if len(results) >= max_results:
            break

    return results


def _request_search(
    url: str,
    query: str,
    *,
    parser,
    max_results: int,
    headers: dict[str, str],
) -> list[SearchResult]:
    response = httpx.get(
        url,
        params={"q": query},
        headers=headers,
        timeout=12.0,
        follow_redirects=True,
    )
    response.raise_for_status()
    return parser(response.text, max_results)


def search_public_web(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search the public web with bounded DuckDuckGo discovery fallbacks.

    Prototype 1 remains API-key-free, but one brittle HTML endpoint must not make
    `0 results` synonymous with `no public posting`. We therefore try the original
    query first, then a relaxed lexical query, and finally DuckDuckGo Lite.
    Callers must still verify every returned candidate against fetched evidence.
    """
    if not query.strip():
        return []

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SimplyNextCareerAgent/0.1)"
    }
    variants = [query.strip()]
    simplified = _simplify_query(query)
    if simplified and simplified != variants[0]:
        variants.append(simplified)

    last_error: Exception | None = None

    # Prefer the richer HTML endpoint. A simplified second attempt is especially
    # useful for exact-role queries containing punctuation/year parentheses.
    for variant in variants:
        try:
            results = _request_search(
                DUCKDUCKGO_HTML_URL,
                variant,
                parser=_parse_html_results,
                max_results=max_results,
                headers=headers,
            )
            if results:
                return results
        except Exception as exc:
            last_error = exc

    # Final bounded fallback: alternate DDG frontend/markup with the relaxed query.
    lite_query = variants[-1]
    try:
        results = _request_search(
            DUCKDUCKGO_LITE_URL,
            lite_query,
            parser=_parse_lite_results,
            max_results=max_results,
            headers=headers,
        )
        if results:
            return results
    except Exception as exc:
        last_error = exc

    # Preserve historical behaviour for genuine zero-result searches. If every
    # endpoint failed at the HTTP layer, surface the last error so callers can
    # distinguish infrastructure failure from an empty search index.
    if last_error is not None:
        raise last_error
    return []
