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
SEARCH_TIMEOUT_SECONDS = 6.0
FAIL_FAST_HTTP_STATUSES = {401, 403, 429}


def _unwrap_duckduckgo_url(href: str) -> str:
    parsed = urlparse(href)
    query = parse_qs(parsed.query)
    target = query.get("uddg", [None])[0]
    return unquote(target) if target else href


def _simplify_query(query: str) -> str:
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
    soup = BeautifulSoup(html, "html.parser")
    results: list[SearchResult] = []
    seen: set[str] = set()
    anchors = soup.select("a.result-link")
    if not anchors:
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
        timeout=SEARCH_TIMEOUT_SECONDS,
        follow_redirects=True,
    )
    response.raise_for_status()
    return parser(response.text, max_results)


def _is_fail_fast_http_error(exc: Exception) -> bool:
    return (
        isinstance(exc, httpx.HTTPStatusError)
        and exc.response.status_code in FAIL_FAST_HTTP_STATUSES
    )


def search_public_web(query: str, max_results: int = 5) -> list[SearchResult]:
    """Bounded public search suitable for bulk prototype research.

    A query may still be relaxed when the HTML endpoint genuinely returns zero
    results, but an HTTP-level block (401/403/429) is endpoint-wide rather than
    query-specific, so do not waste another identical request with different
    punctuation. We then try Lite once. This keeps one logical search bounded to
    at most two blocked/timed-out requests instead of three 12-second waits.
    """
    if not query.strip():
        return []

    headers = {"User-Agent": "Mozilla/5.0 (compatible; SimplyNextCareerAgent/0.1)"}
    variants = [query.strip()]
    simplified = _simplify_query(query)
    if simplified and simplified != variants[0]:
        variants.append(simplified)

    last_error: Exception | None = None
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
            if _is_fail_fast_http_error(exc):
                break

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

    if last_error is not None:
        raise last_error
    return []
