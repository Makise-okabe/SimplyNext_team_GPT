from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from bs4 import BeautifulSoup


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


BING_URL = "https://www.bing.com/search"
BING_RSS_URL = "https://www.bing.com/search?format=rss"
DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"
DUCKDUCKGO_LITE_URL = "https://lite.duckduckgo.com/lite/"
SEARCH_TIMEOUT_SECONDS = 6.0


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


def _append_result(results: list[SearchResult], seen: set[str], *, title: str, href: str, snippet: str = "") -> None:
    url = _unwrap_duckduckgo_url(href.strip())
    if not url.startswith(("http://", "https://")) or url in seen:
        return
    seen.add(url)
    results.append(SearchResult(title=" ".join(title.split()), url=url, snippet=" ".join(snippet.split())))


def _parse_bing_results(html: str, max_results: int) -> list[SearchResult]:
    soup = BeautifulSoup(html, "html.parser")
    results: list[SearchResult] = []
    seen: set[str] = set()
    for item in soup.select("li.b_algo"):
        anchor = item.select_one("h2 a")
        if not anchor:
            continue
        snippet_node = item.select_one(".b_caption p")
        _append_result(
            results,
            seen,
            title=anchor.get_text(" ", strip=True),
            href=anchor.get("href", ""),
            snippet=snippet_node.get_text(" ", strip=True) if snippet_node else "",
        )
        if len(results) >= max_results:
            break
    return results


def _parse_bing_rss(xml_text: str, max_results: int) -> list[SearchResult]:
    results: list[SearchResult] = []
    seen: set[str] = set()
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    for item in root.findall(".//item"):
        title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        description = item.findtext("description") or ""
        _append_result(results, seen, title=title, href=link, snippet=description)
        if len(results) >= max_results:
            break
    return results


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
            snippet=snippet_node.get_text(" ", strip=True) if snippet_node else "",
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
            anchor for anchor in soup.find_all("a", href=True)
            if anchor.get_text(" ", strip=True)
            and (anchor.get("href", "").startswith(("http://", "https://")) or "uddg=" in anchor.get("href", ""))
        ]
    for anchor in anchors:
        snippet = ""
        parent = anchor.parent
        if parent:
            snippet_node = parent.find_next(class_="result-snippet")
            if snippet_node:
                snippet = snippet_node.get_text(" ", strip=True)
        _append_result(results, seen, title=anchor.get_text(" ", strip=True), href=anchor.get("href", ""), snippet=snippet)
        if len(results) >= max_results:
            break
    return results


def _request_search(url: str, query: str, *, parser, max_results: int, headers: dict[str, str]) -> list[SearchResult]:
    response = httpx.get(
        url,
        params={"q": query},
        headers=headers,
        timeout=SEARCH_TIMEOUT_SECONDS,
        follow_redirects=True,
    )
    response.raise_for_status()
    return parser(response.text, max_results)


def search_public_web(query: str, max_results: int = 5) -> list[SearchResult]:
    """Best-effort public search with provider/parser isolation.

    Order: Bing HTML -> Bing RSS -> DuckDuckGo HTML -> DuckDuckGo Lite.
    Empty/blocked HTML never poisons later jobs or later providers.
    """
    if not query.strip():
        return []

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-SG,en;q=0.9",
    }
    variants = [query.strip()]
    simplified = _simplify_query(query)
    if simplified and simplified != variants[0]:
        variants.append(simplified)

    providers = (
        (BING_URL, _parse_bing_results),
        (BING_RSS_URL, _parse_bing_rss),
        (DUCKDUCKGO_HTML_URL, _parse_html_results),
        (DUCKDUCKGO_LITE_URL, _parse_lite_results),
    )

    for url, parser in providers:
        for variant in variants:
            try:
                results = _request_search(url, variant, parser=parser, max_results=max_results, headers=headers)
                if results:
                    return results
            except Exception:
                break
    return []
