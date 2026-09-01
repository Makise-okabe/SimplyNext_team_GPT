from __future__ import annotations

import base64
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
SITE_PATTERN = re.compile(r"(?i)(?:^|\s)site:([^\s\"']+)")


def _unwrap_duckduckgo_url(href: str) -> str:
    parsed = urlparse(href)
    query = parse_qs(parsed.query)
    target = query.get("uddg", [None])[0]
    return unquote(target) if target else href


def _decode_bing_target(value: str) -> str | None:
    target = unquote(value or "").strip()
    if target.startswith(("http://", "https://")):
        return target
    if target.startswith("a1") and len(target) > 4:
        payload = target[2:]
        payload += "=" * ((4 - len(payload) % 4) % 4)
        try:
            decoded = base64.urlsafe_b64decode(payload.encode("ascii")).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return None
        if decoded.startswith(("http://", "https://")):
            return decoded
    return None


def _unwrap_bing_url(href: str) -> str:
    parsed = urlparse(href)
    hostname = (parsed.hostname or "").lower()
    if hostname not in {"bing.com", "www.bing.com"}:
        return href
    query = parse_qs(parsed.query)
    for key in ("u", "url", "r"):
        for value in query.get(key, []):
            decoded = _decode_bing_target(value)
            if decoded:
                return decoded
    return href


def _normalize_result_url(href: str) -> str:
    value = _unwrap_duckduckgo_url(href.strip())
    return _unwrap_bing_url(value)


def _normalize_host(value: str) -> str:
    host_value = (value or "").lower().split(":", 1)[0]
    return host_value[4:] if host_value.startswith("www.") else host_value


def _site_constraint(query: str) -> tuple[str, str] | None:
    match = SITE_PATTERN.search(query or "")
    if not match:
        return None
    raw = match.group(1).strip().rstrip("/")
    parsed = urlparse("https://" + raw)
    target_host = _normalize_host(parsed.hostname or "")
    target_path = parsed.path.rstrip("/")
    if not target_host:
        return None
    return target_host, target_path


def _relax_site_query(query: str) -> str:
    """Replace ``site:host/path`` with a normal host/path keyword.

    Some public search frontends ignore or badly implement site: queries. The
    results are still filtered back to the requested site before being returned.
    """
    match = SITE_PATTERN.search(query or "")
    if not match:
        return query
    target = match.group(1).strip().rstrip("/")
    return re.sub(SITE_PATTERN, f" {target}", query, count=1).strip()


def _result_matches_site(result: SearchResult, constraint: tuple[str, str] | None) -> bool:
    if constraint is None:
        return True
    target_host, target_path = constraint
    parsed = urlparse(result.url)
    result_host = _normalize_host(parsed.hostname or "")
    if result_host != target_host and not result_host.endswith("." + target_host):
        return False
    if target_path:
        path = parsed.path.rstrip("/")
        return path == target_path or path.startswith(target_path + "/")
    return True


def _apply_site_constraint(
    results: list[SearchResult],
    constraint: tuple[str, str] | None,
    max_results: int,
) -> list[SearchResult]:
    if constraint is None:
        return results[:max_results]
    return [result for result in results if _result_matches_site(result, constraint)][:max_results]


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
    url = _normalize_result_url(href)
    if not url.startswith(("http://", "https://")) or url in seen:
        return
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    if hostname in {"bing.com", "www.bing.com"} and parsed.path.startswith("/ck/"):
        return
    seen.add(url)
    results.append(
        SearchResult(
            title=" ".join(title.split()),
            url=url,
            snippet=" ".join(snippet.split()),
        )
    )


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


def _search_variants(query: str, constraint: tuple[str, str] | None) -> list[str]:
    variants: list[str] = [query.strip()]
    simplified = _simplify_query(query)
    if simplified and simplified not in variants:
        variants.append(simplified)

    if constraint is not None:
        relaxed = _relax_site_query(query)
        if relaxed and relaxed not in variants:
            variants.append(relaxed)
        relaxed_simple = _simplify_query(relaxed)
        if relaxed_simple and relaxed_simple not in variants:
            variants.append(relaxed_simple)
    return variants


def search_public_web(query: str, max_results: int = 5) -> list[SearchResult]:
    """Best-effort public search with site-safe relaxed fallback.

    A provider result only succeeds when it survives the requested site filter.
    If strict ``site:`` syntax yields nothing, the same provider stack retries a
    relaxed query while still enforcing that exact host/path on returned URLs.
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
    constraint = _site_constraint(query)
    variants = _search_variants(query, constraint)
    providers = (
        (BING_URL, _parse_bing_results),
        (BING_RSS_URL, _parse_bing_rss),
        (DUCKDUCKGO_HTML_URL, _parse_html_results),
        (DUCKDUCKGO_LITE_URL, _parse_lite_results),
    )

    for variant in variants:
        for url, parser in providers:
            try:
                raw_results = _request_search(
                    url,
                    variant,
                    parser=parser,
                    max_results=max_results,
                    headers=headers,
                )
                results = _apply_site_constraint(raw_results, constraint, max_results)
                if results:
                    return results
            except Exception:
                continue
    return []
