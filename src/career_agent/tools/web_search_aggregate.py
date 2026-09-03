from __future__ import annotations

from career_agent.tools import web_search
from career_agent.tools.web_search import SearchResult


def _headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-SG,en;q=0.9",
    }


def _merge_results(
    collected: list[SearchResult],
    seen: set[str],
    raw_results: list[SearchResult],
    *,
    original_query: str,
    constraint: tuple[str, str] | None,
    max_results: int,
    strict_relevance: bool,
) -> None:
    if constraint is not None:
        filtered = web_search._apply_site_constraint(raw_results, constraint, max_results)
    elif strict_relevance:
        filtered = web_search._apply_query_relevance(raw_results, original_query, max_results)
    else:
        filtered = raw_results[:max_results]

    for result in filtered:
        if result.url in seen:
            continue
        seen.add(result.url)
        collected.append(result)
        if len(collected) >= max_results:
            return


def search_public_web_aggregated(
    query: str,
    *,
    max_results: int = 12,
    min_results: int = 6,
    strict_relevance: bool = True,
) -> list[SearchResult]:
    """Aggregate several public-search providers for shortlist web discovery.

    ``strict_relevance=True`` keeps the older shortlist-JD behavior. Link
    resolution uses ``False`` so the resolver itself can score lower-confidence
    candidates instead of losing them in an earlier search-layer filter.
    Site constraints are always enforced regardless of this setting.
    """
    if not query.strip() or max_results <= 0:
        return []

    constraint = web_search._site_constraint(query)
    variants = web_search._search_variants(query, constraint)
    headers = _headers()
    collected: list[SearchResult] = []
    seen: set[str] = set()

    for variant in variants[:2]:
        try:
            _merge_results(
                collected,
                seen,
                web_search._search_tavily(variant, max_results),
                original_query=query,
                constraint=constraint,
                max_results=max_results,
                strict_relevance=strict_relevance,
            )
        except Exception:
            pass
        if len(collected) >= max_results:
            return collected[:max_results]

    primary_providers = (
        (web_search.BING_URL, web_search._parse_bing_results),
        (web_search.BING_RSS_URL, web_search._parse_bing_rss),
    )
    fallback_providers = (
        (web_search.DUCKDUCKGO_HTML_URL, web_search._parse_html_results),
        (web_search.DUCKDUCKGO_LITE_URL, web_search._parse_lite_results),
    )

    for variant in variants[:2]:
        for url, parser in primary_providers:
            try:
                raw = web_search._request_search(
                    url,
                    variant,
                    parser=parser,
                    max_results=max_results,
                    headers=headers,
                )
            except Exception:
                continue
            _merge_results(
                collected,
                seen,
                raw,
                original_query=query,
                constraint=constraint,
                max_results=max_results,
                strict_relevance=strict_relevance,
            )
            if len(collected) >= max_results:
                return collected[:max_results]

    if len(collected) >= min_results:
        return collected[:max_results]

    for variant in variants[:2]:
        for url, parser in fallback_providers:
            try:
                raw = web_search._request_search(
                    url,
                    variant,
                    parser=parser,
                    max_results=max_results,
                    headers=headers,
                )
            except Exception:
                continue
            _merge_results(
                collected,
                seen,
                raw,
                original_query=query,
                constraint=constraint,
                max_results=max_results,
                strict_relevance=strict_relevance,
            )
            if len(collected) >= max_results:
                return collected[:max_results]

    return collected[:max_results]
