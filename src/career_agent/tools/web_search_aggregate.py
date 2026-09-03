from __future__ import annotations

from career_agent.tools import web_search
from career_agent.tools.web_search import SearchResult

PROVIDER_RESULT_CAP = 6
MAX_QUERY_VARIANTS = 2


def _headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-SG,en;q=0.9",
    }


def _filtered_provider_results(
    raw_results: list[SearchResult],
    *,
    original_query: str,
    constraint: tuple[str, str] | None,
    strict_relevance: bool,
    provider_cap: int,
) -> list[SearchResult]:
    if constraint is not None:
        return web_search._apply_site_constraint(raw_results, constraint, provider_cap)
    if strict_relevance:
        return web_search._apply_query_relevance(raw_results, original_query, provider_cap)
    return raw_results[:provider_cap]


def _merge_results(
    collected: list[SearchResult],
    seen: set[str],
    raw_results: list[SearchResult],
    *,
    original_query: str,
    constraint: tuple[str, str] | None,
    strict_relevance: bool,
    provider_cap: int,
) -> None:
    for result in _filtered_provider_results(
        raw_results,
        original_query=original_query,
        constraint=constraint,
        strict_relevance=strict_relevance,
        provider_cap=provider_cap,
    ):
        if result.url in seen:
            continue
        seen.add(result.url)
        collected.append(result)


def search_public_web_aggregated(
    query: str,
    *,
    max_results: int = 12,
    min_results: int = 6,
    strict_relevance: bool = True,
) -> list[SearchResult]:
    """Aggregate multiple search providers without letting one monopolize recall.

    The Track-B search intentionally returns the first useful provider. Career
    opportunity link resolution needs the opposite behavior: a mixed candidate
    pool from several providers so a generic first page cannot hide a concrete JD
    found by a later provider.

    ``strict_relevance=False`` is used only by the v2 link resolver. In that mode
    this function preserves broad recall and delegates company/title scoring to
    the resolver. Site constraints remain enforced in all modes.
    """
    if not query.strip() or max_results <= 0:
        return []

    constraint = web_search._site_constraint(query)
    variants = web_search._search_variants(query, constraint)[:MAX_QUERY_VARIANTS]
    headers = _headers()
    collected: list[SearchResult] = []
    seen: set[str] = set()
    provider_cap = max(1, min(PROVIDER_RESULT_CAP, max_results))

    # Tavily is useful when configured, but it only gets a bounded share of the
    # candidate pool so public providers can still contribute different results.
    for variant in variants:
        try:
            raw = web_search._search_tavily(variant, provider_cap)
        except Exception:
            raw = []
        _merge_results(
            collected,
            seen,
            raw,
            original_query=query,
            constraint=constraint,
            strict_relevance=strict_relevance,
            provider_cap=provider_cap,
        )

    primary_providers = (
        (web_search.BING_URL, web_search._parse_bing_results),
        (web_search.BING_RSS_URL, web_search._parse_bing_rss),
    )
    for variant in variants:
        for url, parser in primary_providers:
            try:
                raw = web_search._request_search(
                    url,
                    variant,
                    parser=parser,
                    max_results=provider_cap,
                    headers=headers,
                )
            except Exception:
                raw = []
            _merge_results(
                collected,
                seen,
                raw,
                original_query=query,
                constraint=constraint,
                strict_relevance=strict_relevance,
                provider_cap=provider_cap,
            )

    # Do not let a full candidate pool from Tavily/Bing prevent DuckDuckGo from
    # contributing when broad recall is requested. In strict mode, however, the
    # older cost-saving behavior is preserved once enough relevant results exist.
    should_try_fallbacks = not strict_relevance or len(collected) < min_results
    if should_try_fallbacks:
        fallback_providers = (
            (web_search.DUCKDUCKGO_HTML_URL, web_search._parse_html_results),
            (web_search.DUCKDUCKGO_LITE_URL, web_search._parse_lite_results),
        )
        for variant in variants:
            for url, parser in fallback_providers:
                try:
                    raw = web_search._request_search(
                        url,
                        variant,
                        parser=parser,
                        max_results=provider_cap,
                        headers=headers,
                    )
                except Exception:
                    raw = []
                _merge_results(
                    collected,
                    seen,
                    raw,
                    original_query=query,
                    constraint=constraint,
                    strict_relevance=strict_relevance,
                    provider_cap=provider_cap,
                )

    return collected[:max_results]
