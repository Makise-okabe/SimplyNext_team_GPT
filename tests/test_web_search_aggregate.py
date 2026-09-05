from career_agent.tools import web_search, web_search_aggregate
from career_agent.tools.web_search import SearchResult


def test_aggregated_search_keeps_later_provider_results(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    calls: list[str] = []

    generic = SearchResult(
        title="Reolink Jobs",
        url="https://www.linkedin.com/company/reolink/jobs/",
        snippet="Reolink AI Engineer careers jobs",
    )
    concrete = SearchResult(
        title="AI Engineer - Reolink",
        url="https://job-boards.greenhouse.io/reolink/jobs/123",
        snippet="Reolink AI Engineer responsibilities qualifications",
    )

    def fake_request(url, query, *, parser, max_results, headers):
        calls.append(url)
        if url == web_search.BING_URL:
            return [generic]
        if url == web_search.BING_RSS_URL:
            return [concrete]
        return []

    monkeypatch.setattr(web_search, "_request_search", fake_request)

    results = web_search_aggregate.search_public_web_aggregated(
        '"Reolink" "AI Engineer" careers job',
        max_results=12,
        min_results=6,
    )

    urls = [item.url for item in results]
    assert generic.url in urls
    assert concrete.url in urls
    assert web_search.BING_URL in calls
    assert web_search.BING_RSS_URL in calls
