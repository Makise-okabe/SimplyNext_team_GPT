from career_agent.tools import web_search, web_search_aggregate
from career_agent.tools.web_search import SearchResult


def test_aggregated_search_keeps_later_provider_results(monkeypatch):
    monkeypatch.delenv("AWS_AGENTCORE_GATEWAY_URL", raising=False)
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


def test_agentcore_cannot_monopolize_aggregated_results(monkeypatch):
    monkeypatch.setenv("AWS_AGENTCORE_GATEWAY_URL", "https://gateway.example/mcp")
    aws_rows = [
        SearchResult(f"AWS {index}", f"https://noise.example/{index}", "Example Engineer")
        for index in range(8)
    ]
    official = SearchResult(
        "Electrical Intern",
        "https://www.bhglobal.com.sg/jobs/electrical-intern/",
        "BH Global Corporation job scope",
    )
    monkeypatch.setattr(web_search, "_search_aws_agentcore", lambda *args, **kwargs: aws_rows)

    def fake_request(url, query, *, parser, max_results, headers):
        return [official] if url == web_search.BING_URL else []

    monkeypatch.setattr(web_search, "_request_search", fake_request)
    rows = web_search_aggregate.search_public_web_aggregated(
        '"BH Global Corporation Ltd" "Electrical Intern"',
        max_results=4,
        min_results=2,
        strict_relevance=False,
    )
    assert rows[0] == official
    assert len([row for row in rows if "noise.example" in row.url]) <= 2
