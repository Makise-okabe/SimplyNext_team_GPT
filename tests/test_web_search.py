import base64
from types import SimpleNamespace
from urllib.parse import quote

from career_agent.job_research_quality import host
from career_agent.tools import web_search
from career_agent.tools.web_search import (
    SearchResult,
    _apply_site_constraint,
    _parse_bing_results,
    _parse_lite_results,
    _simplify_query,
    _site_constraint,
    _unwrap_bing_url,
    _unwrap_duckduckgo_url,
)


def test_unwrap_duckduckgo_redirect_url() -> None:
    wrapped = (
        "https://duckduckgo.com/l/?uddg="
        "https%3A%2F%2Fcareers.example.com%2Fjobs%2F123%3Ffoo%3Dbar"
    )
    assert _unwrap_duckduckgo_url(wrapped) == (
        "https://careers.example.com/jobs/123?foo=bar"
    )


def test_leave_direct_url_unchanged() -> None:
    url = "https://careers.example.com/jobs/123"
    assert _unwrap_duckduckgo_url(url) == url
    assert _unwrap_bing_url(url) == url


def test_unwrap_bing_base64_redirect_url() -> None:
    target = "https://www.linkedin.com/jobs/view/1234567890"
    encoded = base64.urlsafe_b64encode(target.encode()).decode().rstrip("=")
    wrapped = f"https://www.bing.com/ck/a?foo=1&u={quote('a1' + encoded)}&ntb=1"
    assert _unwrap_bing_url(wrapped) == target


def test_parse_bing_result_exposes_real_target_not_click_tracker() -> None:
    target = "https://careers.example.com/jobs/ai-engineer"
    encoded = base64.urlsafe_b64encode(target.encode()).decode().rstrip("=")
    wrapped = f"https://www.bing.com/ck/a?foo=1&u=a1{encoded}&ntb=1"
    html = f"""
    <html><body>
      <li class="b_algo">
        <h2><a href="{wrapped}">AI Engineer - Example Robotics</a></h2>
        <div class="b_caption"><p>Example Robotics AI Engineer Singapore</p></div>
      </li>
    </body></html>
    """
    results = _parse_bing_results(html, max_results=5)
    assert len(results) == 1
    assert results[0].url == target
    assert "bing.com/ck/" not in results[0].url


def test_simplify_query_relaxes_quotes_parentheses_and_punctuation() -> None:
    query = 'THE BOSTON CONSULTING GROUP careers "Associate, Singapore (2027)" Singapore'
    assert _simplify_query(query) == (
        "THE BOSTON CONSULTING GROUP careers Associate Singapore 2027 Singapore"
    )


def test_parse_duckduckgo_lite_result() -> None:
    html = """
    <html><body>
      <a class="result-link" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fcareers.example.com%2Fjobs%2F58603">
        Associate, Singapore (2027)
      </a>
      <div class="result-snippet">Official careers posting</div>
    </body></html>
    """
    results = _parse_lite_results(html, max_results=5)
    assert len(results) == 1
    assert results[0].title == "Associate, Singapore (2027)"
    assert results[0].url == "https://careers.example.com/jobs/58603"


def test_site_constraint_filters_wrong_domain_results() -> None:
    constraint = _site_constraint('site:linkedin.com/jobs "Tesla" "Security Intelligence"')
    results = [
        SearchResult(
            title="Tesla careers",
            url="https://www.tesla.com/careers/search/job/123",
            snippet="Security Intelligence",
        ),
        SearchResult(
            title="Security Intelligence Operations Specialist - Tesla",
            url="https://www.linkedin.com/jobs/view/123456",
            snippet="Tesla Singapore",
        ),
    ]
    filtered = _apply_site_constraint(results, constraint, max_results=10)
    assert [result.url for result in filtered] == [
        "https://www.linkedin.com/jobs/view/123456"
    ]


def test_site_scoped_search_falls_through_until_provider_returns_matching_domain(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    def fake_request(url, query, *, parser, max_results, headers):
        calls.append(url)
        if url == web_search.BING_URL:
            return [
                SearchResult(
                    title="Tesla careers",
                    url="https://www.tesla.com/careers/search/job/123",
                    snippet="Security Intelligence Operations Specialist",
                )
            ]
        if url == web_search.BING_RSS_URL:
            return [
                SearchResult(
                    title="Security Intelligence Operations Specialist - Tesla",
                    url="https://www.linkedin.com/jobs/view/123456",
                    snippet="Tesla Singapore",
                )
            ]
        return []

    monkeypatch.setattr(web_search, "_request_search", fake_request)
    results = web_search.search_public_web(
        'site:linkedin.com/jobs "Tesla" "Security Intelligence Operations Specialist" Singapore',
        max_results=10,
    )
    assert len(results) == 1
    assert results[0].url == "https://www.linkedin.com/jobs/view/123456"
    assert web_search.BING_URL in calls
    assert web_search.BING_RSS_URL in calls


def test_site_search_retries_relaxed_query_but_keeps_site_filter(monkeypatch) -> None:
    queries: list[str] = []
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    def fake_request(url, query, *, parser, max_results, headers):
        queries.append(query)
        if "site:reolink.com" in query.lower():
            return []
        if "reolink.com" in query.lower():
            return [
                SearchResult(
                    title="Backend Engineer - Reolink",
                    url="https://www.reolink.com/careers/backend-engineer",
                    snippet="Reolink Singapore backend engineering role",
                ),
                SearchResult(
                    title="Wrong mirror",
                    url="https://example.com/backend-engineer",
                    snippet="Reolink backend engineer",
                ),
            ]
        return []

    monkeypatch.setattr(web_search, "_request_search", fake_request)
    results = web_search.search_public_web(
        'site:reolink.com "Backend Engineer" Singapore',
        max_results=10,
    )
    assert [item.url for item in results] == [
        "https://www.reolink.com/careers/backend-engineer"
    ]
    assert any("site:reolink.com" in query.lower() for query in queries)
    assert any("reolink.com" in query.lower() and "site:reolink.com" not in query.lower() for query in queries)


def test_optional_tavily_provider_is_preferred_and_site_filtered(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(
        web_search,
        "_search_tavily",
        lambda query, max_results: [
            SearchResult(
                title="Tesla official",
                url="https://www.tesla.com/careers/search/job/123",
                snippet="Security Intelligence Operations Specialist",
            ),
            SearchResult(
                title="Security Intelligence Operations Specialist - Tesla",
                url="https://www.linkedin.com/jobs/view/987654",
                snippet="Tesla Singapore",
            ),
        ],
    )
    monkeypatch.setattr(
        web_search,
        "_request_search",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("fallback should not run")),
    )
    results = web_search.search_public_web(
        'site:linkedin.com/jobs "Tesla" "Security Intelligence Operations Specialist" Singapore',
        max_results=10,
    )
    assert [item.url for item in results] == [
        "https://www.linkedin.com/jobs/view/987654"
    ]


def test_groq_web_search_is_used_when_same_api_key_is_configured(monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    official = SearchResult(
        title="AI Engineer - Reolink",
        url="https://reolink.com/jobs/123",
        snippet="Official job description",
    )
    monkeypatch.setattr(web_search, "_search_groq", lambda query, max_results: [official])
    monkeypatch.setattr(
        web_search,
        "_request_search",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("public scraping should not run")),
    )
    assert web_search.stable_search_api_name() == "groq_compound_web_search"
    assert web_search.search_public_web('"Reolink" "AI Engineer" careers job') == [official]


def test_groq_parser_uses_executed_search_results_not_generated_answer() -> None:
    message = SimpleNamespace(
        content="A made-up URL must not be parsed: https://wrong.example/jobs/1",
        executed_tools=[SimpleNamespace(search_results=SimpleNamespace(results=[
            SimpleNamespace(
                title="Electrical Intern - BH Global",
                url="https://www.bhglobal.com.sg/jobs/electrical-intern/",
                content="Job scope and requirements",
            )
        ]))],
    )
    items = web_search._groq_search_result_items(message)
    assert len(items) == 1
    assert items[0].url == "https://www.bhglobal.com.sg/jobs/electrical-intern/"


def test_host_canonicalizes_www_prefix() -> None:
    assert host("https://www.reolink.com/careers") == "reolink.com"
    assert host("https://reolink.com/careers") == "reolink.com"
    assert host("https://www.linkedin.com/jobs/view/123") == "linkedin.com"
