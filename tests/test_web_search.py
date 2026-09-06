import base64
import json
from urllib.parse import quote

import httpx

from career_agent.job_research_quality import host
from career_agent.tools import web_search
from career_agent.tools.web_search import (
    SearchResult,
    _apply_site_constraint,
    _agentcore_arguments,
    _parse_bing_results,
    _parse_lite_results,
    _simplify_query,
    _site_constraint,
    _unwrap_bing_url,
    _unwrap_duckduckgo_url,
)


def test_agentcore_site_syntax_uses_native_domain_filter() -> None:
    arguments = _agentcore_arguments(
        'site:bhglobal.com.sg/jobs "Electrical Intern"',
        8,
    )
    assert arguments == {
        "query": '"Electrical Intern"',
        "maxResults": 8,
        "filters": {"domainFilter": {"include": ["bhglobal.com.sg"]}},
    }


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
    monkeypatch.delenv("AWS_AGENTCORE_GATEWAY_URL", raising=False)

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
    monkeypatch.delenv("AWS_AGENTCORE_GATEWAY_URL", raising=False)

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


def test_agentcore_provider_is_preferred_and_site_filtered(monkeypatch) -> None:
    monkeypatch.setenv("AWS_AGENTCORE_GATEWAY_URL", "https://gateway.example/mcp")
    monkeypatch.setattr(
        web_search,
        "_search_aws_agentcore",
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


def test_agentcore_results_are_combined_with_public_results(monkeypatch) -> None:
    monkeypatch.setenv("AWS_AGENTCORE_GATEWAY_URL", "https://gateway.example/mcp")
    official = SearchResult(
        title="AI Engineer - Reolink",
        url="https://reolink.com/jobs/123",
        snippet="Official job description",
    )
    monkeypatch.setattr(web_search, "_search_aws_agentcore", lambda query, max_results: [official])
    public = SearchResult(
        title="AI Engineer - Reolink official careers",
        url="https://careers.reolink.com/jobs/456",
        snippet="Official role page",
    )
    monkeypatch.setattr(web_search, "_request_search", lambda *args, **kwargs: [public])
    assert web_search.stable_search_api_name() == "aws_agentcore_web_search"
    assert web_search.search_public_web('"Reolink" "AI Engineer" careers job') == [official, public]


def test_official_role_result_survives_when_title_omits_company(monkeypatch) -> None:
    monkeypatch.delenv("AWS_AGENTCORE_GATEWAY_URL", raising=False)
    official = SearchResult(
        title="Electrical Intern",
        url="https://www.bhglobal.com.sg/jobs/electrical-intern/",
        snippet="Job Scope: Assist in the design and development of electrical systems",
    )
    monkeypatch.setattr(web_search, "_request_search", lambda *args, **kwargs: [official])

    results = web_search.search_public_web(
        '"BH Global Corporation Ltd" "Electrical Intern" official',
        max_results=8,
    )

    assert results == [official]


def test_agentcore_parser_reads_sse_json_rpc_payload() -> None:
    payload = {
        "jsonrpc": "2.0",
        "result": {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "results": [{
                        "title": "Electrical Intern - BH Global",
                        "url": "https://www.bhglobal.com.sg/jobs/electrical-intern/",
                        "text": "Job scope and requirements",
                    }]
                }),
            }]
        },
    }
    response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text=f"event: message\ndata: {json.dumps(payload)}\n\n",
    )
    assert web_search._agentcore_json_response(response) == payload


def test_agentcore_uses_stateless_mcp_request_metadata(monkeypatch, tmp_path) -> None:
    import boto3
    from botocore.credentials import Credentials

    monkeypatch.setenv("AWS_AGENTCORE_GATEWAY_URL", "https://gateway.example/mcp")
    monkeypatch.setenv("AWS_AGENTCORE_MCP_VERSION", "2026-07-28")
    monkeypatch.setenv("SIMPLYNEXT_AGENTCORE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SIMPLYNEXT_AGENTCORE_MAX_CALLS", "15")
    monkeypatch.setattr(web_search, "_AGENTCORE_NETWORK_CALLS", 0)
    monkeypatch.setattr(web_search, "_AGENTCORE_BUDGET_WARNING_EMITTED", False)
    captured = {}

    class FakeSession:
        def get_credentials(self):
            return Credentials("test-access", "test-secret", "test-session")

    monkeypatch.setattr(boto3, "Session", lambda **kwargs: FakeSession())

    def fake_post(url, *, content, headers, timeout):
        captured.update(url=url, body=json.loads(content), headers=headers, timeout=timeout)
        nested = json.dumps({
            "results": [{
                "title": "Amazon Software Engineer",
                "url": "https://www.amazon.jobs/en/jobs/123/software-engineer",
                "text": "Official role",
            }]
        })
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "result": {"content": [{"type": "text", "text": nested}]}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(web_search.httpx, "post", fake_post)
    results = web_search._search_aws_agentcore("Amazon software engineer careers", 2)

    assert results[0].url == "https://www.amazon.jobs/en/jobs/123/software-engineer"
    assert captured["headers"]["MCP-Protocol-Version"] == "2026-07-28"
    assert captured["headers"]["Mcp-Method"] == "tools/call"
    assert captured["headers"]["Mcp-Name"] == "simplenext-web-search___WebSearch"
    assert captured["body"]["params"]["_meta"]["io.modelcontextprotocol/protocolVersion"] == "2026-07-28"

    monkeypatch.setattr(
        web_search.httpx,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache should avoid AWS")),
    )
    assert web_search._search_aws_agentcore("Amazon software engineer careers", 2) == results


def test_agentcore_request_sends_native_domain_filter(monkeypatch, tmp_path) -> None:
    import boto3
    from botocore.credentials import Credentials

    monkeypatch.setenv("AWS_AGENTCORE_GATEWAY_URL", "https://gateway.example/mcp")
    monkeypatch.setenv("SIMPLYNEXT_AGENTCORE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(web_search, "_AGENTCORE_NETWORK_CALLS", 0)
    captured = {}

    class FakeSession:
        def get_credentials(self):
            return Credentials("access", "secret", "session")

    monkeypatch.setattr(boto3, "Session", lambda **kwargs: FakeSession())

    def fake_post(url, *, content, headers, timeout):
        captured.update(json.loads(content))
        body = {"results": [{
            "title": "Electrical Intern",
            "url": "https://www.bhglobal.com.sg/jobs/electrical-intern/",
            "text": "Job Scope",
        }]}
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "result": {"structuredContent": body}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(web_search.httpx, "post", fake_post)
    rows = web_search._search_aws_agentcore(
        'site:bhglobal.com.sg/jobs "Electrical Intern"', 8, bypass_cache=True
    )
    args = captured["params"]["arguments"]
    assert args["query"] == '"Electrical Intern"'
    assert args["filters"] == {"domainFilter": {"include": ["bhglobal.com.sg"]}}
    assert rows[0].url.endswith("/jobs/electrical-intern/")

def test_agentcore_budget_can_disable_network_calls(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AWS_AGENTCORE_GATEWAY_URL", "https://gateway.example/mcp")
    monkeypatch.setenv("SIMPLYNEXT_AGENTCORE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SIMPLYNEXT_AGENTCORE_MAX_CALLS", "0")
    monkeypatch.setattr(web_search, "_AGENTCORE_NETWORK_CALLS", 0)
    monkeypatch.setattr(web_search, "_AGENTCORE_BUDGET_WARNING_EMITTED", False)
    monkeypatch.setattr(
        web_search.httpx,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("budget should avoid AWS")),
    )

    assert web_search._search_aws_agentcore("Example role", 2) == []


def test_host_canonicalizes_www_prefix() -> None:
    assert host("https://www.reolink.com/careers") == "reolink.com"
    assert host("https://reolink.com/careers") == "reolink.com"
    assert host("https://www.linkedin.com/jobs/view/123") == "linkedin.com"
