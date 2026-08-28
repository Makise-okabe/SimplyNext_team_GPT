from career_agent.tools.web_search import (
    _parse_lite_results,
    _simplify_query,
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
