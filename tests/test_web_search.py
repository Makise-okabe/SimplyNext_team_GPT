import base64
from urllib.parse import quote

from career_agent.tools.web_search import (
    _parse_bing_results,
    _parse_lite_results,
    _simplify_query,
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
