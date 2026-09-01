from __future__ import annotations

from datetime import datetime, timezone

from career_agent.all_job_extraction import ExtractionMetrics
from career_agent.batch_sources import TABLE_END, TABLE_START
from career_agent.goh_extraction import extract_goh_opportunities
from career_agent.job_catalog_pipeline import (
    _company_key,
    _is_generic_talentconnect_seed,
    _sanitize_signal_source_urls,
)
from career_agent.job_research_quality import is_plausible_official_url
from career_agent.models.signal import OpportunitySignal
from career_agent.tools import web_search
from career_agent.tools.web_search import SearchResult


def _signal(company: str, title: str, urls=None) -> OpportunitySignal:
    return OpportunitySignal(
        source_type="outlook",
        source_name="NUS",
        source_message_id="m1",
        source_date=datetime(2026, 8, 31, tzinfo=timezone.utc),
        company=company,
        role_title=title,
        location="Singapore",
        opportunity_type="full_time",
        urls=urls or [],
        raw_text=f"{company} | {title}",
        resolution_status="unresolved",
    )


def test_generic_ats_url_must_match_current_company() -> None:
    keppel = (
        "https://keppel.wd3.myworkdayjobs.com/KeppelCareers/job/Singapore/"
        "Keppel-Associate-Programme-2027_10016398"
    )
    assert is_plausible_official_url(keppel, "Keppel Ltd")
    assert not is_plausible_official_url(keppel, "Ernst & Young Solutions LLP")


def test_goh_structured_row_does_not_inherit_global_wrong_company_url() -> None:
    wrong = "https://employmenthero.com/sg/jobs/position/transcelestial-software-engineer/"
    corpus = f"""SOURCE: EMAIL
JOBS
{TABLE_START}
INDUSTRY | COMPANY | ROLE | TC ID | REMARKS
ICT | Reolink | AI Engineer | 6a123456789012001d123456 | Deadline: 3 Nov 2026
{TABLE_END}
"""

    def fake_base_extractor(**kwargs):
        return (
            [_signal("Reolink", "AI Engineer", [wrong])],
            ExtractionMetrics(llm_calls=0, source_chars=len(corpus)),
            [],
        )

    signals, _, _ = extract_goh_opportunities(
        source_name="Goh Ze Li",
        source_message_id="m1",
        source_date=datetime(2026, 8, 31, tzinfo=timezone.utc),
        corpus=corpus,
        base_extractor=fake_base_extractor,
    )

    match = next(item for item in signals if item.company == "Reolink" and item.role_title == "AI Engineer")
    assert match.urls == []


def test_pipeline_drops_cross_company_concrete_source_url() -> None:
    wrong = "https://employmenthero.com/sg/jobs/position/transcelestial-technologies-software-engineer-space/"
    cleaned = _sanitize_signal_source_urls(_signal("Reolink", "AI Engineer", [wrong]))
    assert cleaned.urls == []


def test_pipeline_keeps_company_matched_official_source_url() -> None:
    point72 = "https://careers.point72.com/CSJobDetail?jobCode=CPA-0014976"
    cleaned = _sanitize_signal_source_urls(
        _signal("Point72 Asia (Singapore Pte Ltd)", "Investment Analyst", [point72])
    )
    assert cleaned.urls == [point72]


def test_company_key_collapses_obvious_brand_and_legal_name_duplicates() -> None:
    assert _company_key(_signal("P&G", "Data Science Intern")) == _company_key(
        _signal("Procter & Gamble", "Data Science Intern")
    )
    assert _company_key(_signal("Watson's", "Marketing Intern")) == _company_key(
        _signal("Watsons", "Marketing Intern")
    )
    assert _company_key(_signal("Deutsche Bank AG", "Internship Programme")) == _company_key(
        _signal("Deutsche Bank", "Internship Programme")
    )


def test_generic_talentconnect_lead_is_not_a_concrete_job() -> None:
    assert _is_generic_talentconnect_seed(_signal("Mastercard", "Career opportunities"))
    assert _is_generic_talentconnect_seed(_signal("Ericsson", "Internship opportunities"))
    assert not _is_generic_talentconnect_seed(
        _signal("Point72", "Point72 Academy Investment Analyst Summer Internship")
    )


def test_search_provider_failure_falls_through_to_next_provider(monkeypatch) -> None:
    calls: list[str] = []

    def fake_request(url, query, *, parser, max_results, headers):
        calls.append(url)
        if url == web_search.BING_URL:
            raise RuntimeError("provider blocked")
        if url == web_search.DUCKDUCKGO_HTML_URL:
            return [
                SearchResult(
                    title="Example Robotics AI Engineer",
                    url="https://careers.examplerobotics.com/job/ai-engineer",
                    snippet="Singapore",
                )
            ]
        return []

    monkeypatch.setattr(web_search, "_request_search", fake_request)
    results = web_search.search_public_web('"Example Robotics" "AI Engineer" Singapore')

    assert len(results) == 1
    assert results[0].url.endswith("/job/ai-engineer")
    assert calls[:2] == [web_search.BING_URL, web_search.DUCKDUCKGO_HTML_URL]
