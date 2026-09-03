from career_agent import job_link_resolver
from career_agent.job_link_resolver import resolve_job_link
from career_agent.models.job_record import JobRecord
from career_agent.tools.web_search import SearchResult


def _job():
    return JobRecord(
        source_key="goh_ze_li",
        source_message_id="m1",
        source_subject="Career opportunities",
        company="Reolink",
        title="AI Engineer",
        opportunity_type="full_time",
        availability_status="active_candidate",
        record_kind="job_posting",
        source_evidence="Reolink | AI Engineer",
    )


def test_resolver_keeps_clickable_secondary_page_without_jd(monkeypatch):
    monkeypatch.setattr(
        job_link_resolver,
        "search_public_web_aggregated",
        lambda query, max_results=16, min_results=8: [
            SearchResult(
                title="AI Engineer - Reolink",
                url="https://www.linkedin.com/jobs/view/123456",
                snippet="Reolink AI Engineer Singapore",
            )
        ],
    )

    resolved, result = resolve_job_link(_job())
    assert result.url == "https://www.linkedin.com/jobs/view/123456"
    assert result.kind == "secondary_exact"
    assert resolved.job_page_url == result.url
    assert resolved.secondary_source_url == result.url
    assert resolved.application_url == result.url
    assert resolved.official_job_url is None


def test_resolver_prefers_official_exact_over_secondary(monkeypatch):
    monkeypatch.setattr(
        job_link_resolver,
        "search_public_web_aggregated",
        lambda query, max_results=16, min_results=8: [
            SearchResult(
                title="AI Engineer - Reolink",
                url="https://www.linkedin.com/jobs/view/123456",
                snippet="Reolink AI Engineer",
            ),
            SearchResult(
                title="AI Engineer - Reolink Careers",
                url="https://reolink.com/careers/jobs/ai-engineer",
                snippet="Reolink AI Engineer careers",
            ),
        ],
    )

    resolved, result = resolve_job_link(_job())
    assert result.kind == "official_exact"
    assert resolved.official_job_url == "https://reolink.com/careers/jobs/ai-engineer"
    assert resolved.job_page_confidence == "high"
