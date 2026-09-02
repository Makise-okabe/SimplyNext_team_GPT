from datetime import datetime, timezone

from career_agent import company_job_research
from career_agent.company_job_research import ResearchContext, research_company_jobs
from career_agent.models.email import EmailMessage
from career_agent.models.signal import OpportunitySignal
from career_agent.tools.web_fetch import FetchedPage
from career_agent.tools.web_search import SearchResult


def _email() -> EmailMessage:
    return EmailMessage(
        message_id="m-fast-path",
        sender_name="Goh Ze Li",
        sender_email="zeli.goh@nus.edu.sg",
        subject="Industry Opportunities",
        received_at=datetime(2026, 8, 31, tzinfo=timezone.utc),
        body_text="trusted source",
    )


def _signal(role: str) -> OpportunitySignal:
    return OpportunitySignal(
        source_type="outlook",
        source_name="Goh Ze Li",
        source_message_id="m-fast-path",
        source_date=datetime(2026, 8, 31, tzinfo=timezone.utc),
        company="Reolink",
        role_title=role,
        location="Singapore",
        opportunity_type="full_time",
        raw_text=f"Reolink | {role}",
    )


def _jd(role: str) -> str:
    return (
        f"Reolink\n{role}\nSingapore\nResponsibilities\n"
        + ("Build reliable services and improve production infrastructure. " * 30)
        + "\nRequirements\nDegree in computing or engineering and strong communication skills."
    )


def test_exact_title_search_can_resolve_official_ats_without_greenhouse_discovery(monkeypatch) -> None:
    circulated = _signal("Site Reliability Engineer")
    exact_url = "https://job-boards.greenhouse.io/reolink/jobs/123"
    queries: list[str] = []

    def fake_search(query: str, max_results: int = 8):
        queries.append(query)
        return [
            SearchResult(
                title="Site Reliability Engineer (SRE) - Reolink",
                url=exact_url,
                snippet="Reolink Site Reliability Engineer Singapore careers",
            )
        ]

    monkeypatch.setattr(company_job_research, "search_public_web", fake_search)
    monkeypatch.setattr(
        company_job_research,
        "fetch_public_page",
        lambda url, timeout_seconds=8.0: FetchedPage(
            requested_url=url,
            final_url=url,
            status_code=200,
            title="Site Reliability Engineer (SRE) - Reolink",
            text=_jd("Site Reliability Engineer (SRE)"),
        ),
    )

    outcome = research_company_jobs(
        email=_email(),
        source_key="goh_ze_li",
        company_items=[(1, circulated)],
        context=ResearchContext(),
    )

    job = outcome.job_records[0]
    assert job.jd_status == "fetched_official"
    assert job.primary_source_url == exact_url
    assert job.jd_source_url == exact_url
    assert outcome.search_calls == 1
    assert outcome.fetch_calls == 1
    assert queries == ['"Reolink" "Site Reliability Engineer" careers job']
