from datetime import datetime, timezone

from career_agent import company_job_research
from career_agent.company_job_research import ResearchContext, research_company_jobs
from career_agent.models.email import EmailMessage
from career_agent.models.signal import OpportunitySignal
from career_agent.tools.greenhouse import GreenhouseJob
from career_agent.tools.web_fetch import FetchedPage
from career_agent.tools.web_search import SearchResult


def _email() -> EmailMessage:
    return EmailMessage(
        message_id="m-greenhouse",
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
        source_message_id="m-greenhouse",
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


def test_greenhouse_fallback_resolves_only_circulated_role(monkeypatch) -> None:
    circulated = _signal("Site Reliability Engineer")
    exact_url = "https://job-boards.greenhouse.io/reolink/jobs/123"
    queries: list[str] = []

    def fake_search(query: str, max_results: int = 10):
        queries.append(query)
        if "greenhouse" in query.lower():
            return [
                SearchResult(
                    title="Reolink Jobs",
                    url="https://job-boards.greenhouse.io/reolink",
                    snippet="Reolink careers and open positions",
                )
            ]
        return []

    monkeypatch.setattr(company_job_research, "search_public_web", fake_search)
    monkeypatch.setattr(
        company_job_research,
        "fetch_greenhouse_jobs",
        lambda board_slug, timeout_seconds=8.0: [
            GreenhouseJob(
                title="Site Reliability Engineer (SRE)",
                url=exact_url,
                location="Singapore",
            ),
            GreenhouseJob(
                title="Unrelated Product Designer",
                url="https://job-boards.greenhouse.io/reolink/jobs/999",
                location="Singapore",
            ),
        ],
    )
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
    assert outcome.fetch_calls == 1
    assert any("greenhouse" in query.lower() for query in queries)
    assert all("999" not in warning for warning in job.warnings)
