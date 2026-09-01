from __future__ import annotations

from datetime import datetime, timezone

from career_agent import company_job_research
from career_agent.catalog_consolidation import consolidate_job_records
from career_agent.company_job_research import ResearchContext, research_company_jobs
from career_agent.job_normalization import (
    CSIT_ROLES,
    clean_company_name,
    clean_role_title,
    expand_known_multi_role_signal,
)
from career_agent.models.email import EmailMessage
from career_agent.models.job_record import JobRecord
from career_agent.models.signal import OpportunitySignal
from career_agent.tools.web_fetch import FetchedPage
from career_agent.tools.web_search import SearchResult


def _email() -> EmailMessage:
    return EmailMessage(
        message_id="m1",
        sender_name="Goh Ze Li",
        sender_email="zeli.goh@nus.edu.sg",
        subject="Industry Opportunities",
        received_at=datetime(2026, 8, 31, tzinfo=timezone.utc),
        body_text="trusted source",
    )


def _signal(company: str, role: str, *, opportunity_type: str = "full_time") -> OpportunitySignal:
    return OpportunitySignal(
        source_type="outlook",
        source_name="Goh Ze Li",
        source_message_id="m1",
        source_date=datetime(2026, 8, 31, tzinfo=timezone.utc),
        company=company,
        role_title=role,
        location="Singapore",
        opportunity_type=opportunity_type,
        raw_text=f"{company} | {role}",
    )


def _jd(company: str, role: str) -> str:
    return (
        f"{company}\n{role}\nSingapore\nResponsibilities\n"
        + ("engineering analysis implementation collaboration systems design. " * 35)
        + "\nRequirements\nDegree and strong communication skills."
    )


def test_known_malformed_identity_cleanup_and_split() -> None:
    assert clean_company_name("EY (more roles on TC)") == "EY"

    title, urls = clean_role_title(
        "Teaching Assistant (Technical) for CDE5311: AI-Powered App Development "
        "<https://inetapps.nus.edu.sg/nsws/app/staff/browse-jobs/view-listing/J2026080109>"
    )
    assert title == "Teaching Assistant (Technical) for CDE5311: AI-Powered App Development"
    assert urls == [
        "https://inetapps.nus.edu.sg/nsws/app/staff/browse-jobs/view-listing/J2026080109"
    ]

    garena, _ = clean_role_title("[Singapore] 2027 Sea Global Management Associate Program (Garena) See")
    assert garena == "[Singapore] 2027 Sea Global Management Associate Program (Garena)"

    csit = _signal(
        "Centre for Strategic Infocomm Technologies",
        "Cyber Security Vulnerability Researcher Cybersecurity Specialist Cyber Threat Researcher "
        "Mobile and Cloud Security Engineer Cybersecurity Software Engineering",
    )
    expanded = expand_known_multi_role_signal(csit)
    assert [item.role_title for item in expanded] == list(CSIT_ROLES)


def test_role_search_fetches_metadata_match_even_without_job_shaped_url(monkeypatch) -> None:
    signal = _signal("Example Robotics", "AI Engineer")
    queries: list[str] = []
    fetched: list[str] = []
    candidate = "https://example-robotics.com/openings/abc123"

    def fake_search(query: str, max_results: int = 10):
        queries.append(query)
        if query == '"Example Robotics" careers jobs Singapore':
            return []
        if '"AI Engineer"' in query and "careers job" in query:
            return [
                SearchResult(
                    title="Example Robotics - AI Engineer",
                    url=candidate,
                    snippet="Singapore AI Engineer opening",
                )
            ]
        return []

    def fake_fetch(url: str, timeout_seconds: float = 8.0):
        fetched.append(url)
        return FetchedPage(
            requested_url=url,
            final_url=url,
            status_code=200,
            title="AI Engineer - Example Robotics",
            text=_jd("Example Robotics", "AI Engineer"),
        )

    monkeypatch.setattr(company_job_research, "search_public_web", fake_search)
    monkeypatch.setattr(company_job_research, "fetch_public_page", fake_fetch)

    outcome = research_company_jobs(
        email=_email(),
        source_key="goh_ze_li",
        company_items=[(1, signal)],
        context=ResearchContext(),
    )

    job = outcome.job_records[0]
    assert candidate in fetched
    assert job.jd_status == "fetched_official"
    assert job.primary_source_url == candidate
    assert not any("Indeed" in query or "Glassdoor" in query or "JobStreet" in query for query in queries)


def _job(
    *,
    source_key: str,
    company: str,
    title: str,
    opportunity_type: str,
    jd_status: str = "unavailable",
    jd_text: str = "",
    official_url: str | None = None,
) -> JobRecord:
    return JobRecord(
        source_key=source_key,
        source_message_id=f"{source_key}-message",
        source_subject="source",
        company=company,
        title=title,
        opportunity_type=opportunity_type,
        record_kind="job_posting",
        research_status="verified_exact_job" if jd_status == "fetched_official" else "source_verified",
        research_confidence="high" if jd_status == "fetched_official" else "medium",
        research_basis="official_company_or_ats_page" if jd_status == "fetched_official" else "trusted_nus_email_web_unresolved",
        primary_source_url=official_url,
        official_job_url=official_url,
        application_url=official_url,
        jd_status=jd_status,
        jd_source_url=official_url,
        jd_text=jd_text,
        source_evidence=f"{company} {title}",
        evidence_summary=[f"seen in {source_key}"],
    )


def test_cross_source_point72_reuses_stronger_official_jd() -> None:
    url = "https://careers.point72.com/CSJobDetail?jobCode=CPA-0014709"
    goh = _job(
        source_key="goh_ze_li",
        company="Point72 Asia (Singapore Pte Ltd)",
        title="2027 Point72 Academy Investment Analyst Summer Internship Program - Singapore",
        opportunity_type="internship",
        jd_status="fetched_official",
        jd_text=_jd("Point72", "Investment Analyst Summer Internship"),
        official_url=url,
    )
    talentconnect = _job(
        source_key="talentconnect",
        company="Point72",
        title="Point72 Academy Investment Analyst Summer Internship",
        opportunity_type="internship",
    )

    consolidated = consolidate_job_records([goh, talentconnect])
    assert len(consolidated) == 1
    job = consolidated[0]
    assert job.jd_status == "fetched_official"
    assert job.official_job_url == url
    assert job.jd_text
    assert any("corroborated across sources" in item for item in job.evidence_summary)
