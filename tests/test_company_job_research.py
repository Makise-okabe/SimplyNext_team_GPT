from __future__ import annotations

from datetime import datetime, timezone

from career_agent import company_job_research
from career_agent.company_job_research import ResearchContext, research_company_jobs
from career_agent.job_research_quality import is_plausible_official_url
from career_agent.matching_dataset import is_matching_ready
from career_agent.models.email import EmailMessage
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


def _signal(company: str, role: str, urls=None) -> OpportunitySignal:
    return OpportunitySignal(
        source_type="outlook",
        source_name="Goh Ze Li",
        source_message_id="m1",
        source_date=datetime(2026, 8, 31, tzinfo=timezone.utc),
        company=company,
        role_title=role,
        location="Singapore",
        opportunity_type="full_time",
        urls=urls or [],
        raw_text=f"{company} | {role}",
    )


def _jd(role: str, company: str = "Ley Choon") -> str:
    return (
        f"{company}\n{role}\nSingapore\nResponsibilities\n"
        + ("Design engineering operations project delivery stakeholder coordination. " * 30)
        + "\nRequirements\nDegree in engineering and strong communication skills."
    )


def test_company_aliases_recognize_bcg_and_ey_official_domains() -> None:
    assert is_plausible_official_url(
        "https://careers.bcg.com/job/123",
        "THE BOSTON CONSULTING GROUP",
    )
    assert is_plausible_official_url(
        "https://careers.ey.com/job/456",
        "Ernst & Young Singapore (EY)",
    )


def test_direct_official_url_uses_zero_searches_and_stops_before_secondary(monkeypatch) -> None:
    official = "https://careers.point72.com/CSJobDetail?jobCode=CPA-0014976"
    signal = _signal(
        "Point72",
        "Point72 Academy Investment Analyst Program for Upcoming Graduates 2027 SG",
        [official],
    )

    def should_not_search(*args, **kwargs):
        raise AssertionError("direct official URL must not trigger web search")

    monkeypatch.setattr(company_job_research, "search_public_web", should_not_search)
    monkeypatch.setattr(
        company_job_research,
        "fetch_public_page",
        lambda url, timeout_seconds=8.0: FetchedPage(
            requested_url=url,
            final_url=url,
            status_code=200,
            title=signal.role_title,
            text=_jd(signal.role_title, "Point72"),
        ),
    )

    context = ResearchContext()
    outcome = research_company_jobs(
        email=_email(),
        source_key="goh_ze_li",
        company_items=[(1, signal)],
        context=context,
    )

    job = outcome.job_records[0]
    assert outcome.search_calls == 0
    assert outcome.fetch_calls == 1
    assert job.primary_source_url == official
    assert job.secondary_source_url is None
    assert job.jd_status == "fetched_official"
    assert is_matching_ready(job)


def test_same_company_roles_share_company_discovery_and_batch_official_search(monkeypatch) -> None:
    signals = [
        _signal("Ley Choon", "Management Associate"),
        _signal("Ley Choon", "Engineer Associate"),
        _signal("Ley Choon", "EHS Officer"),
    ]
    queries: list[str] = []

    def fake_search(query: str, max_results: int = 10):
        queries.append(query)
        if query == '"Ley Choon" careers jobs Singapore':
            return [
                SearchResult(
                    title="Ley Choon Careers",
                    url="https://careers.leychoon.com/jobs",
                    snippet="Ley Choon careers and vacancies",
                )
            ]
        if query.startswith("site:careers.leychoon.com") and " OR " in query:
            return [
                SearchResult(
                    title=signal.role_title,
                    url=f"https://careers.leychoon.com/job/{index}",
                    snippet=f"Ley Choon {signal.role_title} Singapore",
                )
                for index, signal in enumerate(signals, start=1)
            ]
        return []

    def fake_fetch(url: str, timeout_seconds: float = 8.0):
        tail = url.rsplit("/", 1)[-1]
        if tail == "jobs":
            return FetchedPage(
                requested_url=url,
                final_url=url,
                status_code=200,
                title="Ley Choon Careers",
                text="Ley Choon careers homepage",
            )
        index = int(tail) - 1
        signal = signals[index]
        return FetchedPage(
            requested_url=url,
            final_url=url,
            status_code=200,
            title=signal.role_title,
            text=_jd(signal.role_title),
        )

    monkeypatch.setattr(company_job_research, "search_public_web", fake_search)
    monkeypatch.setattr(company_job_research, "fetch_public_page", fake_fetch)

    outcome = research_company_jobs(
        email=_email(),
        source_key="goh_ze_li",
        company_items=list(enumerate(signals, start=1)),
        context=ResearchContext(),
    )

    assert len(outcome.job_records) == 3
    assert all(job.jd_status == "fetched_official" for job in outcome.job_records)
    assert queries.count('"Ley Choon" careers jobs Singapore') == 1
    assert sum(1 for query in queries if " OR " in query) == 1
    assert not any("linkedin.com/jobs" in query.lower() for query in queries)
    assert outcome.search_calls == 2
    # One cached careers-homepage probe + one exact page per role.
    assert outcome.fetch_calls == 4


def test_linkedin_runs_only_after_official_phase_does_not_produce_jd(monkeypatch) -> None:
    signal = _signal("Example Robotics", "AI Engineer")
    queries: list[str] = []

    def fake_search(query: str, max_results: int = 10):
        queries.append(query)
        if query == '"Example Robotics" careers jobs Singapore':
            return [
                SearchResult(
                    title="Example Robotics Careers",
                    url="https://careers.examplerobotics.com/jobs",
                    snippet="Example Robotics careers",
                )
            ]
        if query.startswith("site:careers.examplerobotics.com"):
            return [
                SearchResult(
                    title="AI Engineer",
                    url="https://careers.examplerobotics.com/job/ai-engineer",
                    snippet="Example Robotics AI Engineer Singapore",
                )
            ]
        if "site:linkedin.com/jobs" in query:
            return [
                SearchResult(
                    title="AI Engineer - Example Robotics",
                    url="https://www.linkedin.com/jobs/view/123",
                    snippet="Example Robotics AI Engineer Singapore",
                )
            ]
        return []

    def fake_fetch(url: str, timeout_seconds: float = 8.0):
        if "linkedin.com" in url:
            return FetchedPage(
                requested_url=url,
                final_url=url,
                status_code=200,
                title="AI Engineer - Example Robotics",
                text=_jd("AI Engineer", "Example Robotics"),
            )
        return FetchedPage(
            requested_url=url,
            final_url=url,
            status_code=200,
            title="AI Engineer",
            text="Example Robotics AI Engineer",
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
    assert job.primary_source_url == "https://careers.examplerobotics.com/job/ai-engineer"
    assert job.secondary_source_url == "https://www.linkedin.com/jobs/view/123"
    assert job.jd_status == "fetched_secondary"
    assert job.jd_source_url == job.secondary_source_url
    assert any("site:linkedin.com/jobs" in query for query in queries)


def test_official_closed_page_blocks_secondary_and_matching(monkeypatch) -> None:
    official = "https://careers.example.com/job/closed-role"
    signal = _signal("Example", "Graduate Engineer", [official])
    queries: list[str] = []

    monkeypatch.setattr(
        company_job_research,
        "search_public_web",
        lambda query, max_results=10: queries.append(query) or [],
    )
    monkeypatch.setattr(
        company_job_research,
        "fetch_public_page",
        lambda url, timeout_seconds=8.0: FetchedPage(
            requested_url=url,
            final_url=url,
            status_code=200,
            title="Graduate Engineer",
            text="Example Graduate Engineer. No longer accepting applications.",
        ),
    )

    outcome = research_company_jobs(
        email=_email(),
        source_key="goh_ze_li",
        company_items=[(1, signal)],
        context=ResearchContext(),
    )
    job = outcome.job_records[0]

    assert job.availability_status == "closed_by_official"
    assert job.jd_status == "unavailable"
    assert not is_matching_ready(job)
    assert queries == []


def test_page_cache_fetches_same_url_only_once(monkeypatch) -> None:
    calls: list[str] = []

    def fake_fetch(url: str, timeout_seconds: float = 8.0):
        calls.append(url)
        return FetchedPage(
            requested_url=url,
            final_url=url,
            status_code=200,
            title="Role",
            text=_jd("Role", "Company"),
        )

    monkeypatch.setattr(company_job_research, "fetch_public_page", fake_fetch)
    context = ResearchContext()
    url = "https://careers.company.com/job/1"
    assert context.fetch(url) is not None
    assert context.fetch(url) is not None
    assert calls == [url]
    assert context.fetch_calls == 1
