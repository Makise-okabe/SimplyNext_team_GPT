from __future__ import annotations

from datetime import date, datetime, timezone

from career_agent import batch_job_research
from career_agent.models.email import EmailMessage
from career_agent.models.inbox import CareerEmailRecord
from career_agent.models.opportunity_research import (
    OpportunityResearchPackage,
    ResearchMetrics,
    SourceProvenance,
)
from career_agent.models.signal import OpportunitySignal


def _package(identity, message_id: str = "m1") -> OpportunityResearchPackage:
    return OpportunityResearchPackage(
        identity=identity,
        record_kind="job_posting",
        status="verified_exact_job",
        confidence="medium",
        basis="official_job_metadata_match",
        provenance=SourceProvenance(
            message_id=message_id,
            subject="Industry Opportunities",
            sender_email="zeli.goh@nus.edu.sg",
        ),
        official_job_url=f"https://careers.example.com/job/{identity.signal_index}",
        application_url=f"https://careers.example.com/job/{identity.signal_index}",
        metrics=ResearchMetrics(search_calls=1, fetch_calls=1, judge_llm_calls=0),
    )


def test_career_email_record_becomes_job_record_without_identity_llm(monkeypatch) -> None:
    email = EmailMessage(
        message_id="m1",
        sender_name="Goh Ze Li",
        sender_email="zeli.goh@nus.edu.sg",
        subject="Industry Opportunities",
        received_at=datetime(2026, 8, 31, tzinfo=timezone.utc),
        body_text="BCG Associate, Singapore (2027)",
    )
    record = CareerEmailRecord(source="goh_ze_li", email=email)

    signal = OpportunitySignal(
        source_type="outlook",
        source_name="Goh Ze Li",
        source_message_id="m1",
        source_date=email.received_at,
        company="THE BOSTON CONSULTING GROUP",
        role_title="Associate, Singapore (2027)",
        location="Singapore",
        opportunity_type="full_time",
        raw_text="THE BOSTON CONSULTING GROUP - Associate, Singapore (2027)",
    )

    monkeypatch.setattr(
        batch_job_research,
        "build_source_corpus",
        lambda email, fetch_linked_pdfs=True: (
            email.body_text,
            [],
            [],
            [],
        ),
    )
    monkeypatch.setattr(
        batch_job_research,
        "extract_all_opportunities",
        lambda **kwargs: (
            [signal],
            type("Metrics", (), {"llm_calls": 1, "source_chars": 35})(),
            [],
        ),
    )

    def fake_research(identity, email):
        assert identity.company == "THE BOSTON CONSULTING GROUP"
        assert identity.title == "Associate, Singapore (2027)"
        assert identity.identity_strength == "moderate"
        return OpportunityResearchPackage(
            identity=identity,
            record_kind="job_posting",
            status="verified_exact_job",
            confidence="medium",
            basis="official_job_metadata_match",
            provenance=SourceProvenance(
                message_id="m1",
                subject="Industry Opportunities",
                sender_email="zeli.goh@nus.edu.sg",
            ),
            official_job_url="https://careers.bcg.com/global/en/job/58603/Associate-Singapore-2027",
            application_url="https://careers.bcg.com/global/en/job/58603/Associate-Singapore-2027",
            metrics=ResearchMetrics(search_calls=1, fetch_calls=1, judge_llm_calls=0),
        )

    monkeypatch.setattr(
        batch_job_research,
        "research_concrete_job_or_delegate",
        fake_research,
    )
    monkeypatch.setattr(
        batch_job_research,
        "_fetch_best_jd",
        lambda package, source_context: (
            "fetched_official",
            package.official_job_url,
            "BCG official full job description",
            [],
            1,
        ),
    )

    result = batch_job_research.research_career_email_record(record)

    assert len(result.job_records) == 1
    job = result.job_records[0]
    assert job.company == "THE BOSTON CONSULTING GROUP"
    assert job.title == "Associate, Singapore (2027)"
    assert job.research_status == "verified_exact_job"
    assert job.jd_status == "fetched_official"
    assert job.jd_text == "BCG official full job description"
    assert result.web_search_calls == 1
    assert result.page_fetch_calls == 2
    assert result.judge_llm_calls == 0


def test_job_limit_researches_subset_but_preserves_full_extraction(monkeypatch) -> None:
    email = EmailMessage(
        message_id="m2",
        sender_name="Goh Ze Li",
        sender_email="zeli.goh@nus.edu.sg",
        subject="Industry Opportunities",
        received_at=datetime(2026, 8, 31, tzinfo=timezone.utc),
        body_text="three jobs",
    )
    record = CareerEmailRecord(source="goh_ze_li", email=email)

    signals = [
        OpportunitySignal(
            source_type="outlook",
            source_name="Goh Ze Li",
            source_message_id="m2",
            company=f"Company {index}",
            role_title=f"Role {index}",
            opportunity_type="full_time",
            raw_text=f"Company {index} | Role {index}",
        )
        for index in range(1, 4)
    ]

    monkeypatch.setattr(
        batch_job_research,
        "build_source_corpus",
        lambda email, fetch_linked_pdfs=True: ("three jobs", [], [], []),
    )
    monkeypatch.setattr(
        batch_job_research,
        "extract_all_opportunities",
        lambda **kwargs: (
            signals,
            type("Metrics", (), {"llm_calls": 0, "source_chars": 10})(),
            [],
        ),
    )

    researched: list[str] = []

    def fake_research(identity, email):
        researched.append(identity.title or "")
        return _package(identity, message_id="m2")

    monkeypatch.setattr(
        batch_job_research,
        "research_concrete_job_or_delegate",
        fake_research,
    )
    monkeypatch.setattr(
        batch_job_research,
        "_fetch_best_jd",
        lambda package, source_context: (
            "fetched_official",
            package.official_job_url,
            "JD",
            [],
            1,
        ),
    )

    result = batch_job_research.research_career_email_record(record, job_limit=2)

    assert len(result.opportunities) == 3
    assert len(result.job_records) == 2
    assert researched == ["Role 1", "Role 2"]
    assert result.company_count == 3


def test_expired_deadline_skips_expensive_research(monkeypatch) -> None:
    email = EmailMessage(
        message_id="m3",
        sender_name="Goh Ze Li",
        sender_email="zeli.goh@nus.edu.sg",
        subject="Industry Opportunities",
        received_at=datetime(2026, 8, 31, tzinfo=timezone.utc),
        body_text="old role",
    )
    record = CareerEmailRecord(source="goh_ze_li", email=email)
    signal = OpportunitySignal(
        source_type="outlook",
        source_name="Goh Ze Li",
        source_message_id="m3",
        company="Old Company",
        role_title="Old Engineer",
        opportunity_type="full_time",
        deadline_hint=date(2025, 12, 28),
        raw_text="Old Company | Old Engineer | Deadline: 28 Dec 2025",
    )

    monkeypatch.setattr(
        batch_job_research,
        "build_source_corpus",
        lambda email, fetch_linked_pdfs=True: ("old role", [], [], []),
    )
    monkeypatch.setattr(
        batch_job_research,
        "extract_all_opportunities",
        lambda **kwargs: (
            [signal],
            type("Metrics", (), {"llm_calls": 0, "source_chars": 8})(),
            [],
        ),
    )

    def should_not_research(*args, **kwargs):
        raise AssertionError("expired role must not trigger web research")

    monkeypatch.setattr(
        batch_job_research,
        "research_concrete_job_or_delegate",
        should_not_research,
    )

    result = batch_job_research.research_career_email_record(record)
    assert len(result.job_records) == 1
    job = result.job_records[0]
    assert job.availability_status == "expired_by_source_deadline"
    assert job.research_skipped_reason is not None
    assert job.jd_status == "unavailable"
    assert result.web_search_calls == 0
    assert result.page_fetch_calls == 0
    assert result.judge_llm_calls == 0


def test_dynamic_official_shell_falls_back_to_source_context(monkeypatch) -> None:
    identity = batch_job_research._identity_from_signal(
        OpportunitySignal(
            source_type="outlook",
            source_name="TalentConnect",
            source_message_id="m4",
            company="Xiaomi",
            role_title="Machine Learning Engineer",
            opportunity_type="full_time",
            raw_text="Xiaomi | Machine Learning Engineer",
        ),
        1,
    )
    package = _package(identity, message_id="m4")

    monkeypatch.setattr(
        batch_job_research,
        "fetch_public_page",
        lambda url: type(
            "Page",
            (),
            {
                "text": "Machine Learning Engineer - Xiaomi",
                "final_url": url,
            },
        )(),
    )

    status, url, text, warnings, fetches = batch_job_research._fetch_best_jd(
        package,
        "Xiaomi | Machine Learning Engineer | trusted NUS source context",
    )

    assert status == "source_context_only"
    assert url is None
    assert "trusted NUS source context" in text
    assert fetches == 1
    assert any("shell/insufficient JD text" in warning for warning in warnings)
