from __future__ import annotations

from datetime import datetime, timezone

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
