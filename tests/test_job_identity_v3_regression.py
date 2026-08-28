from datetime import datetime, timezone

from career_agent.job_identity.discover_candidates import _classify_url
from career_agent.job_identity.verify_same_job import _evaluate_page, verify_same_job
from career_agent.models.email import EmailMessage
from career_agent.models.job_identity import JobIdentity
from career_agent.models.job_search import CandidateDiscoveryResult, SearchCandidate
from career_agent.tools.web_fetch import FetchedPage


def _mckinsey_identity() -> JobIdentity:
    return JobIdentity(
        source_message_id="mckinsey-2026",
        signal_index=1,
        company="McKinsey & Company",
        title="Innovation and Learning Centre (ILC) Intern",
        location="Singapore",
        opportunity_type="internship",
        business_unit="Innovation and Learning Centre (ILC)",
        duration="6-month",
        start_period="late June/July",
        end_period="December",
        distinctive_phrases=[
            "Innovation and Learning Centre (ILC)",
            "Advanced Remanufacturing and Technology Center (ARTC)",
            "Supply Chain Control Tower",
        ],
        direct_urls=["https://forms.office.com/r/example"],
        identity_strength="strong",
        source_fingerprint="abc",
    )


def test_careeraxis_is_secondary_source_not_official_employer_host() -> None:
    assert _classify_url("https://careeraxis.ntu.edu.sg/Form.aspx?id=761988") == "source_page"


def test_2025_cycle_is_hard_reject_for_2026_source() -> None:
    identity = _mckinsey_identity()
    candidate = SearchCandidate(
        url="https://careeraxis.ntu.edu.sg/Form.aspx?id=761988",
        host="careeraxis.ntu.edu.sg",
        url_kind="source_page",
        discovery_score=47,
        strategies=["metadata"],
    )
    page = FetchedPage(
        requested_url=candidate.url,
        final_url=candidate.url,
        status_code=200,
        title="McKinsey's Innovation and Learning Centre (ILC) Internship (2H 2025)",
        text="McKinsey Innovation and Learning Centre internship.",
    )

    evaluation = _evaluate_page(identity, candidate, page, source_year=2026)

    assert evaluation.decision == "reject"
    assert any("recruiting-cycle conflict" in item for item in evaluation.hard_conflicts)


def test_stale_secondary_pages_cannot_be_promoted_by_llm_path(monkeypatch) -> None:
    identity = _mckinsey_identity()
    stale_url = "https://careeraxis.ntu.edu.sg/Form.aspx?id=761988"
    discovery = CandidateDiscoveryResult(
        candidates=[
            SearchCandidate(
                url=stale_url,
                host="careeraxis.ntu.edu.sg",
                url_kind="source_page",
                discovery_score=47,
                strategies=["metadata"],
            ),
            SearchCandidate(
                url="https://forms.office.com/r/example",
                host="forms.office.com",
                url_kind="application_form",
                discovery_score=18,
                strategies=["direct_url"],
            ),
        ]
    )
    page = FetchedPage(
        requested_url=stale_url,
        final_url=stale_url,
        status_code=200,
        title="McKinsey's Innovation and Learning Centre (ILC) Internship (2H 2025)",
        text="McKinsey Innovation and Learning Centre internship.",
    )

    monkeypatch.setattr(
        "career_agent.job_identity.verify_same_job._fetch_candidates",
        lambda candidates: ({stale_url: page}, [], [], 1),
    )

    email = EmailMessage(
        message_id="mckinsey-2026",
        sender_email="zeli.goh@nus.edu.sg",
        subject="[By 30 March] 6-month Internship Opportunity: McKinsey ILC Singapore",
        received_at=datetime(2026, 3, 27, tzinfo=timezone.utc),
        attachment_text=(
            "McKinsey & Company Innovation and Learning Centre (ILC) Intern Singapore. "
            "Advanced Remanufacturing and Technology Center (ARTC). "
            "Supply Chain Control Tower."
        ),
    )

    result = verify_same_job(identity, discovery, email, enable_llm_judge=True)

    assert result.identity_status == "source_verified"
    assert result.identity_basis == "trusted_nus_attachment"
    assert result.official_url is None
    assert result.application_url == "https://forms.office.com/r/example"
    assert result.metrics.llm_calls == 0
    assert any("recruiting-cycle conflict" in item for item in result.conflicts)


def test_fetch_unavailability_is_warning_not_pipeline_error(monkeypatch) -> None:
    identity = _mckinsey_identity()
    employer_url = "https://www.mckinsey.com/example"
    discovery = CandidateDiscoveryResult(
        candidates=[
            SearchCandidate(
                url=employer_url,
                host="www.mckinsey.com",
                url_kind="unknown",
                discovery_score=30,
                strategies=["metadata"],
            )
        ]
    )

    monkeypatch.setattr(
        "career_agent.job_identity.verify_same_job._fetch_candidates",
        lambda candidates: (
            {},
            [
                __import__("career_agent.models.job_verification", fromlist=["CandidateEvaluation"]).CandidateEvaluation(
                    requested_url=employer_url,
                    host="www.mckinsey.com",
                    url_kind="unknown",
                    decision="unreadable",
                    fetch_error="candidate page unavailable: timeout",
                )
            ],
            ["candidate page unavailable: timeout"],
            1,
        ),
    )

    email = EmailMessage(
        message_id="mckinsey-2026",
        sender_email="random@example.com",
        subject="McKinsey role",
        received_at=datetime(2026, 3, 27, tzinfo=timezone.utc),
    )

    result = verify_same_job(identity, discovery, email, enable_llm_judge=False)

    assert result.errors == []
    assert result.warnings == ["candidate page unavailable: timeout"]
