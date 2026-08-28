from career_agent.job_identity.verify_same_job import (
    _evaluate_page,
    verify_same_job,
)
from career_agent.models.email import EmailMessage
from career_agent.models.job_identity import JobIdentifier, JobIdentity
from career_agent.models.job_search import CandidateDiscoveryResult, SearchCandidate
from career_agent.tools.web_fetch import FetchedPage


def _identity(**updates) -> JobIdentity:
    payload = {
        "source_message_id": "m1",
        "signal_index": 1,
        "company": "Example Corp",
        "title": "Silicon Validation Intern",
        "location": "Singapore",
        "opportunity_type": "internship",
        "business_unit": "Advanced Products Lab",
        "distinctive_phrases": [
            "high-speed SerDes validation platform",
            "wafer-level debug workflow",
        ],
        "direct_urls": [],
        "identity_strength": "strong",
        "source_fingerprint": "abc",
    }
    payload.update(updates)
    return JobIdentity(**payload)


def _candidate(url: str = "https://careers.example.com/job/123", **updates) -> SearchCandidate:
    payload = {
        "url": url,
        "host": "careers.example.com",
        "url_kind": "employer_or_ats",
        "discovery_score": 70,
        "strategies": ["metadata"],
    }
    payload.update(updates)
    return SearchCandidate(**payload)


def _page(text: str, url: str = "https://careers.example.com/job/123") -> FetchedPage:
    return FetchedPage(
        requested_url=url,
        final_url=url,
        status_code=200,
        title="Silicon Validation Intern | Example Corp",
        text=text,
    )


def test_exact_identifier_is_high_confidence_same_job() -> None:
    identity = _identity(
        identifiers=[
            JobIdentifier(kind="requisition_id", label="Requisition ID", value="R-7788")
        ]
    )
    candidate = _candidate()
    page = _page(
        "Example Corp Singapore Advanced Products Lab. Requisition ID: R-7788. "
        "Silicon Validation Intern using high-speed SerDes validation platform."
    )

    evaluation = _evaluate_page(identity, candidate, page)

    assert evaluation.decision == "same_job"
    assert evaluation.confidence == "high"
    assert "R-7788" in evaluation.identifier_hits


def test_conflicting_identifier_rejects_candidate() -> None:
    identity = _identity(
        identifiers=[
            JobIdentifier(kind="requisition_id", label="Requisition ID", value="R-7788")
        ]
    )
    candidate = _candidate()
    page = _page(
        "Example Corp Silicon Validation Intern Singapore. Requisition ID: R-9999."
    )

    evaluation = _evaluate_page(identity, candidate, page)

    assert evaluation.decision == "reject"
    assert evaluation.hard_conflicts


def test_content_fingerprint_can_prove_same_job_without_id() -> None:
    identity = _identity()
    candidate = _candidate()
    page = _page(
        "Example Corp is hiring a Silicon Validation Intern in Singapore for the "
        "Advanced Products Lab. Work includes our high-speed SerDes validation platform "
        "and wafer-level debug workflow."
    )

    evaluation = _evaluate_page(identity, candidate, page)

    assert evaluation.decision == "same_job"
    assert len(evaluation.distinctive_phrase_hits) == 2
    assert evaluation.business_unit_match is True


def test_same_title_without_identity_fingerprint_is_not_verified() -> None:
    identity = _identity()
    candidate = _candidate(url="https://jobs.example.net/other")
    page = FetchedPage(
        requested_url=candidate.url,
        final_url=candidate.url,
        status_code=200,
        title="Silicon Validation Intern",
        text="Example Corp has a Silicon Validation Intern opening in Singapore.",
    )

    evaluation = _evaluate_page(identity, candidate, page)

    assert evaluation.decision != "same_job"


def test_application_form_never_becomes_official_match() -> None:
    identity = _identity(company="McKinsey & Company", title="ILC Intern")
    discovery = CandidateDiscoveryResult(
        candidates=[
            SearchCandidate(
                url="https://forms.office.com/r/example",
                host="forms.office.com",
                url_kind="application_form",
                discovery_score=18,
                strategies=["direct_url"],
            )
        ]
    )
    email = EmailMessage(
        message_id="m",
        sender_email="zeli.goh@nus.edu.sg",
        subject="McKinsey ILC Intern",
        body_text="McKinsey & Company ILC Intern",
        attachment_text=(
            "McKinsey & Company ILC Intern. Innovation and Learning Centre ILC."
        ),
    )

    result = verify_same_job(identity, discovery, email, enable_llm_judge=False)

    assert result.official_url is None
    assert result.application_url == "https://forms.office.com/r/example"


def test_trusted_attachment_falls_back_to_source_verified() -> None:
    identity = _identity()
    discovery = CandidateDiscoveryResult(candidates=[])
    email = EmailMessage(
        message_id="m",
        sender_email="zeli.goh@nus.edu.sg",
        subject="Role",
        attachment_text=(
            "Example Corp Silicon Validation Intern Singapore Advanced Products Lab. "
            "high-speed SerDes validation platform."
        ),
    )

    result = verify_same_job(identity, discovery, email, enable_llm_judge=False)

    assert result.identity_status == "source_verified"
    assert result.identity_basis == "trusted_nus_attachment"


def test_untrusted_attachment_cannot_source_verify() -> None:
    identity = _identity()
    discovery = CandidateDiscoveryResult(candidates=[])
    email = EmailMessage(
        message_id="m",
        sender_email="random@example.com",
        subject="Role",
        attachment_text=(
            "Example Corp Silicon Validation Intern Singapore Advanced Products Lab. "
            "high-speed SerDes validation platform."
        ),
    )

    result = verify_same_job(identity, discovery, email, enable_llm_judge=False)

    assert result.identity_status == "unresolved"
