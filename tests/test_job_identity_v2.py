from career_agent.job_identity.discover_candidates import (
    _classify_url,
    _merge_candidate,
    _score_result,
    build_progressive_queries,
)
from career_agent.models.job_identity import JobIdentifier, JobIdentity
from career_agent.models.job_search import SearchCandidate
from career_agent.tools.web_search import SearchResult


def _identity(**overrides) -> JobIdentity:
    payload = {
        "source_message_id": "m1",
        "signal_index": 1,
        "company": "McKinsey & Company",
        "title": "Innovation and Learning Centre (ILC) Intern",
        "identifiers": [],
        "location": "Singapore",
        "opportunity_type": "internship",
        "business_unit": "Innovation and Learning Centre (ILC)",
        "team": "ILC team",
        "employment_type": "internship",
        "duration": "6-month",
        "start_period": "late June/July",
        "end_period": "December",
        "target_cohort": ["Year 4 Engineering"],
        "distinctive_phrases": [
            "Supply Chain Control Tower",
            "digital war room",
            "Advanced Remanufacturing and Technology Center (ARTC)",
        ],
        "direct_urls": [],
        "evidence_snippets": [],
        "identity_strength": "strong",
        "source_fingerprint": "abc123",
    }
    payload.update(overrides)
    return JobIdentity(**payload)


def test_progressive_queries_prioritize_exact_identifier() -> None:
    identity = _identity(
        identifiers=[
            JobIdentifier(
                kind="requisition_id",
                label="Requisition ID",
                value="JR-778899",
            )
        ]
    )

    queries = build_progressive_queries(identity)

    assert queries[0][0] == "exact_identifier"
    assert '"JR-778899"' in queries[0][1]
    assert queries[1][0] == "metadata"
    assert len(queries) <= 3


def test_no_identifier_starts_with_metadata_then_distinctive_phrase() -> None:
    queries = build_progressive_queries(_identity())

    assert queries[0][0] == "metadata"
    assert any(strategy == "distinctive_phrase" for strategy, _ in queries)


def test_url_classifier_separates_application_form_and_ats() -> None:
    assert _classify_url("https://forms.office.com/r/abc") == "application_form"
    assert (
        _classify_url("https://company.wd3.myworkdayjobs.com/job/123")
        == "employer_or_ats"
    )
    assert _classify_url("https://sg.indeed.com/viewjob?jk=123") == "aggregator"


def test_search_result_with_identifier_scores_very_high() -> None:
    identity = _identity(
        identifiers=[
            JobIdentifier(kind="job_id", label="Job ID", value="JR123456")
        ]
    )
    result = SearchResult(
        title="McKinsey ILC Intern JR123456",
        url="https://careers.example.com/jobs/JR123456",
        snippet=(
            "McKinsey & Company Innovation and Learning Centre Intern in Singapore"
        ),
    )

    candidate = _score_result(identity, result, "exact_identifier")

    assert "JR123456" in candidate.identifier_hits
    assert candidate.discovery_score >= 65
    assert "company" in candidate.metadata_hits
    assert "title" in candidate.metadata_hits


def test_distinctive_phrase_in_result_adds_evidence() -> None:
    identity = _identity()
    result = SearchResult(
        title="Innovation and Learning Centre internship",
        url="https://example.com/role",
        snippet="Work with a Supply Chain Control Tower and digital war room in Singapore.",
    )

    candidate = _score_result(identity, result, "distinctive_phrase")

    assert "Supply Chain Control Tower" in candidate.distinctive_phrase_hits
    assert "digital war room" in candidate.distinctive_phrase_hits
    assert candidate.discovery_score > 0


def test_merge_preserves_multiple_discovery_strategies() -> None:
    first = SearchCandidate(
        url="https://example.com/job",
        host="example.com",
        discovery_score=40,
        strategies=["metadata"],
        metadata_hits=["company"],
    )
    second = SearchCandidate(
        url="https://example.com/job",
        host="example.com",
        discovery_score=70,
        strategies=["distinctive_phrase"],
        distinctive_phrase_hits=["Supply Chain Control Tower"],
    )

    merged = _merge_candidate(first, second)

    assert merged.discovery_score == 70
    assert merged.strategies == ["metadata", "distinctive_phrase"]
    assert merged.metadata_hits == ["company"]
    assert merged.distinctive_phrase_hits == ["Supply Chain Control Tower"]
