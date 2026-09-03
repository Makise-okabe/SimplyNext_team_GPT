from career_agent.matching_dataset import (
    is_matching_ready,
    matching_evidence_level,
    matching_input_text,
    sanitize_job_sources,
)
from career_agent.models.job_record import JobRecord


def _job(**updates) -> JobRecord:
    base = dict(
        source_message_id="m1",
        source_subject="Career email",
        company="Example Corp",
        title="Engineer",
        availability_status="unknown",
        jd_status="fetched_secondary",
        jd_source_url="https://www.linkedin.com/jobs/view/123",
        jd_text="Engineer responsibilities and requirements " * 30,
    )
    base.update(updates)
    return JobRecord(**base)


def test_aggregator_primary_is_not_exported_as_official_primary() -> None:
    job = _job(
        primary_source_url="https://sg.trabajo.org/job-123",
        official_job_url="https://sg.trabajo.org/job-123",
        secondary_source_url="https://www.linkedin.com/jobs/view/123",
        job_page_url="https://www.linkedin.com/jobs/view/123",
        job_page_kind="secondary_exact",
        job_page_confidence="medium",
    )
    clean = sanitize_job_sources(job)
    assert clean.primary_source_url is None
    assert clean.official_job_url is None
    assert clean.secondary_source_url == "https://www.linkedin.com/jobs/view/123"
    assert clean.job_page_url == "https://www.linkedin.com/jobs/view/123"


def test_matching_ready_requires_nonexpired_fetched_jd() -> None:
    assert is_matching_ready(_job()) is True
    assert is_matching_ready(_job(availability_status="expired_by_source_deadline")) is False
    assert is_matching_ready(_job(jd_status="source_context_only", jd_source_url=None)) is False
    assert is_matching_ready(_job(jd_text="too short")) is False


def test_partial_jd_is_a_valid_mid_strength_matching_input() -> None:
    text = "Engineer works on embedded firmware, hardware debug and C++ product development. " * 5
    job = _job(
        jd_status="partial_secondary",
        jd_text=text,
        jd_source_url="https://www.linkedin.com/jobs/view/456",
    )
    assert is_matching_ready(job) is False
    assert matching_evidence_level(job) == "partial_jd"
    assert matching_input_text(job) == text.strip()
