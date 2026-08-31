from career_agent.matching_dataset import is_matching_ready, sanitize_job_sources
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
    )
    clean = sanitize_job_sources(job)
    assert clean.primary_source_url is None
    assert clean.official_job_url is None
    assert clean.secondary_source_url == "https://www.linkedin.com/jobs/view/123"


def test_matching_ready_requires_nonexpired_fetched_jd() -> None:
    assert is_matching_ready(_job()) is True
    assert is_matching_ready(_job(availability_status="expired_by_source_deadline")) is False
    assert is_matching_ready(_job(jd_status="source_context_only", jd_source_url=None)) is False
    assert is_matching_ready(_job(jd_text="too short")) is False
