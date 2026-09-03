from __future__ import annotations

from urllib.parse import urlparse

from career_agent.models.job_record import JobRecord

AGGREGATOR_HOST_MARKERS = (
    "linkedin.com",
    "indeed.",
    "glassdoor.",
    "jobstreet.",
    "jobsdb.",
    "trabajo.org",
    "talent.com",
    "grabjobs.",
    "foundit.",
    "jooble.",
    "builtin.com",
    "expertini.com",
)


def _host(url: str | None) -> str:
    if not url:
        return ""
    try:
        return urlparse(url).netloc.lower().split(":", 1)[0]
    except ValueError:
        return ""


def is_aggregator_url(url: str | None) -> bool:
    host = _host(url)
    return bool(host and any(marker in host for marker in AGGREGATOR_HOST_MARKERS))


def sanitize_job_sources(job: JobRecord) -> JobRecord:
    """Keep official and secondary provenance separate without dropping UI links."""
    primary = job.primary_source_url
    secondary = job.secondary_source_url

    if is_aggregator_url(primary):
        if secondary is None and primary:
            secondary = primary
        primary = None

    official = job.official_job_url
    if is_aggregator_url(official):
        official = None

    return job.model_copy(
        update={
            "primary_source_url": primary,
            "secondary_source_url": secondary,
            "official_job_url": official,
        }
    )


def is_matching_ready(job: JobRecord) -> bool:
    """High-evidence matching input: active job with a full fetched JD."""
    if job.availability_status in {"expired_by_source_deadline", "closed_by_official"}:
        return False
    if job.jd_status not in {"fetched_official", "fetched_secondary"}:
        return False
    return bool(job.jd_source_url and len(job.jd_text.strip()) >= 500)


def has_partial_jd(job: JobRecord) -> bool:
    if job.availability_status in {"expired_by_source_deadline", "closed_by_official"}:
        return False
    return job.jd_status in {"partial_official", "partial_secondary"} and len(job.jd_text.strip()) >= 180


def is_matching_candidate(job: JobRecord) -> bool:
    """Any active known job can be ranked, even with title/email evidence only."""
    if job.availability_status in {"expired_by_source_deadline", "closed_by_official"}:
        return False
    return bool((job.company or "").strip() and (job.title or "").strip())


def matching_evidence_level(job: JobRecord) -> str:
    if is_matching_ready(job):
        return "full_jd"
    if has_partial_jd(job):
        return "partial_jd"
    if is_matching_candidate(job):
        return "source_only"
    return "inactive"


def matching_input_text(job: JobRecord) -> str:
    """Best available deterministic evidence handed to the matching pipeline."""
    level = matching_evidence_level(job)
    if level in {"full_jd", "partial_jd"} and job.jd_text.strip():
        return job.jd_text.strip()

    parts = [
        f"Company: {job.company or ''}",
        f"Role: {job.title or ''}",
    ]
    if job.location:
        parts.append(f"Location: {job.location}")
    if job.opportunity_type:
        parts.append(f"Opportunity type: {job.opportunity_type}")
    if job.deadline_hint:
        parts.append(f"Deadline: {job.deadline_hint}")
    if job.source_evidence:
        parts.append(f"Source evidence: {job.source_evidence}")
    return "\n".join(parts).strip()
