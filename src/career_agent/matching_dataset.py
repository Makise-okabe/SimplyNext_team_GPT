from __future__ import annotations

from urllib.parse import urlparse

from career_agent.models.job_record import JobRecord

# Public aggregators/mirrors are useful as secondary evidence or JD fallbacks, but
# they must never be presented to the matching layer as employer-official primary
# sources.
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
    """Remove obviously non-official primary labels before matching/export."""
    primary = job.primary_source_url
    secondary = job.secondary_source_url

    if is_aggregator_url(primary):
        # Keep known useful mirrors as secondary evidence when there is no better
        # secondary already recorded. trabajo/other weak aggregators are dropped.
        if secondary is None and primary and any(
            marker in _host(primary)
            for marker in ("linkedin.com", "indeed.", "jobstreet.", "jobsdb.")
        ):
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
    """Only researched, non-expired jobs with a real fetched JD enter matching."""
    if job.availability_status == "expired_by_source_deadline":
        return False
    if job.jd_status not in {"fetched_official", "fetched_secondary"}:
        return False
    if not job.jd_source_url:
        return False
    return len(job.jd_text.strip()) >= 500
