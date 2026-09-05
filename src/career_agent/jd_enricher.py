from __future__ import annotations

from career_agent.job_page_verifier import apply_page_verification, verify_job_page, clear_unverified_links
from career_agent.models.job_record import JobRecord
from career_agent.research_session import current_session
from career_agent.tools.web_fetch import fetch_public_page

FULL_JD_MIN_CHARS = 500
PARTIAL_JD_MIN_CHARS = 180


def enrich_job_description(job: JobRecord) -> JobRecord:
    """Use the same verified page for the destination and its job description."""
    if job.link_verification_status == "verified":
        return job  # Resolver already extracted this page; no second fetch.
    url = job.job_page_url or job.official_job_url or job.secondary_source_url
    if not url:
        return job
    try:
        page = current_session().fetch(url, fetch_public_page)
    except Exception as exc:
        return clear_unverified_links(job).model_copy(update={
            "link_verification_status": "unavailable", "link_verification_reason": f"Page fetch failed: {type(exc).__name__}",
        })
    return apply_page_verification(job, verify_job_page(job, page))
