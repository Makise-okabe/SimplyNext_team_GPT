from __future__ import annotations

from career_agent.company_job_research import _company_match, _title_overlap
from career_agent.job_research_quality import clean_jd_text
from career_agent.models.job_record import JobRecord
from career_agent.tools.web_fetch import fetch_public_page

FULL_JD_MIN_CHARS = 500
PARTIAL_JD_MIN_CHARS = 180


def _plain_text(text: str) -> str:
    lines = [" ".join(line.split()).strip() for line in (text or "").splitlines()]
    return "\n".join(line for line in lines if line)


def _page_matches(job: JobRecord, title: str, text: str) -> bool:
    value = f"{title}\n{text}"
    return _company_match(job.company, value) and _title_overlap(job.title, value) >= 0.20


def enrich_job_description(job: JobRecord) -> JobRecord:
    """Fetch a resolved job page and keep the best usable evidence available.

    A failed fetch never removes an already resolved job link. Full structured JD
    is preferred, but a shorter page excerpt is still useful as partial evidence.
    """
    url = job.job_page_url or job.official_job_url or job.secondary_source_url
    if not url:
        return job

    try:
        page = fetch_public_page(url, timeout_seconds=10.0)
    except Exception as exc:
        return job.model_copy(
            update={
                "warnings": list(dict.fromkeys([*job.warnings, f"JD fetch failed: {type(exc).__name__}: {exc}"])),
            }
        )

    if not _page_matches(job, page.title, page.text):
        return job.model_copy(
            update={
                "warnings": list(dict.fromkeys([*job.warnings, "resolved job page did not match company/title strongly enough for JD extraction"])),
            }
        )

    official = job.job_page_kind.startswith("official")
    structured = clean_jd_text(page.text)
    if len(structured.strip()) >= FULL_JD_MIN_CHARS:
        return job.model_copy(
            update={
                "jd_status": "fetched_official" if official else "fetched_secondary",
                "jd_source_url": page.final_url or url,
                "jd_text": structured[:30_000],
                "research_status": "verified_exact_job",
                "research_confidence": "high" if official else "medium",
                "research_basis": "resolved_job_page_full_jd",
                "evidence_summary": list(dict.fromkeys([*job.evidence_summary, "resolved job page supplied a full job description"])),
            }
        )

    plain = _plain_text(page.text)
    if len(plain) >= PARTIAL_JD_MIN_CHARS:
        return job.model_copy(
            update={
                "jd_status": "partial_official" if official else "partial_secondary",
                "jd_source_url": page.final_url or url,
                "jd_text": plain[:12_000],
                "research_status": "verified_exact_job",
                "research_confidence": "medium" if official else "low",
                "research_basis": "resolved_job_page_partial_jd",
                "evidence_summary": list(dict.fromkeys([*job.evidence_summary, "resolved job page supplied partial job evidence"])),
            }
        )

    return job
