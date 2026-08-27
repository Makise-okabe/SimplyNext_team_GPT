from __future__ import annotations

from urllib.parse import urlparse

from career_agent.models.job import Job

LOGIN_WALL_HOSTS = {
    "nus-csm.symplicity.com",
}


def _normalize(value: str | None) -> str:
    return " ".join((value or "").lower().split())


def _matching_page(job: dict, pages: list[dict]) -> dict | None:
    official_url = job.get("official_url")
    if not official_url:
        return None

    for page in pages:
        if official_url in {page.get("requested_url"), page.get("final_url")}:
            return page
    return None


def _verification_status(job: dict, page: dict | None) -> str:
    official_url = job.get("official_url")
    if not official_url:
        return "unresolved"

    host = urlparse(official_url).netloc.lower()
    if host in LOGIN_WALL_HOSTS:
        return "partial"

    if not page or page.get("status_code") != 200:
        return "partial"

    haystack = _normalize(f"{page.get('title', '')} {page.get('text', '')}")
    company = _normalize(job.get("company"))
    title = _normalize(job.get("title"))

    company_ok = bool(company and company in haystack)
    title_tokens = [token for token in title.split() if len(token) >= 4]
    title_ok = bool(title_tokens and sum(token in haystack for token in title_tokens) >= min(2, len(title_tokens)))

    return "verified" if company_ok and title_ok else "partial"


def verify_job(state: dict) -> dict:
    """Validate candidate jobs conservatively against pages we actually fetched."""
    pages = state.get("resolved_pages") or []
    verified_jobs: list[dict] = []
    errors = list(state.get("errors", []))

    for candidate in state.get("candidate_jobs") or []:
        page = _matching_page(candidate, pages)
        status = _verification_status(candidate, page)
        evidence = list(candidate.get("evidence") or [])

        if page:
            if page.get("title"):
                evidence.append(f"Page title: {page['title']}")
            evidence.append(f"Fetched URL: {page.get('final_url')}")

        try:
            job = Job(
                company=candidate.get("company") or "Unknown",
                title=candidate.get("title") or "Unknown role",
                location=candidate.get("location"),
                opportunity_type=candidate.get("opportunity_type", "unknown"),
                official_url=candidate.get("official_url"),
                deadline=candidate.get("deadline"),
                degree_requirements=candidate.get("degree_requirements", []),
                required_skills=candidate.get("required_skills", []),
                preferred_skills=candidate.get("preferred_skills", []),
                visa_information=candidate.get("visa_information"),
                raw_description=candidate.get("raw_description", ""),
                verification_status=status,
                evidence=evidence,
            )
            verified_jobs.append(job.model_dump(mode="json"))
        except Exception as exc:
            errors.append(f"job verification model error: {exc}")

    return {"verified_jobs": verified_jobs, "errors": errors}
