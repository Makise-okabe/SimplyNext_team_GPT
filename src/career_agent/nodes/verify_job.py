from __future__ import annotations

import re
from urllib.parse import urlparse

from career_agent.config import Settings
from career_agent.models.job import Job

LOGIN_WALL_HOSTS = {
    "nus-csm.symplicity.com",
}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _normalize(value: str | None) -> str:
    return " ".join((value or "").lower().split())


def _tokens(value: str | None) -> list[str]:
    return [token for token in TOKEN_PATTERN.findall(_normalize(value)) if len(token) >= 4]


def _token_match(value: str | None, haystack: str, minimum: int = 2) -> bool:
    tokens = _tokens(value)
    if not tokens:
        return False
    hits = sum(token in haystack for token in tokens)
    return hits >= min(minimum, len(tokens))


def _matching_page(job: dict, pages: list[dict]) -> dict | None:
    official_url = job.get("official_url")
    if not official_url:
        return None

    for page in pages:
        if official_url in {page.get("requested_url"), page.get("final_url")}:
            return page
    return None


def _web_verification(job: dict, page: dict | None) -> tuple[str, str]:
    official_url = job.get("official_url")
    if not official_url:
        return "unresolved", "none"

    host = urlparse(official_url).netloc.lower()
    if host in LOGIN_WALL_HOSTS:
        return "partial", "public_web"

    if not page or page.get("status_code") != 200:
        return "partial", "public_web"

    haystack = _normalize(f"{page.get('title', '')} {page.get('text', '')}")
    company = _normalize(job.get("company"))
    company_ok = bool(company and (company in haystack or _token_match(company, haystack, 1)))
    title_ok = _token_match(job.get("title"), haystack, 2)

    if company_ok and title_ok:
        return "verified", "official_web"
    return "partial", "public_web"


def _source_attachment_verification(job: dict, email: dict) -> bool:
    sender = (email.get("sender_email") or "").strip().lower()
    if sender not in Settings().trusted_senders:
        return False

    attachment_text = _normalize(email.get("attachment_text"))
    if not attachment_text:
        return False

    company_ok = _token_match(job.get("company"), attachment_text, 1)
    title_ok = _token_match(job.get("title"), attachment_text, 2)
    return company_ok and title_ok


def verify_job(state: dict) -> dict:
    """Validate jobs against official pages or trusted attached JDs.

    ``source_verified`` deliberately differs from ``verified``: it means the
    opportunity is strongly grounded in a trusted NUS career email attachment,
    but we did not prove that a live official employer posting still exists.
    """
    pages = state.get("resolved_pages") or []
    email = state.get("email") or {}
    verified_jobs: list[dict] = []
    errors = list(state.get("errors", []))

    for candidate in state.get("candidate_jobs") or []:
        page = _matching_page(candidate, pages)
        status, basis = _web_verification(candidate, page)
        evidence = list(candidate.get("evidence") or [])

        if page:
            if page.get("title"):
                evidence.append(f"Page title: {page['title']}")
            evidence.append(f"Fetched URL: {page.get('final_url')}")

        if status == "unresolved" and _source_attachment_verification(candidate, email):
            status = "source_verified"
            basis = "trusted_email_attachment"
            source = email.get("sender_email") or email.get("sender_name") or "trusted NUS source"
            evidence.append(f"Trusted source attachment matched job identity: {source}")

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
                verification_basis=basis,
                evidence=evidence,
            )
            verified_jobs.append(job.model_dump(mode="json"))
        except Exception as exc:
            errors.append(f"job verification model error: {exc}")

    return {"verified_jobs": verified_jobs, "errors": errors}
