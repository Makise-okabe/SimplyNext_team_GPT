"""One content-based verification contract for discovery, enrichment and UI output."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from career_agent.job_research_quality import clean_jd_text, is_plausible_official_url, is_secondary_url, page_is_closed
from career_agent.models.job_record import JobRecord
from career_agent.tools.web_fetch import FetchedPage, public_http_url


@dataclass(frozen=True)
class PageVerification:
    status: str
    reason: str
    url: str | None = None
    kind: str = "unresolved"
    confidence: str = "low"
    details: dict = field(default_factory=dict)


def _plain(value) -> str:
    if isinstance(value, list):
        return "\n".join(_plain(item) for item in value)
    if isinstance(value, dict):
        return _plain(value.get("name") or value.get("value") or "")
    return BeautifulSoup(str(value or ""), "html.parser").get_text("\n", strip=True)


def clean_search_title(title: str) -> str:
    # Remove recruiting dates, keeping meaningful qualifiers such as RF or FPGA.
    value = re.sub(r"\([^)]*(?:20\d{2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b)[^)]*\)", " ", title, flags=re.I)
    return " ".join(value.split())


def titles_match(expected: str, observed: str, company: str = "") -> bool:
    from career_agent.job_link_resolver import _resolver_title_tokens
    source = _resolver_title_tokens(clean_search_title(expected))
    target = _resolver_title_tokens(re.sub(re.escape(company), "", observed, flags=re.I) if company else observed)
    target -= {"at", "in", "opportunity", "opportunities"}
    if not source or not target:
        return False
    # Do not accept a senior/specialist substitute for a student's original role.
    senior = {"senior", "sr", "lead", "principal", "staff", "director", "manager"}
    if (target & senior) - (source & senior):
        return False
    if ("intern" in source) != ("intern" in target):
        return False
    # Firmware/software have related tasks, but a different job is still different.
    for family in ({"electrical", "mechanical", "civil"}, {"hardware", "software"}, {"analog", "digital"}):
        if source & family and target & family and not source & target & family:
            return False
    return len(source & target) / len(source) >= (1.0 if len(source) <= 3 else 0.8)


def _location(value) -> str:
    if isinstance(value, list):
        return "; ".join(filter(None, (_location(item) for item in value)))
    if isinstance(value, dict):
        address = value.get("address", value)
        if isinstance(address, dict):
            return ", ".join(dict.fromkeys(filter(None, (_plain(address.get(k)) for k in ("addressLocality", "addressRegion", "addressCountry"))))) or _plain(value.get("name"))
        return _plain(address)
    return _plain(value)


def _section(text: str, labels: tuple[str, ...]) -> list[str]:
    headers = {"responsibilities", "key responsibilities", "requirements", "qualifications", "minimum qualifications", "preferred qualifications", "required skills", "preferred skills", "benefits", "about us", "job description"}
    lines, active, result = text.splitlines(), False, []
    for line in lines:
        normalized = line.strip().rstrip(":").lower()
        if normalized in headers:
            active = normalized in labels
        elif active and line.strip():
            result.append(line.strip().lstrip("•*- ")[:600])
    return result[:16]


def _expired(value) -> bool:
    if not value:
        return False
    try:
        text = str(value)
        end = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if len(text) == 10:
            return end.date() < datetime.now(timezone.utc).date()
        return end.replace(tzinfo=end.tzinfo or timezone.utc) < datetime.now(timezone.utc)
    except ValueError:
        return False


def verify_job_page(job: JobRecord, page: FetchedPage) -> PageVerification:
    from career_agent.job_link_resolver import _looks_job_like, _resolver_company_match
    url = page.final_url or page.requested_url
    if not public_http_url(url) or not 200 <= page.status_code < 300:
        return PageVerification("unavailable", "Destination is not a successful public page")
    path = urlparse(url).path.rstrip("/").lower()
    if path in {"", "/careers", "/jobs", "/search", "/en", "/en-us"} or any(part in path.split("/") for part in ("search", "login", "signin")):
        return PageVerification("generic_page", "Destination is a homepage, listing, or sign-in page")
    lowered = page.text[:1500].lower()
    if any(marker in lowered for marker in ("verify you are human", "checking your browser", "access denied", "enable javascript and cookies", "page not found")):
        return PageVerification("unavailable", "Destination returned an access challenge or error page")
    if len(page.job_postings) > 1:
        return PageVerification("generic_page", "Multiple postings on this page; a specific job page is needed")
    posting = page.job_postings[0] if page.job_postings else {}
    if not posting and not _looks_job_like(url):
        return PageVerification("generic_page", "No specific job route or JobPosting metadata")
    observed_title = _plain(posting.get("title")) or (page.headings[0] if page.headings else page.title)
    if not titles_match(job.title or "", observed_title, job.company or ""):
        return PageVerification("wrong_role", f"Destination identifies a different role: {observed_title[:180]}")
    official = is_plausible_official_url(url, job.company)
    employer = _plain(posting.get("hiringOrganization"))
    if employer and not _resolver_company_match(job.company, employer):
        return PageVerification("wrong_company", f"Posting names a different employer: {employer[:120]}")
    if not employer and not official and not _resolver_company_match(job.company, f"{page.title}\n{page.text[:3000]}"):
        return PageVerification("wrong_company", "Destination does not establish the expected employer")
    if not official and not is_secondary_url(url):
        return PageVerification("untrusted_source", "Destination is neither an employer/ATS match nor a supported secondary source")
    identifier = _plain(posting.get("identifier"))
    if job.job_id and identifier and job.job_id.casefold() != identifier.casefold():
        return PageVerification("wrong_job_id", f"Employer job ID differs: {identifier}")
    location = _location(posting.get("jobLocation"))
    if job.location and "singapore" in job.location.lower() and location and "singapore" not in location.lower() and location.lower() != "sg":
        return PageVerification("wrong_location", f"Posting location differs from the email: {location}")
    description = _plain(posting.get("description")) if posting else page.text
    if _expired(posting.get("validThrough")) or page_is_closed(description) or page_is_closed(page.text[:3000]):
        return PageVerification("closed", "The matching posting is closed or its validity date has passed", details={"official": official})
    # Require independent readable job information; a descriptive URL is not evidence.
    if len(description.strip()) < 100:
        return PageVerification("insufficient_evidence", "Page identity matches but readable job information is insufficient")
    structured = clean_jd_text(description)
    jd = structured or description
    full = len(jd) >= 500 and (bool(structured) or bool(posting))
    partial = len(jd) >= 180
    basis = "official" if official else "secondary"
    details = {
        "jd_text": jd[:30000] if full or partial else "",
        "jd_status": f"fetched_{basis}" if full else f"partial_{basis}" if partial else job.jd_status,
        "jd_source_url": url if full or partial else job.jd_source_url,
        "job_id": identifier or job.job_id,
        "location": location or job.location,
        "responsibilities": [_plain(posting["responsibilities"])] if posting.get("responsibilities") else _section(jd, ("responsibilities", "key responsibilities")),
        "required_skills": [_plain(posting["skills"])] if posting.get("skills") else _section(jd, ("required skills", "requirements")),
        "preferred_skills": _section(jd, ("preferred skills", "preferred qualifications")),
        "qualifications": [_plain(posting["qualifications"])] if posting.get("qualifications") else _section(jd, ("qualifications", "minimum qualifications")),
    }
    employment = _plain(posting.get("employmentType")).lower().replace("_", " ")
    if "intern" in employment:
        details["opportunity_type"] = "internship"
    elif "full" in employment and job.opportunity_type == "unknown":
        details["opportunity_type"] = "full_time"
    return PageVerification("verified", f"Fetched role and employer match ({page.extraction_method})", url, f"{basis}_exact", "high" if official else "medium", details)


def clear_unverified_links(job: JobRecord) -> JobRecord:
    # Preserve email provenance, but never let stale aliases bypass the CTA contract.
    urls = list(dict.fromkeys(filter(None, [*job.source_urls, job.job_page_url, job.official_job_url, job.application_url, job.primary_source_url, job.secondary_source_url])))
    evidence_reset = {}
    if job.jd_status in {"fetched_official", "fetched_secondary", "partial_official", "partial_secondary"}:
        evidence_reset = {"jd_text": "", "jd_source_url": None, "jd_status": "source_context_only", "responsibilities": [], "required_skills": [], "preferred_skills": [], "qualifications": []}
    return job.model_copy(update={**evidence_reset, "job_page_url": None, "official_job_url": None, "application_url": None, "primary_source_url": None, "secondary_source_url": None, "job_page_kind": "unresolved", "job_page_confidence": "low", "source_urls": urls})


def apply_page_verification(job: JobRecord, check: PageVerification) -> JobRecord:
    base = clear_unverified_links(job)
    update = {"link_verification_status": check.status, "link_verification_reason": check.reason, "link_checked_at": datetime.now(timezone.utc).isoformat()}
    if check.status != "verified":
        if check.status == "closed" and check.details.get("official"):
            update["availability_status"] = "closed_by_official"
        return base.model_copy(update=update)
    update.update(check.details)
    update.update({"job_page_url": check.url, "application_url": check.url, "job_page_kind": check.kind, "job_page_confidence": check.confidence, "search_resolution_status": "resolved_job_page", "research_status": "verified_exact_job", "research_confidence": check.confidence, "research_basis": "fetched_job_identity", "research_skipped_reason": None})
    if check.kind == "official_exact":
        update.update(primary_source_url=check.url, official_job_url=check.url)
    else:
        update["secondary_source_url"] = check.url
    return base.model_copy(update=update)
