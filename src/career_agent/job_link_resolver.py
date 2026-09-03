from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import quote_plus, urlparse

from career_agent.job_research_quality import is_plausible_official_url
from career_agent.models.job_record import JobRecord
from career_agent.tools.web_search import SearchResult, search_public_web

MIN_OFFICIAL_EXACT_TITLE_OVERLAP = 0.65
MIN_SECONDARY_EXACT_TITLE_OVERLAP = 0.80
TITLE_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
TITLE_STOPWORDS = {
    "the", "and", "for", "with", "role", "position", "hiring",
    "career", "careers", "job", "jobs", "singapore",
}
COMPANY_LEGAL_STOPWORDS = {
    "the", "pte", "ltd", "limited", "inc", "private", "company",
    "corporation", "corp", "plc", "llc", "singapore", "branch",
}
MEANINGFUL_SHORT_TITLE_TOKENS = {
    "ai", "ml", "ic", "rf", "it", "qa", "ui", "ux", "hr", "3d", "5g",
}
TITLE_TOKEN_CANONICAL = {
    "engineers": "engineer",
    "developer": "develop",
    "developers": "develop",
    "development": "develop",
    "developing": "develop",
    "internship": "intern",
    "internships": "intern",
    "researcher": "research",
    "researchers": "research",
    "applications": "application",
    "designing": "design",
    "designer": "design",
    "designers": "design",
    "analytics": "analysis",
    "analyst": "analysis",
    "analysts": "analysis",
}


@dataclass(frozen=True)
class LinkResolution:
    url: str | None
    kind: str
    confidence: str
    search_query: str | None
    candidate_count: int


def _canonical_title_token(token: str) -> str:
    return TITLE_TOKEN_CANONICAL.get(token.lower(), token.lower())


def _resolver_title_tokens(value: str | None) -> set[str]:
    tokens: set[str] = set()
    for raw in TITLE_TOKEN_PATTERN.findall((value or "").lower()):
        if raw in TITLE_STOPWORDS:
            continue
        if len(raw) < 3 and raw not in MEANINGFUL_SHORT_TITLE_TOKENS and not raw.isdigit():
            continue
        tokens.add(_canonical_title_token(raw))
    return tokens


def _resolver_title_overlap(title: str | None, text: str) -> float:
    source = _resolver_title_tokens(title)
    if not source:
        return 0.0
    target = _resolver_title_tokens(text)
    return len(source & target) / len(source)


def _company_tokens(company: str | None) -> list[str]:
    return [
        token
        for token in TITLE_TOKEN_PATTERN.findall((company or "").lower())
        if token not in COMPANY_LEGAL_STOPWORDS
    ]


def _resolver_company_match(company: str | None, identity_text: str) -> bool:
    """Match employer identity without substring traps such as Face AI -> Facebook.

    Multi-token brands that include a short distinguishing token (for example
    ``Spirit AI`` or ``Face AI``) must keep that token. This prevents a shared
    generic brand word from resolving to a different employer such as
    Spirit AeroSystems or Facebook.
    """
    company_tokens = _company_tokens(company)
    if not company_tokens:
        return False

    identity_tokens = set(TITLE_TOKEN_PATTERN.findall((identity_text or "").lower()))
    distinctive = [token for token in company_tokens if len(token) >= 4]
    short = [token for token in company_tokens if 2 <= len(token) < 4]
    one_letter = [token for token in company_tokens if len(token) == 1]

    distinctive_match = any(token in identity_tokens for token in distinctive)

    # If the employer name includes a meaningful short brand/acronym token, it
    # must also appear as a whole token. E.g. Spirit AI must contain both
    # "spirit" and "ai"; matching only "spirit" is insufficient.
    if short and len(company_tokens) >= 2:
        if not all(token in identity_tokens for token in short):
            return False
        if distinctive:
            return distinctive_match
        return True

    if one_letter and len(one_letter) == len(company_tokens):
        if all(token in identity_tokens for token in one_letter):
            return True
        compact_letters = "".join(one_letter)
        compact_identity = re.sub(r"[^a-z0-9]", "", (identity_text or "").lower())
        return bool(compact_letters and compact_letters in compact_identity)

    if distinctive_match:
        return True

    if short and any(token in identity_tokens for token in short):
        return True

    acronym = "".join(token[0] for token in company_tokens if token)
    if len(acronym) >= 2 and acronym in identity_tokens:
        return True

    compact_company = "".join(company_tokens)
    compact_identity = re.sub(r"[^a-z0-9]", "", (identity_text or "").lower())
    return len(compact_company) >= 5 and compact_company in compact_identity


def _looks_job_like(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    value = f"{parsed.netloc.lower()} {parsed.path.lower()} {parsed.query.lower()}"
    return any(
        marker in value
        for marker in (
            "/job/", "/jobs/", "jobdetail", "job-detail", "jobid=", "job_id=",
            "jobcode=", "requisition", "reqid=", "/position/", "/positions/",
            "/opening/", "/openings/", "linkedin.com/jobs/view",
        )
    )


def _career_page_like(url: str) -> bool:
    value = (url or "").lower()
    return any(marker in value for marker in ("career", "careers", "jobs", "recruit"))


def _score_result(job: JobRecord, result: SearchResult) -> tuple[float, str, str] | None:
    identity = f"{result.title} {result.url}"
    company_matches = _resolver_company_match(job.company, identity)
    if not company_matches:
        return None

    official = is_plausible_official_url(result.url, job.company)
    overlap = _resolver_title_overlap(job.title, identity)
    concrete = _looks_job_like(result.url)

    if official and concrete and overlap >= MIN_OFFICIAL_EXACT_TITLE_OVERLAP:
        kind = "official_exact"
        confidence = "high"
    elif (not official) and concrete and overlap >= MIN_SECONDARY_EXACT_TITLE_OVERLAP:
        kind = "secondary_exact"
        confidence = "medium"
    elif official and _career_page_like(result.url):
        kind = "company_careers"
        confidence = "low"
    else:
        return None

    score = overlap * 100.0
    if official:
        score += 35.0
    if concrete:
        score += 20.0
    if kind == "company_careers":
        score -= 25.0
    return score, kind, confidence


def _existing_resolution(job: JobRecord) -> LinkResolution | None:
    candidates = [
        job.job_page_url,
        job.official_job_url,
        job.primary_source_url,
        job.application_url,
        *job.source_urls,
        job.secondary_source_url,
    ]
    seen: set[str] = set()
    for url in candidates:
        if not url or url in seen:
            continue
        seen.add(url)
        result = SearchResult(
            title=f"{job.company or ''} {job.title or ''}".strip(),
            url=url,
            snippet="",
        )
        scored = _score_result(job, result)
        if scored is None:
            continue
        _, kind, confidence = scored
        return LinkResolution(url=url, kind=kind, confidence=confidence, search_query=None, candidate_count=1)
    return None


def _query(company: str, title: str) -> str:
    return f'"{company}" "{title}" careers job'


def _search_fallback_url(company: str, title: str) -> str:
    query = f'"{company}" "{title}" jobs'
    return f"https://www.google.com/search?q={quote_plus(query)}"


def resolve_job_link(job: JobRecord) -> tuple[JobRecord, LinkResolution]:
    """Resolve one exact role page with one logical search, else keep a fallback."""
    existing = _existing_resolution(job)
    if existing is not None:
        return _apply_resolution(job, existing), existing

    company = (job.company or "").strip()
    title = (job.title or "").strip()
    if not company or not title:
        unresolved = LinkResolution(None, "unresolved", "low", None, 0)
        return job.model_copy(update={"search_resolution_status": "not_searched"}), unresolved

    query = _query(company, title)
    try:
        results = search_public_web(query, max_results=8)
    except Exception:
        results = []

    scored_results: list[tuple[float, SearchResult, str, str]] = []
    for result in results:
        scored = _score_result(job, result)
        if scored is None:
            continue
        score, kind, confidence = scored
        scored_results.append((score, result, kind, confidence))

    if not scored_results:
        unresolved = LinkResolution(None, "unresolved", "low", query, 0)
        fallback = _search_fallback_url(company, title)
        return job.model_copy(
            update={
                "job_page_url": None,
                "job_page_kind": "unresolved",
                "job_page_confidence": "low",
                "search_fallback_url": fallback,
                "search_resolution_status": "search_fallback_only",
            }
        ), unresolved

    scored_results.sort(key=lambda item: item[0], reverse=True)
    _, best, kind, confidence = scored_results[0]
    resolution = LinkResolution(
        url=best.url,
        kind=kind,
        confidence=confidence,
        search_query=query,
        candidate_count=len(scored_results),
    )
    return _apply_resolution(job, resolution), resolution


def _apply_resolution(job: JobRecord, resolution: LinkResolution) -> JobRecord:
    if not resolution.url:
        return job.model_copy(
            update={
                "job_page_url": None,
                "job_page_kind": "unresolved",
                "job_page_confidence": "low",
            }
        )

    update = {
        "job_page_url": resolution.url,
        "job_page_kind": resolution.kind,
        "job_page_confidence": resolution.confidence,
        "search_resolution_status": "resolved_job_page",
    }
    if resolution.kind.startswith("official"):
        update["primary_source_url"] = resolution.url
        update["official_job_url"] = resolution.url
        update["application_url"] = job.application_url or resolution.url
    elif resolution.kind.startswith("secondary"):
        update["secondary_source_url"] = resolution.url
        update["application_url"] = job.application_url or resolution.url
    elif resolution.kind == "company_careers":
        update["primary_source_url"] = resolution.url
        update["official_job_url"] = resolution.url
    return job.model_copy(update=update)