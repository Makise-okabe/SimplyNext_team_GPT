from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import quote_plus, urlparse

from career_agent.company_job_research import _company_match
from career_agent.job_research_quality import is_aggregator_url, is_plausible_official_url
from career_agent.models.job_record import JobRecord
from career_agent.tools.web_search import SearchResult
from career_agent.tools.web_search_aggregate import search_public_web_aggregated

MIN_EXACT_TITLE_OVERLAP = 0.50
MIN_PROBABLE_TITLE_OVERLAP = 0.22
TITLE_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
TITLE_STOPWORDS = {
    "the", "and", "for", "with", "role", "position", "hiring",
    "career", "careers", "job", "jobs", "singapore",
}
MEANINGFUL_SHORT_TITLE_TOKENS = {
    "ai", "ml", "ic", "rf", "it", "qa", "ui", "ux", "hr", "3d", "5g",
}
TITLE_TOKEN_CANONICAL = {
    "engineers": "engineer",
    "engineering": "engineer",
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
            "/opening/", "/openings/", "greenhouse", "lever.co", "myworkdayjobs",
            "workdayjobs", "smartrecruiters", "successfactors", "employmenthero",
            "linkedin.com/jobs/view", "jobstreet", "indeed.",
        )
    )


def _career_page_like(url: str) -> bool:
    value = (url or "").lower()
    return any(marker in value for marker in ("career", "careers", "jobs", "recruit"))


def _score_result(job: JobRecord, result: SearchResult) -> tuple[float, str, str] | None:
    metadata = f"{result.title} {result.snippet} {result.url}"
    if not _company_match(job.company, metadata):
        return None

    overlap = _resolver_title_overlap(job.title, metadata)
    official = is_plausible_official_url(result.url, job.company)
    secondary = is_aggregator_url(result.url)
    concrete = _looks_job_like(result.url)

    if overlap >= MIN_EXACT_TITLE_OVERLAP and concrete:
        kind = "official_exact" if official else "secondary_exact"
        confidence = "high" if official else "medium"
    elif overlap >= MIN_PROBABLE_TITLE_OVERLAP and concrete:
        kind = "official_probable" if official else "secondary_probable"
        confidence = "medium" if official else "low"
    elif official and _career_page_like(result.url):
        kind = "company_careers"
        confidence = "low"
    else:
        return None

    score = overlap * 100.0
    if official:
        score += 35.0
    elif secondary:
        score += 12.0
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
        result = SearchResult(title=job.title or "", url=url, snippet=job.company or "")
        scored = _score_result(job, result)
        if scored is None:
            continue
        _, kind, confidence = scored
        return LinkResolution(url=url, kind=kind, confidence=confidence, search_query=None, candidate_count=1)
    return None


def _queries(company: str, title: str) -> list[str]:
    return [
        f'"{company}" "{title}"',
        f'{company} {title} job',
        f'{company} {title} careers',
        f'site:linkedin.com/jobs "{company}" "{title}"',
    ]


def _search_fallback_url(company: str, title: str) -> str:
    query = f'"{company}" "{title}" jobs'
    return f"https://www.google.com/search?q={quote_plus(query)}"


def resolve_job_link(job: JobRecord) -> tuple[JobRecord, LinkResolution]:
    """Resolve a concrete job page; otherwise attach an explicit search fallback."""
    existing = _existing_resolution(job)
    if existing is not None:
        return _apply_resolution(job, existing), existing

    company = (job.company or "").strip()
    title = (job.title or "").strip()
    if not company or not title:
        unresolved = LinkResolution(None, "unresolved", "low", None, 0)
        return job.model_copy(update={"search_resolution_status": "not_searched"}), unresolved

    queries = _queries(company, title)
    scored_results: list[tuple[float, SearchResult, str, str, str]] = []
    seen: set[str] = set()

    for query in queries:
        for result in search_public_web_aggregated(
            query,
            max_results=20,
            min_results=8,
            strict_relevance=False,
        ):
            if result.url in seen:
                continue
            seen.add(result.url)
            scored = _score_result(job, result)
            if scored is None:
                continue
            score, kind, confidence = scored
            scored_results.append((score, result, kind, confidence, query))

    if not scored_results:
        unresolved = LinkResolution(None, "unresolved", "low", queries[0], 0)
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
    _, best, kind, confidence, query = scored_results[0]
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
