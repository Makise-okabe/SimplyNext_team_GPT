from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import quote_plus, urlparse, parse_qs, unquote

from career_agent.job_research_quality import is_plausible_official_url, is_secondary_url
from career_agent.models.job_record import JobRecord
from career_agent.tools.web_search import SearchResult, search_public_web
from career_agent.tools.web_fetch import fetch_public_page, public_http_url
from career_agent.job_page_verifier import apply_page_verification, clear_unverified_links, clean_search_title, verify_job_page
from career_agent.research_session import current_session

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
    "sr", "ai", "ml", "ic", "rf", "it", "qa", "ui", "ux", "hr", "3d", "5g",
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
    company_tokens = _company_tokens(company)
    if not company_tokens:
        return False

    identity_tokens = set(TITLE_TOKEN_PATTERN.findall((identity_text or "").lower()))
    distinctive = [token for token in company_tokens if len(token) >= 4]
    short = [token for token in company_tokens if 2 <= len(token) < 4]
    one_letter = [token for token in company_tokens if len(token) == 1]

    distinctive_match = any(token in identity_tokens for token in distinctive)

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
    if not public_http_url(url):
        return False
    parsed = urlparse(url)
    path = unquote(parsed.path).lower().rstrip("/")
    if any(part in path.split("/") for part in ("search", "login", "signin")):
        return False
    if path in {"", "/jobs", "/job", "/careers", "/positions", "/openings"}:
        return False
    if any(parse_qs(parsed.query).get(key) for key in ("jobId", "jobid", "job_id", "jobCode", "jobcode", "reqid", "gh_jid")):
        return True
    if (parsed.hostname or "").endswith(".lever.co"):
        return len([p for p in path.split("/") if p]) >= 2
    return bool(re.search(r"/(?:jobs?|positions?|openings?)/[^/]+|/(?:jobdetail|job-detail|requisition)/[^/]+", path))


def _career_page_like(url: str) -> bool:
    value = (url or "").lower()
    return any(marker in value for marker in ("career", "careers", "jobs", "recruit"))


def _score_result(job: JobRecord, result: SearchResult) -> tuple[float, str, str] | None:
    identity = f"{result.title} {result.url}"
    if not _resolver_company_match(job.company, identity):
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
    elif official and (not concrete) and _career_page_like(result.url):
        # A concrete page for another role is not a generic careers landing page.
        # Example: Electrical Intern must not resolve to /jobs/mechanical-intern.
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


def _query(company: str, title: str) -> str:
    return f'"{company}" "{clean_search_title(title)}" careers job'


def _search_fallback_url(company: str, title: str) -> str:
    return f"https://www.google.com/search?q={quote_plus(f'\"{company}\" \"{clean_search_title(title)}\" official careers job')}"


def resolve_job_link(job: JobRecord) -> tuple[JobRecord, LinkResolution]:
    """Discover candidates, then verify fetched identity before publishing any URL.

    At most three queries and six destination fetches per role. Failed candidates
    never remove the original opportunity. Search snippets never certify a page.
    """
    session = current_session()
    company, title = (job.company or "").strip(), (job.title or "").strip()
    base = clear_unverified_links(job)
    if not company or not title or job.availability_status in {"expired_by_source_deadline", "closed_by_official"}:
        return base, LinkResolution(None, "unresolved", "low", None, 0)
    attempts, tried, fetched = [], set(), 0
    best_secondary = None
    best_candidate: tuple[float, str, str, str] | None = None
    best_careers_url = job.company_careers_url
    closed = None
    query_used = None

    def remember_candidate(url: str, *, score: float, reason: str) -> None:
        nonlocal best_candidate
        if not public_http_url(url):
            return
        if is_plausible_official_url(url, company):
            kind = "official_candidate"
            score += 100.0
        elif is_secondary_url(url):
            kind = "secondary_candidate"
        else:
            return
        candidate = (score, url, kind, reason)
        if best_candidate is None or candidate[0] > best_candidate[0]:
            best_candidate = candidate

    def fallback_fields() -> dict:
        update = {
            "search_fallback_url": _search_fallback_url(company, title),
            "search_resolution_status": "search_fallback_only",
            "company_careers_url": best_careers_url,
        }
        if best_candidate:
            _, url, kind, reason = best_candidate
            update.update(
                candidate_job_url=url,
                candidate_job_kind=kind,
                candidate_job_reason=reason,
            )
        return update

    def check(url, query):
        nonlocal fetched, closed, best_secondary, best_careers_url
        if not public_http_url(url) or url in tried or fetched >= 6:
            return None
        tried.add(url)
        fetched += 1
        try:
            page = session.fetch(url, fetch_public_page)
        except Exception as exc:
            attempts.append({"url": url, "status": "unavailable", "reason": type(exc).__name__, "query": query})
            return None
        verification = verify_job_page(job, page)
        attempts.append({"url": url, "final_url": page.final_url, "status": verification.status, "reason": verification.reason, "query": query})
        if verification.status == "verified":
            resolved = apply_page_verification(base, verification)
            if verification.kind == "official_exact":
                return resolved
            best_secondary = best_secondary or resolved
        elif verification.status == "closed" and verification.details.get("official"):
            closed = apply_page_verification(base, verification)
        # Official career pages are discovery seeds, never application buttons.
        if verification.status == "generic_page" and is_plausible_official_url(page.final_url, company):
            best_careers_url = best_careers_url or page.final_url
            links = [link for link in page.links if _looks_job_like(link) and is_plausible_official_url(link, company)]
            links.sort(key=lambda link: _resolver_title_overlap(title, unquote(link)), reverse=True)
            for link in links[:2]:
                if _resolver_title_overlap(clean_search_title(title), unquote(link)) < .5:
                    continue
                found = check(link, "employer_page_link")
                if found:
                    return found
        return None

    def finish(found):
        update = {"link_attempts": attempts}
        if found.link_verification_status != "verified":
            update.update(fallback_fields())
        found = found.model_copy(update=update)
        return found, LinkResolution(found.job_page_url, found.job_page_kind, found.job_page_confidence, query_used, len(attempts))

    existing = list(dict.fromkeys(filter(None, [job.job_page_url, job.official_job_url, job.primary_source_url, job.application_url, *job.source_urls, job.secondary_source_url])))
    existing.sort(key=lambda url: not is_plausible_official_url(url, company))
    for url in existing[:3]:
        if not (_looks_job_like(url) or is_plausible_official_url(url, company)):
            continue
        if _looks_job_like(url):
            remember_candidate(url, score=80.0, reason="Job-specific link supplied by the career email")
        elif is_plausible_official_url(url, company):
            best_careers_url = best_careers_url or url
        found = check(url, "email_or_existing_link")
        if found:
            return finish(found)
        if closed:
            return finish(closed)

    cleaned = clean_search_title(title)
    queries = [_query(company, title), f'"{company}" {cleaned} careers Singapore', f'"{company}" {cleaned} job']
    if job.location and "singapore" not in job.location.lower():
        queries[1] = f'"{company}" {cleaned} careers {job.location}'
    official_hosts = list(dict.fromkeys(urlparse(u).hostname for u in existing if is_plausible_official_url(u, company)))
    if official_hosts:
        queries[1] = f'site:{official_hosts[0]} {cleaned}'
    if job.job_id:
        queries.insert(0, f'"{company}" "{job.job_id}"')
    for query_used in list(dict.fromkeys(queries))[:3]:
        results = session.search(query_used, search_public_web)
        scored = []
        for result in results:
            score = _score_result(job, result)
            # Weak/missing search titles can hide a valid official posting.
            # Fetched content is still mandatory and authoritative.
            if score or (is_plausible_official_url(result.url, company) and _looks_job_like(result.url)):
                scored.append(((score[0] if score else 0), result))
                if score and _looks_job_like(result.url):
                    remember_candidate(
                        result.url,
                        score=(score[0] if score else 0),
                        reason="Search result matched the employer and role title but the page could not be fully verified",
                    )
                elif score and score[1] == "company_careers":
                    best_careers_url = best_careers_url or result.url
        scored.sort(key=lambda item: (not is_plausible_official_url(item[1].url, company), -item[0]))
        for _, result in scored:
            found = check(result.url, query_used)
            if found:
                return finish(found)
            if closed:
                return finish(closed)
        if best_secondary or fetched >= 6:
            break
    if best_secondary:
        return finish(best_secondary)
    from datetime import datetime, timezone
    unresolved = base.model_copy(update={
        **fallback_fields(),
        "link_verification_status": "unresolved",
        "link_verification_reason": "No destination passed company, role and availability checks",
        "link_checked_at": datetime.now(timezone.utc).isoformat(),
        "link_attempts": attempts,
    })
    return unresolved, LinkResolution(None, "unresolved", "low", query_used, len(attempts))
