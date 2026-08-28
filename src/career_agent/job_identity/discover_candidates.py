from __future__ import annotations

import re
import time
from urllib.parse import urlparse

from career_agent.models.job_identity import JobIdentity
from career_agent.models.job_search import (
    CandidateDiscoveryResult,
    CandidateSearchMetrics,
    SearchCandidate,
    SearchTraceStep,
)
from career_agent.tools.web_search import SearchResult, search_public_web

MAX_SEARCH_ROUNDS = 3
MAX_RESULTS_PER_QUERY = 5
MAX_FINAL_CANDIDATES = 5

APPLICATION_FORM_HOSTS = {
    "forms.office.com",
    "forms.microsoft.com",
    "forms.cloud.microsoft",
    "forms.gle",
    "docs.google.com",
    "typeform.com",
    "www.typeform.com",
    "airtable.com",
    "www.airtable.com",
}

NOISE_DIRECT_HOSTS = {
    "outlook.live.com",
    "outlook.office.com",
    "outlook.office365.com",
    "aka.ms",
    "login.microsoftonline.com",
}

AGGREGATOR_HOST_HINTS = (
    "indeed.",
    "glassdoor.",
    "jobstreet.",
    "jobsdb.",
    "talent.com",
    "grabjobs.",
)

ATS_HOST_HINTS = (
    "myworkdayjobs.com",
    "workday.com",
    "greenhouse.io",
    "lever.co",
    "smartrecruiters.com",
    "successfactors.com",
    "oraclecloud.com",
    "icims.com",
    "eightfold.ai",
)

UNIVERSITY_SOURCE_HOSTS = {
    "careeraxis.ntu.edu.sg",
    "nus-csm.symplicity.com",
}

GENERIC_WORDS = {
    "and",
    "the",
    "for",
    "with",
    "intern",
    "internship",
    "engineer",
    "engineering",
    "graduate",
    "singapore",
    "company",
    "limited",
}

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
PARENTHETICAL = re.compile(r"\s*\([^)]{1,30}\)\s*")
CORPORATE_SUFFIX = re.compile(
    r"(?i)\s+(?:&\s+company|company|pte\.?\s+ltd\.?|ltd\.?|inc\.?|corp\.?|corporation)$"
)


def _quote(value: str | None) -> str:
    cleaned = " ".join((value or "").split()).strip()
    return f'"{cleaned}"' if cleaned else ""


def _tokens(value: str | None) -> set[str]:
    return {
        token
        for token in TOKEN_PATTERN.findall((value or "").lower())
        if len(token) >= 3 and token not in GENERIC_WORDS
    }


def _search_company(identity: JobIdentity) -> str:
    company = " ".join((identity.company or "").split()).strip()
    return CORPORATE_SUFFIX.sub("", company).strip() or company


def _without_parenthetical(value: str | None) -> str:
    return " ".join(PARENTHETICAL.sub(" ", value or "").split())


def _classify_url(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
    except ValueError:
        return "unknown"

    if host in APPLICATION_FORM_HOSTS:
        return "application_form"
    if host in UNIVERSITY_SOURCE_HOSTS or host.endswith(".nus.edu.sg") or host.endswith(".ntu.edu.sg"):
        return "source_page"
    if any(hint in host for hint in AGGREGATOR_HOST_HINTS):
        return "aggregator"
    if any(hint in host for hint in ATS_HOST_HINTS):
        return "employer_or_ats"
    return "unknown"


def _metadata_query(identity: JobIdentity) -> str:
    company = _search_company(identity)
    unit = _without_parenthetical(identity.business_unit)
    title = _without_parenthetical(identity.title)
    location = identity.location or ""

    if unit:
        role_word = "internship" if identity.opportunity_type == "internship" else "job"
        parts = [company, _quote(unit), location, role_word]
    else:
        parts = [company, _quote(title), location]
    return " ".join(part for part in parts if part).strip()


def _identifier_query(identity: JobIdentity) -> str | None:
    if not identity.identifiers:
        return None
    strongest = identity.identifiers[0]
    parts = [_quote(strongest.value), _search_company(identity)]
    return " ".join(part for part in parts if part).strip()


def _best_distinctive_phrase(identity: JobIdentity) -> str | None:
    title = _without_parenthetical(identity.title).lower()
    unit = _without_parenthetical(identity.business_unit).lower()

    for phrase in identity.distinctive_phrases:
        cleaned = _without_parenthetical(phrase)
        lowered = cleaned.lower()
        if not cleaned:
            continue
        if lowered in title or lowered in unit or title in lowered or (unit and unit in lowered):
            continue
        return cleaned

    return _without_parenthetical(identity.distinctive_phrases[0]) if identity.distinctive_phrases else None


def _distinctive_query(identity: JobIdentity) -> str | None:
    phrase = _best_distinctive_phrase(identity)
    if not phrase:
        return None
    parts = [_search_company(identity), _quote(phrase)]
    if identity.location:
        parts.append(identity.location)
    return " ".join(part for part in parts if part).strip()


def build_progressive_queries(identity: JobIdentity) -> list[tuple[str, str]]:
    queries: list[tuple[str, str]] = []
    identifier = _identifier_query(identity)
    if identifier:
        queries.append(("exact_identifier", identifier))

    metadata = _metadata_query(identity)
    if metadata:
        queries.append(("metadata", metadata))

    distinctive = _distinctive_query(identity)
    if distinctive and distinctive != metadata:
        queries.append(("distinctive_phrase", distinctive))

    return queries[:MAX_SEARCH_ROUNDS]


def _score_result(identity: JobIdentity, result: SearchResult, strategy: str) -> SearchCandidate:
    haystack = f"{result.title} {result.snippet} {result.url}".lower()
    host = urlparse(result.url).netloc.lower()
    url_kind = _classify_url(result.url)

    score = 0.0
    reasons: list[str] = []
    identifier_hits: list[str] = []
    phrase_hits: list[str] = []
    metadata_hits: list[str] = []

    for identifier in identity.identifiers:
        if identifier.value.lower() in haystack:
            identifier_hits.append(identifier.value)
    if identifier_hits:
        score += 55
        reasons.append("explicit job identifier found in search result")

    company_tokens = _tokens(identity.company)
    title_tokens = _tokens(identity.title)
    location_tokens = _tokens(identity.location)
    haystack_tokens = _tokens(haystack)

    company_overlap = company_tokens & haystack_tokens
    title_overlap = title_tokens & haystack_tokens
    location_overlap = location_tokens & haystack_tokens

    if company_tokens and company_overlap:
        score += min(15, 5 + 4 * len(company_overlap))
        metadata_hits.append("company")
    if title_tokens and title_overlap:
        title_ratio = len(title_overlap) / max(1, len(title_tokens))
        score += 25 * title_ratio
        metadata_hits.append("title")
    if location_tokens and location_overlap:
        score += 6
        metadata_hits.append("location")

    unit = _without_parenthetical(identity.business_unit)
    if unit and unit.lower() in haystack:
        score += 8
        metadata_hits.append("business_unit")

    for phrase in identity.distinctive_phrases:
        if _without_parenthetical(phrase).lower() in haystack:
            phrase_hits.append(phrase)
    if phrase_hits:
        score += min(24, 8 * len(phrase_hits))
        reasons.append(f"{len(phrase_hits)} distinctive phrase(s) matched")

    if url_kind == "employer_or_ats":
        score += 8
        reasons.append("ATS/employer-like host")
    elif url_kind == "aggregator":
        score -= 6
        reasons.append("aggregator host")
    elif url_kind == "source_page":
        score -= 2
        reasons.append("university/source page, not employer-controlled")
    elif url_kind == "application_form":
        score -= 3
        reasons.append("application form, not identity proof")

    if strategy == "exact_identifier" and identifier_hits:
        score += 10

    return SearchCandidate(
        url=result.url,
        title=result.title,
        snippet=result.snippet,
        host=host,
        url_kind=url_kind,
        discovery_score=round(max(0.0, score), 2),
        strategies=[strategy],
        identifier_hits=identifier_hits,
        distinctive_phrase_hits=phrase_hits,
        metadata_hits=metadata_hits,
        reasons=reasons,
    )


def _merge_candidate(existing: SearchCandidate, incoming: SearchCandidate) -> SearchCandidate:
    return existing.model_copy(
        update={
            "discovery_score": max(existing.discovery_score, incoming.discovery_score),
            "strategies": list(dict.fromkeys([*existing.strategies, *incoming.strategies])),
            "identifier_hits": list(dict.fromkeys([*existing.identifier_hits, *incoming.identifier_hits])),
            "distinctive_phrase_hits": list(dict.fromkeys([*existing.distinctive_phrase_hits, *incoming.distinctive_phrase_hits])),
            "metadata_hits": list(dict.fromkeys([*existing.metadata_hits, *incoming.metadata_hits])),
            "reasons": list(dict.fromkeys([*existing.reasons, *incoming.reasons])),
        }
    )


def _direct_candidates(identity: JobIdentity) -> dict[str, SearchCandidate]:
    candidates: dict[str, SearchCandidate] = {}
    for url in identity.direct_urls:
        try:
            host = urlparse(url).netloc.lower()
        except ValueError:
            continue
        if host in NOISE_DIRECT_HOSTS:
            continue

        kind = _classify_url(url)
        score = 40.0 if kind == "employer_or_ats" else 18.0
        reasons = ["URL supplied directly by source email"]
        if kind == "application_form":
            reasons.append("application form retained as application evidence only")
        candidates[url] = SearchCandidate(
            url=url,
            host=host,
            url_kind=kind,
            discovery_score=score,
            strategies=["direct_url"],
            reasons=reasons,
        )
    return candidates


def _strong_identifier_candidate(candidates: dict[str, SearchCandidate]) -> bool:
    return any(candidate.identifier_hits and candidate.discovery_score >= 65 for candidate in candidates.values())


def discover_candidates(identity: JobIdentity) -> CandidateDiscoveryResult:
    started = time.perf_counter()
    candidates = _direct_candidates(identity)
    trace: list[SearchTraceStep] = []
    errors: list[str] = []
    search_calls = 0
    raw_results_seen = 0
    stopped_reason = "search_budget_exhausted"

    if any(c.url_kind == "employer_or_ats" for c in candidates.values()):
        stopped_reason = "direct_employer_candidate_available"
    else:
        for round_number, (strategy, query) in enumerate(build_progressive_queries(identity), start=1):
            round_started = time.perf_counter()
            try:
                results = search_public_web(query, max_results=MAX_RESULTS_PER_QUERY)
                search_calls += 1
                raw_results_seen += len(results)
            except Exception as exc:
                search_calls += 1
                errors.append(f"{strategy} search failed: {type(exc).__name__}: {exc}")
                results = []

            for result in results:
                incoming = _score_result(identity, result, strategy)
                if incoming.url in candidates:
                    candidates[incoming.url] = _merge_candidate(candidates[incoming.url], incoming)
                else:
                    candidates[incoming.url] = incoming

            trace.append(
                SearchTraceStep(
                    round_number=round_number,
                    strategy=strategy,
                    query=query,
                    results_returned=len(results),
                    candidates_after_merge=len(candidates),
                    elapsed_ms=int((time.perf_counter() - round_started) * 1000),
                )
            )

            if strategy == "exact_identifier" and _strong_identifier_candidate(candidates):
                stopped_reason = "strong_identifier_candidate_found"
                break
            if search_calls >= MAX_SEARCH_ROUNDS:
                stopped_reason = "max_search_rounds_reached"
                break
        else:
            stopped_reason = "all_progressive_queries_completed"

    ranked = sorted(
        candidates.values(),
        key=lambda candidate: (
            candidate.discovery_score,
            bool(candidate.identifier_hits),
            candidate.url_kind == "employer_or_ats",
        ),
        reverse=True,
    )[:MAX_FINAL_CANDIDATES]

    return CandidateDiscoveryResult(
        candidates=ranked,
        trace=trace,
        metrics=CandidateSearchMetrics(
            search_calls=search_calls,
            raw_results_seen=raw_results_seen,
            unique_candidates=len(candidates),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            llm_calls=0,
            stopped_reason=stopped_reason,
        ),
        errors=errors,
    )
