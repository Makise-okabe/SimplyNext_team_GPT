from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import parse_qs, urlparse

from career_agent.config import Settings
from career_agent.job_identity.verify_same_job import verify_same_job
from career_agent.models.email import EmailMessage
from career_agent.models.job_identity import JobIdentity
from career_agent.models.job_search import CandidateDiscoveryResult, SearchCandidate
from career_agent.models.opportunity_research import (
    OpportunityResearchPackage,
    ResearchCandidate,
    ResearchMetrics,
    ResearchTraceStep,
    SourceProvenance,
)
from career_agent.tools.web_fetch import FetchedPage, fetch_public_page
from career_agent.tools.web_search import SearchResult, search_public_web

MAX_SEARCH_CALLS = 3
MAX_RESULTS_PER_QUERY = 8
MAX_FETCHES = 4

APPLICATION_HOSTS = {
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

INSTITUTIONAL_HOST_HINTS = (
    "nus.edu.sg",
    "ntu.edu.sg",
    "careeraxis.ntu.edu.sg",
    "symplicity.com",
)

SECONDARY_HOST_HINTS = (
    "linkedin.com",
    "glassdoor.",
    "indeed.",
    "jobstreet.",
    "jobsdb.",
    "talent.com",
    "efinancialcareers.",
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

NOISE_HOSTS = {
    "outlook.live.com",
    "outlook.office.com",
    "outlook.office365.com",
    "aka.ms",
    "login.microsoftonline.com",
}

GENERIC_COMPANY_WORDS = {
    "company",
    "limited",
    "ltd",
    "pte",
    "inc",
    "corp",
    "corporation",
    "asia",
    "singapore",
    "semiconductor",
    "technologies",
    "technology",
    "private",
}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
PARENTHETICAL = re.compile(r"\s*\([^)]{1,40}\)\s*")


def _normalize(value: str | None) -> str:
    return " ".join((value or "").lower().split())


def _without_parenthetical(value: str | None) -> str:
    return " ".join(PARENTHETICAL.sub(" ", value or "").split())


def _tokens(value: str | None) -> set[str]:
    return set(TOKEN_PATTERN.findall(_normalize(value)))


def _company_tokens(company: str | None) -> set[str]:
    return {
        token
        for token in _tokens(company)
        if len(token) >= 3 and token not in GENERIC_COMPANY_WORDS
    }


def _title_overlap(identity: JobIdentity, text: str) -> float:
    source = {
        token
        for token in _tokens(_without_parenthetical(identity.title))
        if len(token) >= 3
    }
    if not source:
        return 0.0
    return len(source & _tokens(text)) / len(source)


def _is_noise_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return True
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    if host in NOISE_HOSTS:
        return True
    if host.endswith("duckduckgo.com") and path.endswith("/y.js"):
        return True
    if host.endswith("bing.com") and "/aclick" in path:
        return True
    return False


def _is_application_url(url: str) -> bool:
    try:
        return urlparse(url).netloc.lower() in APPLICATION_HOSTS
    except ValueError:
        return False


def _looks_like_company_host(identity: JobIdentity, host: str) -> bool:
    host_tokens = set(TOKEN_PATTERN.findall(host.lower()))
    company_tokens = _company_tokens(identity.company)
    return bool(company_tokens and company_tokens & host_tokens)


def _ats_is_tied_to_company(identity: JobIdentity, title: str, snippet: str) -> bool:
    haystack = f"{title} {snippet}"
    company_tokens = _company_tokens(identity.company)
    return bool(company_tokens and company_tokens & _tokens(haystack))


def evidence_tier(
    identity: JobIdentity,
    url: str,
    title: str = "",
    snippet: str = "",
) -> str:
    if _is_noise_url(url):
        return "weak"

    parsed = urlparse(url)
    host = parsed.netloc.lower()

    if host in APPLICATION_HOSTS:
        return "weak"
    if any(hint in host for hint in INSTITUTIONAL_HOST_HINTS):
        return "institutional"
    if any(hint in host for hint in SECONDARY_HOST_HINTS):
        return "secondary"
    if _looks_like_company_host(identity, host):
        return "official"
    if any(hint in host for hint in ATS_HOST_HINTS) and _ats_is_tied_to_company(
        identity,
        title,
        snippet,
    ):
        return "official"
    return "weak"


def _job_like_url(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()
    query = parse_qs(parsed.query)
    if any(key.lower() in {"jobid", "job_id", "requisitionid", "reqid"} for key in query):
        return True
    return any(
        marker in path
        for marker in (
            "/job/",
            "/jobs/",
            "/jobdetail",
            "/position/",
            "/positions/",
            "/career/",
            "/careers/",
        )
    )


def extract_job_id_from_url(url: str) -> str | None:
    try:
        query = parse_qs(urlparse(url).query)
    except ValueError:
        return None
    for key, values in query.items():
        if key.lower() in {"jobid", "job_id", "requisitionid", "reqid"} and values:
            return values[0]
    return None


def infer_initial_record_kind(identity: JobIdentity) -> str:
    title = _normalize(identity.title)
    if "challenge" in title:
        return "challenge"
    if any(word in title for word in ("career event", "workshop", "career fair", "seminar")):
        return "event"
    if any(word in title for word in ("programme", "program", "academy", "bootcamp")):
        return "programme"
    if identity.identifiers:
        return "job_posting"

    for url in identity.direct_urls:
        if evidence_tier(identity, url) == "official" and _job_like_url(url):
            return "job_posting"

    if identity.opportunity_type == "full_time" and identity.title:
        return "job_posting"
    return "unknown"


def _candidate_relation(identity: JobIdentity, url: str, tier: str) -> str:
    if _is_application_url(url):
        return "application"
    if tier == "official" and _job_like_url(url):
        return "exact_posting"
    if tier == "official":
        return "official_background"
    return "secondary_evidence"


def _candidate_score(identity: JobIdentity, result: SearchResult, tier: str) -> float:
    haystack = f"{result.title} {result.snippet} {result.url}"
    score = 0.0
    if tier == "official":
        score += 40
    elif tier == "institutional":
        score += 20
    elif tier == "secondary":
        score += 12

    overlap = _title_overlap(identity, haystack)
    score += 30 * overlap
    if identity.location and _normalize(identity.location) in _normalize(haystack):
        score += 5

    business_unit = _without_parenthetical(identity.business_unit)
    if business_unit and _normalize(business_unit) in _normalize(haystack):
        score += 15

    for identifier in identity.identifiers:
        if identifier.value.lower() in haystack.lower():
            score += 50

    for phrase in identity.distinctive_phrases[:4]:
        if _normalize(_without_parenthetical(phrase)) in _normalize(haystack):
            score += 8

    return round(min(100.0, score), 2)


def _from_search_result(identity: JobIdentity, result: SearchResult) -> ResearchCandidate | None:
    if _is_noise_url(result.url):
        return None
    tier = evidence_tier(identity, result.url, result.title, result.snippet)
    return ResearchCandidate(
        url=result.url,
        title=result.title,
        snippet=result.snippet,
        host=urlparse(result.url).netloc.lower(),
        tier=tier,
        relation=_candidate_relation(identity, result.url, tier),
        score=_candidate_score(identity, result, tier),
        reasons=[f"classified as {tier} evidence"],
    )


def _direct_candidates(identity: JobIdentity) -> list[ResearchCandidate]:
    candidates: list[ResearchCandidate] = []
    for url in identity.direct_urls:
        if _is_noise_url(url):
            continue
        tier = evidence_tier(identity, url)
        relation = _candidate_relation(identity, url, tier)
        score = 95.0 if tier == "official" and relation == "exact_posting" else 30.0
        if relation == "application":
            score = 10.0
        reasons = ["URL supplied directly by source email/JD"]
        job_id = extract_job_id_from_url(url)
        if job_id:
            reasons.append(f"job identifier found in URL: {job_id}")
            score = max(score, 98.0 if tier == "official" else 50.0)
        candidates.append(
            ResearchCandidate(
                url=url,
                host=urlparse(url).netloc.lower(),
                tier=tier,
                relation=relation,
                score=score,
                reasons=reasons,
            )
        )
    return candidates


def _official_query(identity: JobIdentity, record_kind: str) -> str:
    company = " ".join((identity.company or "").split())
    location = identity.location or ""
    if identity.identifiers:
        return f'{company} careers "{identity.identifiers[0].value}" {location}'.strip()
    if record_kind in {"programme", "recruitment_campaign", "event", "challenge"}:
        anchor = _without_parenthetical(identity.business_unit or identity.title)
    else:
        anchor = _without_parenthetical(identity.title)
    return f'{company} careers "{anchor}" {location}'.strip()


def _site_query(identity: JobIdentity, host: str) -> str:
    location = identity.location or ""
    if identity.identifiers:
        anchor = identity.identifiers[0].value
    else:
        anchor = _without_parenthetical(identity.title)
    return f'site:{host} "{anchor}" {location}'.strip()


def _secondary_query(identity: JobIdentity) -> str:
    company = " ".join((identity.company or "").split())
    title = _without_parenthetical(identity.title)
    location = identity.location or ""
    return f'"{company}" "{title}" {location} LinkedIn Glassdoor'.strip()


def _search_round(
    identity: JobIdentity,
    query: str,
    round_number: int,
    scope: str,
) -> tuple[list[ResearchCandidate], ResearchTraceStep, str | None]:
    started = time.perf_counter()
    try:
        results = search_public_web(query, max_results=MAX_RESULTS_PER_QUERY)
        error = None
    except Exception as exc:
        results = []
        error = f"{scope} search failed: {type(exc).__name__}: {exc}"

    candidates = [
        candidate
        for result in results
        if (candidate := _from_search_result(identity, result)) is not None
    ]
    trace = ResearchTraceStep(
        round_number=round_number,
        scope=scope,
        query=query,
        results_returned=len(results),
        official_results=sum(1 for item in candidates if item.tier == "official"),
        secondary_results=sum(
            1 for item in candidates if item.tier in {"secondary", "institutional"}
        ),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )
    return candidates, trace, error


def _merge_candidates(values: list[ResearchCandidate]) -> list[ResearchCandidate]:
    merged: dict[str, ResearchCandidate] = {}
    for candidate in values:
        existing = merged.get(candidate.url)
        if existing is None or candidate.score > existing.score:
            merged[candidate.url] = candidate
    return sorted(
        merged.values(),
        key=lambda item: (
            item.tier == "official",
            item.tier == "institutional",
            item.score,
        ),
        reverse=True,
    )


def _to_v3_discovery(candidates: list[ResearchCandidate]) -> CandidateDiscoveryResult:
    converted: list[SearchCandidate] = []
    for candidate in candidates:
        if candidate.relation == "application":
            url_kind = "application_form"
        elif candidate.tier == "official":
            url_kind = "employer_or_ats"
        elif candidate.tier == "institutional":
            url_kind = "source_page"
        elif candidate.tier == "secondary":
            url_kind = "aggregator"
        else:
            url_kind = "unknown"

        converted.append(
            SearchCandidate(
                url=candidate.url,
                title=candidate.title,
                snippet=candidate.snippet,
                host=candidate.host,
                url_kind=url_kind,
                discovery_score=candidate.score,
                strategies=["direct_url"] if "source email" in " ".join(candidate.reasons).lower() else ["metadata"],
                reasons=candidate.reasons,
            )
        )
    return CandidateDiscoveryResult(candidates=converted)


def _official_context_support(
    identity: JobIdentity,
    candidate: ResearchCandidate,
    page: FetchedPage | None,
) -> tuple[bool, list[str]]:
    if candidate.tier != "official":
        return False, []

    haystack = f"{candidate.title} {candidate.snippet}"
    if page:
        haystack += f" {page.title} {page.text}"

    reasons: list[str] = []
    business_unit = _without_parenthetical(identity.business_unit)
    if business_unit and _normalize(business_unit) in _normalize(haystack):
        reasons.append(f"official source confirms programme/unit: {business_unit}")

    phrase_hits = [
        phrase
        for phrase in identity.distinctive_phrases
        if _normalize(_without_parenthetical(phrase)) in _normalize(haystack)
    ]
    if phrase_hits:
        reasons.append(f"official source matches {len(phrase_hits)} distinctive phrase(s)")

    overlap = _title_overlap(identity, haystack)
    if overlap >= 0.45:
        reasons.append(f"official source title/context overlap={overlap:.2f}")

    return bool(reasons), reasons


def _fetch_official_candidates(
    candidates: list[ResearchCandidate],
) -> tuple[dict[str, FetchedPage], list[str], int]:
    fetchable = [
        item
        for item in candidates
        if item.tier == "official" and item.relation != "application"
    ][:MAX_FETCHES]
    if not fetchable:
        return {}, [], 0

    pages: dict[str, FetchedPage] = {}
    warnings: list[str] = []
    with ThreadPoolExecutor(max_workers=len(fetchable)) as executor:
        futures = {
            executor.submit(fetch_public_page, item.url): item
            for item in fetchable
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                pages[item.url] = future.result()
            except Exception as exc:
                warnings.append(
                    f"official page unavailable: {item.url}: {type(exc).__name__}: {exc}"
                )
    return pages, warnings, len(fetchable)


def _outlook_web_link(email: EmailMessage) -> str | None:
    for url in email.links:
        try:
            parsed = urlparse(url)
        except ValueError:
            continue
        if parsed.netloc.lower() == "outlook.live.com" and "/owa" in parsed.path.lower():
            return url
    return None


def build_provenance(email: EmailMessage) -> SourceProvenance:
    return SourceProvenance(
        message_id=email.message_id,
        subject=email.subject,
        sender_name=email.sender_name,
        sender_email=email.sender_email,
        received_at=email.received_at,
        transport_sender_name=email.transport_sender_name,
        transport_sender_email=email.transport_sender_email,
        original_email_url=_outlook_web_link(email),
        attachment_names=email.attachments,
    )


def _trusted_source_support(identity: JobIdentity, email: EmailMessage) -> bool:
    sender = (email.sender_email or "").strip().lower()
    return (
        sender in Settings().trusted_senders
        and bool(identity.company)
        and bool(identity.title)
    )


def _application_url(candidates: list[ResearchCandidate]) -> str | None:
    return next((item.url for item in candidates if item.relation == "application"), None)


def focus_email_for_target(
    email: EmailMessage,
    company: str | None = None,
    title: str | None = None,
    radius: int = 3200,
) -> EmailMessage:
    """Create an in-memory target window for a large multi-job newsletter.

    This is a live-test/diagnostic helper. It never persists the focused body.
    The original EmailMessage should still be used for provenance output.
    """
    if not company and not title:
        return email

    text = email.body_text or ""
    lowered = text.lower()
    anchors = [value for value in (title, company) if value]
    index = -1
    for anchor in anchors:
        index = lowered.find(anchor.lower())
        if index >= 0:
            break

    if index < 0:
        focused_text = text[: radius * 2]
    else:
        start = max(0, index - radius)
        end = min(len(text), index + radius)
        focused_text = text[start:end]

    company_tokens = _company_tokens(company)
    focused_links: list[str] = []
    for url in email.links:
        if _is_noise_url(url):
            continue
        host_tokens = set(TOKEN_PATTERN.findall(urlparse(url).netloc.lower()))
        if company_tokens & host_tokens or _job_like_url(url) or _is_application_url(url):
            focused_links.append(url)

    return email.model_copy(
        update={
            "body_text": focused_text,
            "attachment_text": focused_text,
            "links": list(dict.fromkeys(focused_links)),
        }
    )


def research_opportunity(
    identity: JobIdentity,
    email: EmailMessage,
) -> OpportunityResearchPackage:
    """Official-first research for either a concrete job or a broader programme.

    Exact job verification is attempted against official company/ATS evidence
    before any LinkedIn/Glassdoor/university fallback is considered.
    """
    started = time.perf_counter()
    errors: list[str] = []
    warnings: list[str] = []
    trace: list[ResearchTraceStep] = []
    search_calls = 0
    fetch_calls = 0
    judge_llm_calls = 0

    record_kind = infer_initial_record_kind(identity)
    candidates = _direct_candidates(identity)
    application_url = _application_url(candidates)

    official = [item for item in candidates if item.tier == "official"]
    secondary = [
        item for item in candidates if item.tier in {"secondary", "institutional"}
    ]

    direct_official_job = next(
        (
            item
            for item in official
            if item.relation == "exact_posting"
        ),
        None,
    )

    if direct_official_job is None:
        found, step, error = _search_round(
            identity,
            _official_query(identity, record_kind),
            1,
            "official",
        )
        search_calls += 1
        trace.append(step)
        if error:
            errors.append(error)
        candidates = _merge_candidates([*candidates, *found])
        official = [item for item in candidates if item.tier == "official"]
        secondary = [
            item for item in candidates if item.tier in {"secondary", "institutional"}
        ]

        if official and search_calls < MAX_SEARCH_CALLS:
            host = official[0].host
            found, step, error = _search_round(
                identity,
                _site_query(identity, host),
                2,
                "official",
            )
            search_calls += 1
            trace.append(step)
            if error:
                errors.append(error)
            candidates = _merge_candidates([*candidates, *found])
            official = [item for item in candidates if item.tier == "official"]
            secondary = [
                item for item in candidates if item.tier in {"secondary", "institutional"}
            ]

    # Concrete-job path: official evidence gets first and exclusive chance.
    if record_kind == "job_posting" or direct_official_job is not None:
        official_discovery = _to_v3_discovery(
            [*official, *[item for item in candidates if item.relation == "application"]]
        )
        official_verification = verify_same_job(
            identity,
            official_discovery,
            email,
            enable_llm_judge=True,
        )
        fetch_calls += official_verification.metrics.fetch_calls
        judge_llm_calls += official_verification.metrics.llm_calls
        warnings.extend(official_verification.warnings)
        errors.extend(official_verification.errors)

        if official_verification.identity_status == "verified":
            return OpportunityResearchPackage(
                identity=identity,
                record_kind="job_posting",
                status="verified_exact_job",
                confidence=official_verification.confidence,
                basis=official_verification.identity_basis,
                provenance=build_provenance(email),
                official_job_url=official_verification.official_url,
                application_url=official_verification.application_url or application_url,
                candidates=candidates,
                trace=trace,
                evidence_summary=official_verification.matched_evidence,
                warnings=list(dict.fromkeys(warnings)),
                errors=list(dict.fromkeys(errors)),
                metrics=ResearchMetrics(
                    search_calls=search_calls,
                    fetch_calls=fetch_calls,
                    judge_llm_calls=judge_llm_calls,
                    elapsed_ms=int((time.perf_counter() - started) * 1000),
                ),
            )

        # Strong direct official URL with a jobId can survive dynamic-page fetch
        # failures because the trusted source tied that exact URL to this role.
        if direct_official_job and extract_job_id_from_url(direct_official_job.url):
            job_id = extract_job_id_from_url(direct_official_job.url)
            return OpportunityResearchPackage(
                identity=identity,
                record_kind="job_posting",
                status="verified_exact_job",
                confidence="medium",
                basis="trusted_source_direct_official_job_url",
                provenance=build_provenance(email),
                official_job_url=direct_official_job.url,
                application_url=application_url or direct_official_job.url,
                candidates=candidates,
                trace=trace,
                evidence_summary=[
                    f"trusted source supplied employer careers URL with jobId={job_id}"
                ],
                warnings=list(dict.fromkeys(warnings)),
                errors=list(dict.fromkeys(errors)),
                metrics=ResearchMetrics(
                    search_calls=search_calls,
                    fetch_calls=fetch_calls,
                    judge_llm_calls=judge_llm_calls,
                    elapsed_ms=int((time.perf_counter() - started) * 1000),
                ),
            )

    # Unknown/internship path: official programme/company context can be enough
    # without pretending an exact public JD exists.
    pages, page_warnings, page_fetches = _fetch_official_candidates(official)
    fetch_calls += page_fetches
    warnings.extend(page_warnings)
    context_urls: list[str] = []
    context_reasons: list[str] = []
    for candidate in official:
        supported, reasons = _official_context_support(
            identity,
            candidate,
            pages.get(candidate.url),
        )
        if supported:
            context_urls.append(candidate.url)
            context_reasons.extend(reasons)

    if context_urls:
        return OpportunityResearchPackage(
            identity=identity,
            record_kind=(
                record_kind
                if record_kind not in {"unknown", "job_posting"}
                else "programme"
            ),
            status="official_context_supported",
            confidence="high" if len(context_reasons) >= 2 else "medium",
            basis="official_company_context",
            provenance=build_provenance(email),
            official_background_urls=list(dict.fromkeys(context_urls)),
            application_url=application_url,
            candidates=candidates,
            trace=trace,
            evidence_summary=list(dict.fromkeys(context_reasons)),
            warnings=list(dict.fromkeys(warnings)),
            errors=list(dict.fromkeys(errors)),
            metrics=ResearchMetrics(
                search_calls=search_calls,
                fetch_calls=fetch_calls,
                judge_llm_calls=judge_llm_calls,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            ),
        )

    # Only after official evidence fails do we spend the final search round on
    # LinkedIn/Glassdoor/university/public secondary evidence.
    if search_calls < MAX_SEARCH_CALLS:
        found, step, error = _search_round(
            identity,
            _secondary_query(identity),
            search_calls + 1,
            "secondary",
        )
        search_calls += 1
        trace.append(step)
        if error:
            errors.append(error)
        candidates = _merge_candidates([*candidates, *found])
        secondary = [
            item for item in candidates if item.tier in {"secondary", "institutional"}
        ]

    if secondary and record_kind == "job_posting":
        secondary_verification = verify_same_job(
            identity,
            _to_v3_discovery(secondary),
            email,
            enable_llm_judge=True,
        )
        fetch_calls += secondary_verification.metrics.fetch_calls
        judge_llm_calls += secondary_verification.metrics.llm_calls
        warnings.extend(secondary_verification.warnings)
        errors.extend(secondary_verification.errors)
        if secondary_verification.identity_status == "verified":
            return OpportunityResearchPackage(
                identity=identity,
                record_kind="job_posting",
                status="secondary_corroborated",
                confidence="medium",
                basis="secondary_same_job_evidence",
                provenance=build_provenance(email),
                secondary_evidence_urls=[
                    secondary_verification.matched_candidate_url
                ] if secondary_verification.matched_candidate_url else [],
                application_url=application_url,
                candidates=candidates,
                trace=trace,
                evidence_summary=secondary_verification.matched_evidence,
                warnings=list(dict.fromkeys(warnings)),
                errors=list(dict.fromkeys(errors)),
                metrics=ResearchMetrics(
                    search_calls=search_calls,
                    fetch_calls=fetch_calls,
                    judge_llm_calls=judge_llm_calls,
                    elapsed_ms=int((time.perf_counter() - started) * 1000),
                ),
            )

    if _trusted_source_support(identity, email):
        return OpportunityResearchPackage(
            identity=identity,
            record_kind=record_kind,
            status="source_verified",
            confidence="high",
            basis="trusted_nus_email",
            provenance=build_provenance(email),
            secondary_evidence_urls=[item.url for item in secondary[:3]],
            application_url=application_url,
            candidates=candidates,
            trace=trace,
            evidence_summary=[
                "concrete opportunity came from a configured trusted NUS career source"
            ],
            warnings=list(dict.fromkeys(warnings)),
            errors=list(dict.fromkeys(errors)),
            metrics=ResearchMetrics(
                search_calls=search_calls,
                fetch_calls=fetch_calls,
                judge_llm_calls=judge_llm_calls,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            ),
        )

    return OpportunityResearchPackage(
        identity=identity,
        record_kind=record_kind,
        status="unresolved",
        confidence="low",
        basis="none",
        provenance=build_provenance(email),
        secondary_evidence_urls=[item.url for item in secondary[:3]],
        application_url=application_url,
        candidates=candidates,
        trace=trace,
        warnings=list(dict.fromkeys(warnings)),
        errors=list(dict.fromkeys(errors)),
        metrics=ResearchMetrics(
            search_calls=search_calls,
            fetch_calls=fetch_calls,
            judge_llm_calls=judge_llm_calls,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        ),
    )
