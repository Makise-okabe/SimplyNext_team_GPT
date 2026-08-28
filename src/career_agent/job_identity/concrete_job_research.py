from __future__ import annotations

import re
import time
from urllib.parse import unquote, urlparse

from career_agent.config import Settings
from career_agent.job_identity.official_research import (
    _application_url,
    _direct_candidates,
    _job_like_url,
    _merge_candidates,
    _search_round,
    _secondary_query,
    _to_v3_discovery,
    build_provenance,
    infer_initial_record_kind,
    research_opportunity as legacy_research_opportunity,
)
from career_agent.job_identity.verify_same_job import verify_same_job
from career_agent.models.email import EmailMessage
from career_agent.models.job_identity import JobIdentity
from career_agent.models.opportunity_research import (
    OpportunityResearchPackage,
    ResearchCandidate,
    ResearchMetrics,
)

SENIORITY_WORDS = {
    "senior",
    "staff",
    "principal",
    "lead",
    "junior",
    "associate",
    "sr",
    "jr",
}
COMPANY_ALIAS_STOPWORDS = {
    "the",
    "company",
    "limited",
    "ltd",
    "pte",
    "inc",
    "corp",
    "corporation",
    "private",
    "asia",
    "singapore",
    "holdings",
    "holding",
}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
YEAR_PATTERN = re.compile(r"\b20\d{2}\b")

# Round 1 may stop the search only when the candidate is already a strong,
# official, job-like match. The score is not used alone: title and core-title
# overlap must also clear their thresholds.
EARLY_STOP_SCORE = 70.0
EARLY_STOP_FULL_TITLE_OVERLAP = 0.80
EARLY_STOP_CORE_TITLE_OVERLAP = 0.90


def _clean_company(company: str | None) -> str:
    value = " ".join((company or "").split())
    for suffix in (
        " Pte Ltd",
        " Pte. Ltd.",
        " Private Limited",
        " Limited",
        " Ltd",
        " Asia",
    ):
        if value.lower().endswith(suffix.lower()):
            value = value[: -len(suffix)].strip()
    return value


def _core_title(title: str | None) -> str:
    tokens = [token for token in (title or "").replace("/", " ").split() if token]
    kept = [
        token
        for token in tokens
        if token.lower().strip(",.-") not in SENIORITY_WORDS
    ]
    if len(kept) >= 2:
        return " ".join(kept)
    return " ".join(tokens)


def _company_host_aliases(company: str | None) -> set[str]:
    """Derive conservative brand/domain aliases from a legal company name."""
    tokens = [
        token
        for token in TOKEN_PATTERN.findall((company or "").lower())
        if token not in COMPANY_ALIAS_STOPWORDS
    ]
    aliases = {token for token in tokens if len(token) >= 3}
    if len(tokens) >= 2:
        acronym = "".join(token[0] for token in tokens if token)
        if len(acronym) >= 3:
            aliases.add(acronym)
    return aliases


def _host_tokens(host: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(host.lower()))


def _promote_brand_official_candidates(
    identity: JobIdentity,
    candidates: list[ResearchCandidate],
) -> list[ResearchCandidate]:
    """Promote a company-owned acronym domain missed by the base classifier."""
    aliases = _company_host_aliases(identity.company)
    if not aliases:
        return candidates

    promoted: list[ResearchCandidate] = []
    for candidate in candidates:
        if candidate.tier != "weak" or not (aliases & _host_tokens(candidate.host)):
            promoted.append(candidate)
            continue

        relation = "exact_posting" if _job_like_url(candidate.url) else "official_background"
        promoted.append(
            candidate.model_copy(
                update={
                    "tier": "official",
                    "relation": relation,
                    "score": min(100.0, candidate.score + 40.0),
                    "reasons": list(
                        dict.fromkeys(
                            [
                                *candidate.reasons,
                                "company brand/acronym matched official host",
                            ]
                        )
                    ),
                }
            )
        )
    return promoted


def _refresh_trace_counts(step, candidates: list[ResearchCandidate]) -> None:
    step.official_results = sum(1 for item in candidates if item.tier == "official")
    step.secondary_results = sum(
        1 for item in candidates if item.tier in {"secondary", "institutional"}
    )


def _title_tokens(value: str | None) -> set[str]:
    return {
        token
        for token in TOKEN_PATTERN.findall((value or "").lower())
        if len(token) >= 3 or token.isdigit()
    }


def _candidate_metadata_text(candidate: ResearchCandidate) -> str:
    try:
        parsed = urlparse(candidate.url)
        url_text = unquote(f"{parsed.netloc} {parsed.path} {parsed.query}")
    except ValueError:
        url_text = candidate.url
    return f"{candidate.title} {candidate.snippet} {url_text}"


def _metadata_title_overlap(title: str | None, candidate: ResearchCandidate) -> float:
    source = _title_tokens(title)
    if not source:
        return 0.0
    return len(source & _title_tokens(_candidate_metadata_text(candidate))) / len(source)


def _official_metadata_match(
    identity: JobIdentity,
    candidate: ResearchCandidate,
) -> bool:
    """Return True for a strong official job-result metadata match."""
    if candidate.tier != "official" or candidate.relation != "exact_posting":
        return False
    if candidate.score < EARLY_STOP_SCORE:
        return False

    full_overlap = _metadata_title_overlap(identity.title, candidate)
    core_overlap = _metadata_title_overlap(_core_title(identity.title), candidate)
    return (
        full_overlap >= EARLY_STOP_FULL_TITLE_OVERLAP
        and core_overlap >= EARLY_STOP_CORE_TITLE_OVERLAP
    )


def _best_strong_official_candidate(
    identity: JobIdentity,
    candidates: list[ResearchCandidate],
) -> ResearchCandidate | None:
    strong = [
        candidate
        for candidate in candidates
        if _official_metadata_match(identity, candidate)
    ]
    if not strong:
        return None
    return max(
        strong,
        key=lambda candidate: (
            _metadata_title_overlap(identity.title, candidate),
            candidate.score,
        ),
    )


def _trusted_source(email: EmailMessage) -> bool:
    return (email.sender_email or "").strip().lower() in Settings().trusted_senders


def _blocking_conflicts(identity: JobIdentity, verification) -> list[str]:
    """Return only conflicts that should block the V5 metadata fallback.

    A future recruiting year explicitly named by the source role is corroboration,
    not a conflict with the email receipt year. A different candidate cycle still
    blocks the fallback.
    """
    identity_years = set(YEAR_PATTERN.findall(identity.title or ""))
    blocking: list[str] = []

    for evaluation in verification.evaluations:
        for conflict in evaluation.hard_conflicts:
            if conflict.startswith("recruiting-cycle conflict:") and identity_years:
                candidate_part = conflict.split("candidate title year(s)=", 1)
                if len(candidate_part) == 2:
                    candidate_years = set(YEAR_PATTERN.findall(candidate_part[1]))
                    if candidate_years and candidate_years <= identity_years:
                        continue
            blocking.append(conflict)
    return blocking


def _official_exact_query(identity: JobIdentity) -> str:
    company = _clean_company(identity.company)
    title = " ".join((identity.title or "").split())
    location = identity.location or ""
    return f'{company} careers "{title}" {location}'.strip()


def _official_broad_query(identity: JobIdentity) -> str:
    company = _clean_company(identity.company)
    core_title = _core_title(identity.title)
    location = identity.location or "Singapore"
    return f'{company} careers "{core_title}" {location}'.strip()


def _package_from_verification(
    *,
    identity: JobIdentity,
    email: EmailMessage,
    verification,
    candidates: list[ResearchCandidate],
    trace,
    search_calls: int,
    fetch_calls: int,
    judge_llm_calls: int,
    started: float,
    status: str,
    confidence: str | None = None,
    basis: str | None = None,
) -> OpportunityResearchPackage:
    return OpportunityResearchPackage(
        identity=identity,
        record_kind="job_posting",
        status=status,
        confidence=confidence or verification.confidence,
        basis=basis or verification.identity_basis,
        provenance=build_provenance(email),
        official_job_url=(
            verification.official_url if status == "verified_exact_job" else None
        ),
        secondary_evidence_urls=(
            [verification.matched_candidate_url]
            if status == "secondary_corroborated"
            and verification.matched_candidate_url
            else []
        ),
        application_url=verification.application_url or _application_url(candidates),
        candidates=candidates,
        trace=trace,
        evidence_summary=verification.matched_evidence,
        warnings=list(dict.fromkeys(verification.warnings)),
        errors=list(dict.fromkeys(verification.errors)),
        metrics=ResearchMetrics(
            search_calls=search_calls,
            fetch_calls=fetch_calls,
            judge_llm_calls=judge_llm_calls,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        ),
    )


def _official_metadata_package(
    *,
    identity: JobIdentity,
    email: EmailMessage,
    candidate: ResearchCandidate,
    candidates: list[ResearchCandidate],
    trace,
    search_calls: int,
    fetch_calls: int,
    judge_llm_calls: int,
    warnings: list[str],
    errors: list[str],
    started: float,
) -> OpportunityResearchPackage:
    full_overlap = _metadata_title_overlap(identity.title, candidate)
    warnings = [
        *warnings,
        "V3 fetched-content verification was inconclusive; V5 accepted a medium-confidence exact-job match from trusted source + strong official job metadata",
    ]
    return OpportunityResearchPackage(
        identity=identity,
        record_kind="job_posting",
        status="verified_exact_job",
        confidence="medium",
        basis="official_job_metadata_match",
        provenance=build_provenance(email),
        official_job_url=candidate.url,
        application_url=candidate.url,
        candidates=candidates,
        trace=trace,
        evidence_summary=[
            f"trusted NUS career source names the role: {identity.title}",
            f"official employer/ATS job metadata title overlap={full_overlap:.2f}",
            f"official job URL: {candidate.url}",
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


def _verify_official_candidates(
    *,
    identity: JobIdentity,
    email: EmailMessage,
    official: list[ResearchCandidate],
    candidates: list[ResearchCandidate],
    trace,
    search_calls: int,
    fetch_calls: int,
    judge_llm_calls: int,
    warnings: list[str],
    errors: list[str],
    started: float,
) -> tuple[OpportunityResearchPackage | None, int, int, list[str], list[str]]:
    """Verify exact official job candidates and apply the bounded V5 fallback."""
    exact = [item for item in official if item.relation == "exact_posting"]
    if not exact:
        return None, fetch_calls, judge_llm_calls, warnings, errors

    verification = verify_same_job(
        identity,
        _to_v3_discovery(exact),
        email,
        enable_llm_judge=True,
    )
    fetch_calls += verification.metrics.fetch_calls
    judge_llm_calls += verification.metrics.llm_calls
    warnings = [*warnings, *verification.warnings]
    errors = [*errors, *verification.errors]
    verification.warnings = list(dict.fromkeys(warnings))
    verification.errors = list(dict.fromkeys(errors))

    if verification.identity_status == "verified":
        return (
            _package_from_verification(
                identity=identity,
                email=email,
                verification=verification,
                candidates=candidates,
                trace=trace,
                search_calls=search_calls,
                fetch_calls=fetch_calls,
                judge_llm_calls=judge_llm_calls,
                started=started,
                status="verified_exact_job",
            ),
            fetch_calls,
            judge_llm_calls,
            warnings,
            errors,
        )

    metadata_candidate = _best_strong_official_candidate(identity, exact)
    if (
        metadata_candidate
        and _trusted_source(email)
        and not _blocking_conflicts(identity, verification)
    ):
        return (
            _official_metadata_package(
                identity=identity,
                email=email,
                candidate=metadata_candidate,
                candidates=candidates,
                trace=trace,
                search_calls=search_calls,
                fetch_calls=fetch_calls,
                judge_llm_calls=judge_llm_calls,
                warnings=warnings,
                errors=errors,
                started=started,
            ),
            fetch_calls,
            judge_llm_calls,
            warnings,
            errors,
        )

    return None, fetch_calls, judge_llm_calls, warnings, errors


def research_concrete_job_or_delegate(
    identity: JobIdentity,
    email: EmailMessage,
) -> OpportunityResearchPackage:
    """Official-first concrete-job research with confidence-based early stopping."""
    record_kind = infer_initial_record_kind(identity)
    if record_kind != "job_posting":
        return legacy_research_opportunity(identity, email)

    base_direct = _direct_candidates(identity)
    direct = _promote_brand_official_candidates(identity, base_direct)

    # Preserve the already-tested direct-official path (e.g. IBM jobId URL).
    if any(
        item.tier == "official" and item.relation == "exact_posting"
        for item in base_direct
    ):
        return legacy_research_opportunity(identity, email)

    started = time.perf_counter()
    candidates = direct
    trace = []
    errors: list[str] = []
    warnings: list[str] = []
    search_calls = 0
    fetch_calls = 0
    judge_llm_calls = 0

    # Round 1: exact employer/ATS title search.
    found, step, error = _search_round(
        identity,
        _official_exact_query(identity),
        1,
        "official",
    )
    found = _promote_brand_official_candidates(identity, found)
    _refresh_trace_counts(step, found)
    search_calls += 1
    trace.append(step)
    if error:
        errors.append(error)
    candidates = _merge_candidates([*candidates, *found])

    # EARLY STOP: if Round 1 already contains a strong official exact posting,
    # verify only that exact candidate set now. Do not spend Round 2 unless the
    # first result is too weak, ambiguous, or conflicted.
    round1_exact = [
        item
        for item in found
        if item.tier == "official" and item.relation == "exact_posting"
    ]
    if _best_strong_official_candidate(identity, round1_exact):
        package, fetch_calls, judge_llm_calls, warnings, errors = _verify_official_candidates(
            identity=identity,
            email=email,
            official=round1_exact,
            candidates=candidates,
            trace=trace,
            search_calls=search_calls,
            fetch_calls=fetch_calls,
            judge_llm_calls=judge_llm_calls,
            warnings=warnings,
            errors=errors,
            started=started,
        )
        if package is not None:
            return package

    # Round 2: only when Round 1 was not strong enough to finish safely. This
    # broadens seniority/title variants while remaining official-first.
    found, step, error = _search_round(
        identity,
        _official_broad_query(identity),
        2,
        "official",
    )
    found = _promote_brand_official_candidates(identity, found)
    _refresh_trace_counts(step, found)
    search_calls += 1
    trace.append(step)
    if error:
        errors.append(error)
    candidates = _merge_candidates([*candidates, *found])

    official = [item for item in candidates if item.tier == "official"]
    package, fetch_calls, judge_llm_calls, warnings, errors = _verify_official_candidates(
        identity=identity,
        email=email,
        official=official,
        candidates=candidates,
        trace=trace,
        search_calls=search_calls,
        fetch_calls=fetch_calls,
        judge_llm_calls=judge_llm_calls,
        warnings=warnings,
        errors=errors,
        started=started,
    )
    if package is not None:
        return package

    # Round 3: secondary evidence is allowed only after official attempts fail.
    found, step, error = _search_round(
        identity,
        _secondary_query(identity),
        3,
        "secondary",
    )
    found = _promote_brand_official_candidates(identity, found)
    _refresh_trace_counts(step, found)
    search_calls += 1
    trace.append(step)
    if error:
        errors.append(error)
    candidates = _merge_candidates([*candidates, *found])

    # A secondary query can occasionally surface the employer page. Give that
    # newly discovered official result one chance before aggregators.
    newly_official = [item for item in found if item.tier == "official"]
    if newly_official:
        package, fetch_calls, judge_llm_calls, warnings, errors = _verify_official_candidates(
            identity=identity,
            email=email,
            official=newly_official,
            candidates=candidates,
            trace=trace,
            search_calls=search_calls,
            fetch_calls=fetch_calls,
            judge_llm_calls=judge_llm_calls,
            warnings=warnings,
            errors=errors,
            started=started,
        )
        if package is not None:
            return package

    secondary = [
        item for item in candidates if item.tier in {"secondary", "institutional"}
    ]
    if secondary:
        verification = verify_same_job(
            identity,
            _to_v3_discovery(secondary),
            email,
            enable_llm_judge=True,
        )
        fetch_calls += verification.metrics.fetch_calls
        judge_llm_calls += verification.metrics.llm_calls
        warnings.extend(verification.warnings)
        errors.extend(verification.errors)
        verification.warnings = list(dict.fromkeys(warnings))
        verification.errors = list(dict.fromkeys(errors))
        if verification.identity_status == "verified":
            return _package_from_verification(
                identity=identity,
                email=email,
                verification=verification,
                candidates=candidates,
                trace=trace,
                search_calls=search_calls,
                fetch_calls=fetch_calls,
                judge_llm_calls=judge_llm_calls,
                started=started,
                status="secondary_corroborated",
                confidence="medium",
                basis="secondary_same_job_evidence",
            )

    if _trusted_source(email):
        return OpportunityResearchPackage(
            identity=identity,
            record_kind="job_posting",
            status="source_verified",
            confidence="medium",
            basis="trusted_nus_email",
            provenance=build_provenance(email),
            secondary_evidence_urls=[item.url for item in secondary[:3]],
            application_url=_application_url(candidates),
            candidates=candidates,
            trace=trace,
            evidence_summary=[
                "trusted NUS career source confirms the role was circulated; exact current employer posting was not verified"
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
        record_kind="job_posting",
        status="unresolved",
        confidence="low",
        basis="none",
        provenance=build_provenance(email),
        secondary_evidence_urls=[item.url for item in secondary[:3]],
        application_url=_application_url(candidates),
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
