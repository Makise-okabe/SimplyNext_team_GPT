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
GENERIC_TITLE_WORDS = {"engineer", "manager", "developer", "analyst", "consultant"}
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
    """Derive conservative brand/domain aliases from a legal company name.

    Example: `THE BOSTON CONSULTING GROUP` -> `bcg`. We keep ordinary meaningful
    words too, while only generating acronyms of at least three characters to
    avoid promoting unrelated two-letter domains.
    """
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
    """Promote company-owned acronym domains missed by the base classifier.

    Search still discovers the URL; this helper only fixes evidence ownership.
    It never promotes known secondary/institutional results and requires an exact
    host-token match to a company-derived alias such as `bcg` in `careers.bcg.com`.
    """
    aliases = _company_host_aliases(identity.company)
    if not aliases:
        return candidates

    promoted: list[ResearchCandidate] = []
    for candidate in candidates:
        if candidate.tier != "weak" or not (aliases & _host_tokens(candidate.host)):
            promoted.append(candidate)
            continue

        relation = "exact_posting" if _job_like_url(candidate.url) else "official_background"
        reasons = [*candidate.reasons, "company brand/acronym matched official host"]
        promoted.append(
            candidate.model_copy(
                update={
                    "tier": "official",
                    "relation": relation,
                    "score": min(100.0, candidate.score + 40.0),
                    "reasons": list(dict.fromkeys(reasons)),
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
    """Conservative V5 fallback for strong official job-result metadata.

    Normal V3 fetched-content verification always runs first. This fallback only
    applies when that verification is inconclusive and the search result itself is
    an official job-like page with strong title agreement. It is useful for
    dynamic employer sites that expose the exact role in indexed metadata but do
    not provide enough static body text for V3's stricter JD-content guardrails.
    """
    if candidate.tier != "official" or candidate.relation != "exact_posting":
        return False
    if candidate.score < 70:
        return False

    full_overlap = _metadata_title_overlap(identity.title, candidate)
    core_overlap = _metadata_title_overlap(_core_title(identity.title), candidate)
    return full_overlap >= 0.80 and core_overlap >= 0.90


def _trusted_source(email: EmailMessage) -> bool:
    return (email.sender_email or "").strip().lower() in Settings().trusted_senders


def _blocking_conflicts(identity: JobIdentity, verification) -> list[str]:
    """Return only conflicts that should block the V5 metadata fallback.

    V3 intentionally compares candidate recruiting years to the email receipt
    year. That is conservative, but graduate recruiting commonly opens one year
    early. If the source identity itself explicitly names the same future cycle
    (e.g. a 2026 email for `Associate, Singapore (2027)`), the year difference is
    corroborated rather than contradictory. Different candidate years still block.
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
    candidates,
    trace,
    search_calls: int,
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
            if status == "secondary_corroborated" and verification.matched_candidate_url
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
            fetch_calls=verification.metrics.fetch_calls,
            judge_llm_calls=verification.metrics.llm_calls,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        ),
    )


def research_concrete_job_or_delegate(
    identity: JobIdentity,
    email: EmailMessage,
) -> OpportunityResearchPackage:
    """Harden V5 concrete-job research while preserving programme behaviour.

    Direct official links remain on the already-tested legacy path. Concrete jobs
    without a direct employer link get two official search rounds before any
    secondary source is allowed to participate.
    """
    record_kind = infer_initial_record_kind(identity)
    if record_kind != "job_posting":
        return legacy_research_opportunity(identity, email)

    direct = _promote_brand_official_candidates(identity, _direct_candidates(identity))
    direct_official = [
        item
        for item in direct
        if item.tier == "official" and item.relation == "exact_posting"
    ]
    if direct_official:
        return legacy_research_opportunity(identity, email)

    started = time.perf_counter()
    candidates = direct
    trace = []
    errors: list[str] = []
    warnings: list[str] = []
    search_calls = 0
    fetch_calls = 0
    judge_llm_calls = 0

    # Round 1: employer/ATS exact title.
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

    # Round 2: still official, but tolerate employer title variants such as
    # "Staff Analog Layout Engineer" vs "Senior Staff Analog Layout Engineer".
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
    if official:
        verification = verify_same_job(
            identity,
            _to_v3_discovery(official),
            email,
            enable_llm_judge=True,
        )
        fetch_calls += verification.metrics.fetch_calls
        judge_llm_calls += verification.metrics.llm_calls
        warnings.extend(verification.warnings)
        errors.extend(verification.errors)
        verification.errors = list(dict.fromkeys([*verification.errors, *errors]))
        verification.warnings = list(dict.fromkeys([*verification.warnings, *warnings]))
        if verification.identity_status == "verified":
            package = _package_from_verification(
                identity=identity,
                email=email,
                verification=verification,
                candidates=candidates,
                trace=trace,
                search_calls=search_calls,
                started=started,
                status="verified_exact_job",
            )
            package.metrics.fetch_calls = fetch_calls
            package.metrics.judge_llm_calls = judge_llm_calls
            return package

        # Keep sealed V3 conservative. V5 may still accept a bounded metadata
        # match when the trusted source and a strong official job result agree and
        # V3 observed no genuine identifier/year conflict.
        metadata_candidate = next(
            (candidate for candidate in official if _official_metadata_match(identity, candidate)),
            None,
        )
        if metadata_candidate and _trusted_source(email) and not _blocking_conflicts(identity, verification):
            return _official_metadata_package(
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
            )

    # Round 3: only after both official attempts fail to verify the same job.
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

    # A supposedly secondary search can still surface an employer page. Give any
    # newly discovered official page one verification chance before aggregators.
    newly_official = [item for item in found if item.tier == "official"]
    if newly_official:
        verification = verify_same_job(
            identity,
            _to_v3_discovery(newly_official),
            email,
            enable_llm_judge=True,
        )
        fetch_calls += verification.metrics.fetch_calls
        judge_llm_calls += verification.metrics.llm_calls
        warnings.extend(verification.warnings)
        errors.extend(verification.errors)
        if verification.identity_status == "verified":
            verification.errors = list(dict.fromkeys([*verification.errors, *errors]))
            verification.warnings = list(dict.fromkeys([*verification.warnings, *warnings]))
            package = _package_from_verification(
                identity=identity,
                email=email,
                verification=verification,
                candidates=candidates,
                trace=trace,
                search_calls=search_calls,
                started=started,
                status="verified_exact_job",
            )
            package.metrics.fetch_calls = fetch_calls
            package.metrics.judge_llm_calls = judge_llm_calls
            return package

        metadata_candidate = next(
            (
                candidate
                for candidate in newly_official
                if _official_metadata_match(identity, candidate)
            ),
            None,
        )
        if metadata_candidate and _trusted_source(email) and not _blocking_conflicts(identity, verification):
            return _official_metadata_package(
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
            )

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
        verification.errors = list(dict.fromkeys([*verification.errors, *errors]))
        verification.warnings = list(dict.fromkeys([*verification.warnings, *warnings]))
        if verification.identity_status == "verified":
            package = _package_from_verification(
                identity=identity,
                email=email,
                verification=verification,
                candidates=candidates,
                trace=trace,
                search_calls=search_calls,
                started=started,
                status="secondary_corroborated",
                confidence="medium",
                basis="secondary_same_job_evidence",
            )
            package.metrics.fetch_calls = fetch_calls
            package.metrics.judge_llm_calls = judge_llm_calls
            return package

    # NUS source still establishes that the opportunity was genuinely circulated,
    # but for a concrete job this is not the same as current exact-job verification.
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
