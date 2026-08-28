from __future__ import annotations

import time

from career_agent.job_identity.official_research import (
    _application_url,
    _direct_candidates,
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

    direct = _direct_candidates(identity)
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
    search_calls = 0

    # Round 1: employer/ATS exact title.
    found, step, error = _search_round(
        identity,
        _official_exact_query(identity),
        1,
        "official",
    )
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
        verification.errors.extend(errors)
        if verification.identity_status == "verified":
            return _package_from_verification(
                identity=identity,
                email=email,
                verification=verification,
                candidates=candidates,
                trace=trace,
                search_calls=search_calls,
                started=started,
                status="verified_exact_job",
            )

    # Round 3: only after both official attempts fail to verify the same job.
    found, step, error = _search_round(
        identity,
        _secondary_query(identity),
        3,
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
    if secondary:
        verification = verify_same_job(
            identity,
            _to_v3_discovery(secondary),
            email,
            enable_llm_judge=True,
        )
        verification.errors.extend(errors)
        if verification.identity_status == "verified":
            return _package_from_verification(
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
        errors=list(dict.fromkeys(errors)),
        metrics=ResearchMetrics(
            search_calls=search_calls,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        ),
    )
