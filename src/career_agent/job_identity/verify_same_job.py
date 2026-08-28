from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from career_agent.config import Settings
from career_agent.job_identity.extract_identity import extract_identifiers
from career_agent.models.email import EmailMessage
from career_agent.models.job_identity import JobIdentity
from career_agent.models.job_search import CandidateDiscoveryResult, SearchCandidate
from career_agent.models.job_verification import (
    CandidateEvaluation,
    SameJobJudgeOutput,
    SameJobVerificationResult,
    VerificationMetrics,
)
from career_agent.tools.web_fetch import FetchedPage, fetch_public_page

MAX_FETCH_CANDIDATES = 4
MAX_PARALLEL_FETCHES = 4
MAX_JUDGE_CANDIDATES = 2
MAX_JUDGE_PAGE_CHARS = 2600

GENERIC_WORDS = {
    "and",
    "the",
    "for",
    "with",
    "company",
    "limited",
    "ltd",
    "pte",
}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
PARENTHETICAL = re.compile(r"\s*\([^)]{1,30}\)\s*")


def _normalize(value: str | None) -> str:
    return " ".join((value or "").lower().split())


def _without_parenthetical(value: str | None) -> str:
    return " ".join(PARENTHETICAL.sub(" ", value or "").split())


def _tokens(value: str | None) -> set[str]:
    return {
        token
        for token in TOKEN_PATTERN.findall(_normalize(value))
        if len(token) >= 3 and token not in GENERIC_WORDS
    }


def _overlap_ratio(needle: str | None, haystack: str) -> float:
    source = _tokens(needle)
    if not source:
        return 0.0
    hits = source & _tokens(haystack)
    return len(hits) / len(source)


def _company_match(identity: JobIdentity, haystack: str) -> bool:
    tokens = _tokens(identity.company)
    if not tokens:
        return False
    hits = tokens & _tokens(haystack)
    # Company names often contain a generic suffix; one distinctive token is
    # sufficient when other independent evidence is also required later.
    return bool(hits)


def _same_kind_identifier_conflicts(identity: JobIdentity, page_text: str) -> list[str]:
    if not identity.identifiers:
        return []

    source_by_kind: dict[str, set[str]] = {}
    for identifier in identity.identifiers:
        source_by_kind.setdefault(identifier.kind, set()).add(identifier.value.lower())

    page_identifiers = extract_identifiers(page_text)
    page_by_kind: dict[str, set[str]] = {}
    for identifier in page_identifiers:
        page_by_kind.setdefault(identifier.kind, set()).add(identifier.value.lower())

    conflicts: list[str] = []
    for kind, source_values in source_by_kind.items():
        page_values = page_by_kind.get(kind)
        if page_values and not (source_values & page_values):
            conflicts.append(
                f"{kind} conflict: source={sorted(source_values)} page={sorted(page_values)}"
            )
    return conflicts


def _evaluate_page(
    identity: JobIdentity,
    candidate: SearchCandidate,
    page: FetchedPage,
) -> CandidateEvaluation:
    haystack = f"{page.title}\n{page.text}\n{page.final_url}"
    normalized = _normalize(haystack)

    identifier_hits = [
        identifier.value
        for identifier in identity.identifiers
        if identifier.value.lower() in normalized
    ]
    conflicts = _same_kind_identifier_conflicts(identity, haystack)
    company_match = _company_match(identity, haystack)
    title_overlap = _overlap_ratio(_without_parenthetical(identity.title), haystack)
    location_match = bool(identity.location and _normalize(identity.location) in normalized)

    business_unit = _without_parenthetical(identity.business_unit)
    business_unit_match = bool(
        business_unit and _normalize(business_unit) in normalized
    )

    phrase_hits: list[str] = []
    for phrase in identity.distinctive_phrases:
        cleaned = _without_parenthetical(phrase)
        if cleaned and _normalize(cleaned) in normalized:
            phrase_hits.append(phrase)

    score = 0.0
    reasons: list[str] = []

    if identifier_hits:
        score += 70
        reasons.append("exact source job identifier appears on fetched page")
    if company_match:
        score += 14
        reasons.append("company identity matched")
    if title_overlap:
        score += 28 * title_overlap
        reasons.append(f"title token overlap={title_overlap:.2f}")
    if location_match:
        score += 7
        reasons.append("location matched")
    if business_unit_match:
        score += 12
        reasons.append("business unit matched")
    if phrase_hits:
        score += min(30, 10 * len(phrase_hits))
        reasons.append(f"{len(phrase_hits)} distinctive JD phrase(s) matched")
    if candidate.url_kind == "employer_or_ats":
        score += 6
        reasons.append("employer/ATS-like host")
    if "direct_url" in candidate.strategies and candidate.url_kind == "employer_or_ats":
        score += 8
        reasons.append("employer/ATS URL supplied directly by source")
    if candidate.url_kind == "aggregator":
        score -= 8
        reasons.append("aggregator evidence is weaker")

    if conflicts:
        score -= 100
        reasons.append("explicit identifier conflict detected")

    score = max(0.0, min(100.0, score))

    if conflicts:
        decision = "reject"
        confidence = "high"
    elif identifier_hits and company_match:
        decision = "same_job"
        confidence = "high"
    elif (
        company_match
        and title_overlap >= 0.75
        and (business_unit_match or len(phrase_hits) >= 2)
        and score >= 62
    ):
        decision = "same_job"
        confidence = "high" if len(phrase_hits) >= 2 else "medium"
    elif (
        candidate.url_kind == "employer_or_ats"
        and "direct_url" in candidate.strategies
        and company_match
        and title_overlap >= 0.70
        and (location_match or business_unit_match)
        and score >= 52
    ):
        decision = "same_job"
        confidence = "medium"
    elif score >= 38 and company_match and title_overlap >= 0.45:
        decision = "possible"
        confidence = "medium"
    else:
        decision = "reject"
        confidence = "medium" if page.status_code == 200 else "low"

    return CandidateEvaluation(
        requested_url=candidate.url,
        final_url=page.final_url,
        host=urlparse(page.final_url).netloc.lower(),
        url_kind=candidate.url_kind,
        page_title=page.title,
        status_code=page.status_code,
        evidence_score=round(score, 2),
        decision=decision,
        confidence=confidence,
        identifier_hits=identifier_hits,
        company_match=company_match,
        title_overlap=round(title_overlap, 3),
        location_match=location_match,
        business_unit_match=business_unit_match,
        distinctive_phrase_hits=phrase_hits,
        hard_conflicts=conflicts,
        reasons=reasons,
    )


def _fetch_candidates(
    candidates: list[SearchCandidate],
) -> tuple[dict[str, FetchedPage], list[CandidateEvaluation], list[str], int]:
    fetchable = [
        candidate
        for candidate in candidates
        if candidate.url_kind != "application_form"
    ][:MAX_FETCH_CANDIDATES]

    pages: dict[str, FetchedPage] = {}
    failures: list[CandidateEvaluation] = []
    errors: list[str] = []

    if not fetchable:
        return pages, failures, errors, 0

    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_FETCHES, len(fetchable))) as executor:
        futures = {
            executor.submit(fetch_public_page, candidate.url): candidate
            for candidate in fetchable
        }
        for future in as_completed(futures):
            candidate = futures[future]
            try:
                pages[candidate.url] = future.result()
            except Exception as exc:
                error = f"fetch failed for {candidate.url}: {type(exc).__name__}: {exc}"
                errors.append(error)
                failures.append(
                    CandidateEvaluation(
                        requested_url=candidate.url,
                        host=candidate.host,
                        url_kind=candidate.url_kind,
                        decision="unreadable",
                        fetch_error=error,
                    )
                )

    return pages, failures, errors, len(fetchable)


def _source_attachment_match(identity: JobIdentity, email: EmailMessage) -> bool:
    sender = (email.sender_email or "").strip().lower()
    if sender not in Settings().trusted_senders:
        return False

    text = email.attachment_text or ""
    if not text:
        return False

    company_ok = _company_match(identity, text)
    title_ratio = _overlap_ratio(_without_parenthetical(identity.title), text)
    unit = _without_parenthetical(identity.business_unit)
    unit_ok = bool(unit and _normalize(unit) in _normalize(text))
    phrase_hits = sum(
        1
        for phrase in identity.distinctive_phrases
        if _normalize(_without_parenthetical(phrase)) in _normalize(text)
    )
    return company_ok and title_ratio >= 0.65 and (unit_ok or phrase_hits >= 1)


def _build_judge():
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is missing from .env")

    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
    ).with_structured_output(SameJobJudgeOutput)


def _judge_ambiguous(
    identity: JobIdentity,
    evaluations: list[CandidateEvaluation],
    pages: dict[str, FetchedPage],
) -> SameJobJudgeOutput:
    possible = [item for item in evaluations if item.decision == "possible"]
    possible.sort(key=lambda item: item.evidence_score, reverse=True)
    possible = possible[:MAX_JUDGE_CANDIDATES]

    blocks: list[str] = []
    for index, evaluation in enumerate(possible, start=1):
        page = pages.get(evaluation.requested_url)
        if not page:
            continue
        blocks.append(
            f"CANDIDATE {index}\n"
            f"URL: {page.final_url}\n"
            f"PAGE TITLE: {page.title}\n"
            f"DETERMINISTIC EVIDENCE: {evaluation.model_dump(mode='json')}\n"
            f"PAGE TEXT:\n{page.text[:MAX_JUDGE_PAGE_CHARS]}"
        )

    prompt = f"""
You are the final tie-breaker for SAME-JOB identity resolution.
Do not decide whether a role is attractive or relevant. Decide only whether one
candidate page represents the exact employment opportunity in JOB IDENTITY.

Be conservative:
- same title alone is not enough;
- a different explicit Job/Requisition ID means different;
- prefer exact ID, team/business unit, location, dates/duration, and distinctive
  JD wording;
- if evidence cannot distinguish multiple similar roles, return ambiguous;
- candidate_url must be one of the supplied URLs when decision=same_job.

JOB IDENTITY:
{identity.model_dump(mode='json')}

CANDIDATES:
{chr(10).join(blocks)}
""".strip()
    return _build_judge().invoke(prompt)


def _evaluation_for_url(
    evaluations: list[CandidateEvaluation],
    url: str | None,
) -> CandidateEvaluation | None:
    if not url:
        return None
    for item in evaluations:
        if url in {item.requested_url, item.final_url}:
            return item
    return None


def verify_same_job(
    identity: JobIdentity,
    discovery: CandidateDiscoveryResult,
    email: EmailMessage,
    enable_llm_judge: bool = True,
) -> SameJobVerificationResult:
    """V3: prove whether a candidate page is the same job as the email/JD.

    Search ranking is never treated as verification. Public verification requires
    fetched-page evidence. A trusted NUS attachment may support source_verified
    when no public page proves same-job identity.
    """
    started = time.perf_counter()
    application_url = next(
        (
            candidate.url
            for candidate in discovery.candidates
            if candidate.url_kind == "application_form"
        ),
        None,
    )

    pages, failures, fetch_errors, fetch_calls = _fetch_candidates(discovery.candidates)
    evaluations: list[CandidateEvaluation] = list(failures)

    for candidate in discovery.candidates:
        page = pages.get(candidate.url)
        if page:
            evaluations.append(_evaluate_page(identity, candidate, page))

    evaluations.sort(key=lambda item: item.evidence_score, reverse=True)
    same = [item for item in evaluations if item.decision == "same_job"]
    possible = [item for item in evaluations if item.decision == "possible"]
    errors = [*discovery.errors, *fetch_errors]
    llm_calls = 0

    status = "unresolved"
    basis = "none"
    confidence = "low"
    matched: CandidateEvaluation | None = None

    if len(same) == 1:
        matched = same[0]
        status = "verified"
        confidence = matched.confidence
        if matched.identifier_hits:
            basis = "exact_identifier"
        elif "direct_url" in next(
            (
                candidate.strategies
                for candidate in discovery.candidates
                if candidate.url == matched.requested_url
            ),
            [],
        ) and matched.url_kind == "employer_or_ats":
            basis = "direct_official_url"
        else:
            basis = "jd_content_match"
    elif len(same) > 1:
        status = "ambiguous"
        basis = "multiple_candidates"
        confidence = "medium"
    elif possible and enable_llm_judge:
        try:
            judge = _judge_ambiguous(identity, evaluations, pages)
            llm_calls = 1
            candidate = _evaluation_for_url(evaluations, judge.candidate_url)

            # The LLM can only elevate a candidate that already has meaningful
            # deterministic support. It cannot invent a match from a weak page.
            if (
                judge.decision == "same_job"
                and candidate is not None
                and candidate.company_match
                and candidate.title_overlap >= 0.55
                and (
                    candidate.identifier_hits
                    or candidate.business_unit_match
                    or candidate.distinctive_phrase_hits
                )
            ):
                matched = candidate
                status = "verified"
                basis = "exact_identifier" if candidate.identifier_hits else "jd_content_match"
                confidence = "medium"
                candidate.reasons.append(f"LLM tie-breaker: {judge.reason}")
            elif judge.decision == "ambiguous":
                status = "ambiguous"
                basis = "multiple_candidates"
                confidence = "medium"
        except Exception as exc:
            llm_calls = 1
            errors.append(f"same-job judge failed: {type(exc).__name__}: {exc}")

    if status == "unresolved" and _source_attachment_match(identity, email):
        status = "source_verified"
        basis = "trusted_nus_attachment"
        confidence = "high"

    matched_evidence: list[str] = []
    conflicts: list[str] = []
    if matched:
        matched_evidence.extend(matched.reasons)
        matched_evidence.extend(
            f"distinctive phrase: {phrase}"
            for phrase in matched.distinctive_phrase_hits
        )
        matched_evidence.extend(
            f"identifier: {value}" for value in matched.identifier_hits
        )
        conflicts.extend(matched.hard_conflicts)
    elif status == "source_verified":
        matched_evidence.append(
            f"trusted attachment from {email.sender_email or email.sender_name} matched job identity"
        )

    return SameJobVerificationResult(
        identity_status=status,
        identity_basis=basis,
        confidence=confidence,
        official_url=(matched.final_url or matched.requested_url) if matched else None,
        application_url=application_url,
        matched_candidate_url=(matched.final_url or matched.requested_url) if matched else None,
        evaluations=evaluations,
        matched_evidence=matched_evidence,
        conflicts=conflicts,
        metrics=VerificationMetrics(
            fetch_calls=fetch_calls,
            pages_fetched=len(pages),
            llm_calls=llm_calls,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        ),
        errors=errors,
    )
