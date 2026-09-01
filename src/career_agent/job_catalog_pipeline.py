from __future__ import annotations

import re
from collections import defaultdict
from datetime import date
from typing import Callable
from urllib.parse import unquote

from career_agent.all_job_extraction import extract_all_opportunities
from career_agent.batch_sources import build_source_corpus
from career_agent.company_job_research import ResearchContext, research_company_jobs
from career_agent.goh_extraction import extract_goh_opportunities
from career_agent.job_research_quality import is_plausible_official_url, is_secondary_url
from career_agent.models.email import EmailMessage
from career_agent.models.inbox import CareerEmailRecord
from career_agent.models.job_record import EmailOpportunityResearchResult, JobRecord
from career_agent.models.signal import OpportunitySignal
from career_agent.talentconnect_extraction import extract_talentconnect_opportunities

ProgressCallback = Callable[[str], None]

COMPANY_CANONICAL_ALIASES = {
    "pg": "procter gamble",
    "procter gamble": "procter gamble",
    "procterandgamble": "procter gamble",
    "watsons": "watsons",
    "deutschebank": "deutsche bank",
    "ey": "ernst young",
    "ernstyoung": "ernst young",
    "ernstyoungsingaporeey": "ernst young",
    "ernstyoungsolutions": "ernst young",
}
LEGAL_SUFFIXES = {
    "pte",
    "ltd",
    "limited",
    "private",
    "inc",
    "corp",
    "corporation",
    "plc",
    "llp",
    "ag",
}
GENERIC_TALENTCONNECT_TITLES = {
    "career opportunities",
    "hiring opportunities",
    "graduate opportunities",
    "internship opportunities",
    "employment opportunities",
}
DIRECT_JOB_URL_MARKERS = (
    "/job/",
    "/jobs/",
    "/position/",
    "jobdetail",
    "jobcode=",
    "jobid=",
    "requisition",
)


def _canonical_company_text(company: str | None) -> str:
    raw = (company or "unknown").lower().replace("&", " and ")
    raw = re.sub(r"[^a-z0-9]+", " ", raw)
    tokens = [token for token in raw.split() if token not in LEGAL_SUFFIXES]
    value = " ".join(tokens).strip()
    compact = "".join(token for token in tokens if token != "and")
    if compact in COMPANY_CANONICAL_ALIASES:
        return COMPANY_CANONICAL_ALIASES[compact]
    return value.replace(" and ", " ").strip()


def _company_key(signal: OpportunitySignal) -> str:
    return _canonical_company_text(signal.company)


def _is_generic_talentconnect_seed(signal: OpportunitySignal) -> bool:
    return (signal.role_title or "").strip().lower() in GENERIC_TALENTCONNECT_TITLES


def _url_mentions_company(url: str, company: str | None) -> bool:
    canonical = _canonical_company_text(company)
    if canonical == "unknown":
        return False
    lowered = unquote(url).lower()
    compact_url = re.sub(r"[^a-z0-9]", "", lowered)
    tokens = [token for token in canonical.split() if len(token) >= 3]
    if not tokens:
        return False
    return any(token in compact_url for token in tokens)


def _sanitize_signal_source_urls(signal: OpportunitySignal) -> OpportunitySignal:
    """Reject concrete job URLs that clearly belong to another employer.

    Generic/short links are retained for later resolution. Secondary mirrors are
    retained. Concrete employer/ATS links must either pass the official company
    gate or at least carry the current company identity in the URL itself.
    """
    kept: list[str] = []
    for url in signal.urls:
        lowered = unquote(url).lower()
        if is_secondary_url(url) or is_plausible_official_url(url, signal.company):
            kept.append(url)
            continue
        looks_concrete = any(marker in lowered for marker in DIRECT_JOB_URL_MARKERS)
        if looks_concrete and not _url_mentions_company(url, signal.company):
            continue
        kept.append(url)
    if kept == signal.urls:
        return signal
    return signal.model_copy(update={"urls": list(dict.fromkeys(kept))})


def _expired_job_record(
    *,
    source_key: str,
    email: EmailMessage,
    signal: OpportunitySignal,
) -> JobRecord:
    return JobRecord(
        source_key=source_key if source_key in {"goh_ze_li", "talentconnect"} else "unknown",
        source_message_id=email.message_id,
        source_sender_email=email.sender_email,
        source_subject=email.subject,
        company=signal.company,
        title=signal.role_title,
        location=signal.location,
        opportunity_type=signal.opportunity_type,
        deadline_hint=signal.deadline_hint,
        availability_status="expired_by_source_deadline",
        research_skipped_reason="source deadline has passed; web research skipped",
        target_major=signal.target_major,
        target_degree_level=signal.target_degree_level,
        source_urls=signal.urls,
        record_kind="job_posting",
        research_status="source_verified",
        research_confidence="medium",
        research_basis="trusted_nus_email_expired_source_deadline",
        jd_status="unavailable",
        source_evidence=signal.raw_text,
        evidence_summary=["trusted NUS career source circulated this opportunity"],
    )


def _extract(record: CareerEmailRecord, corpus: str):
    email = record.email
    kwargs = dict(
        source_name=email.sender_name or email.sender_email or record.source,
        source_message_id=email.message_id,
        source_date=email.received_at,
        corpus=corpus,
    )
    if record.source == "talentconnect":
        return extract_talentconnect_opportunities(**kwargs)
    return extract_goh_opportunities(
        **kwargs,
        base_extractor=extract_all_opportunities,
    )


def research_career_email_for_catalog(
    record: CareerEmailRecord,
    *,
    fetch_linked_pdfs: bool = True,
    progress: ProgressCallback | None = None,
) -> EmailOpportunityResearchResult:
    """Extract one trusted email and research its concrete jobs by company."""
    email = record.email
    corpus, source_links, documents, source_warnings = build_source_corpus(
        email,
        fetch_linked_pdfs=fetch_linked_pdfs,
    )
    opportunities, extraction_metrics, extraction_errors = _extract(record, corpus)
    opportunities = [_sanitize_signal_source_urls(signal) for signal in opportunities]
    _ = source_links

    groups: dict[str, list[tuple[int, OpportunitySignal]]] = defaultdict(list)
    expired_records: list[JobRecord] = []
    today = date.today()
    filtered_generic = 0

    for index, signal in enumerate(opportunities, start=1):
        if record.source == "talentconnect" and _is_generic_talentconnect_seed(signal):
            filtered_generic += 1
            continue
        if signal.deadline_hint and signal.deadline_hint < today:
            expired_records.append(
                _expired_job_record(source_key=record.source, email=email, signal=signal)
            )
            continue
        groups[_company_key(signal)].append((index, signal))

    context = ResearchContext()
    job_records: list[JobRecord] = list(expired_records)
    warnings = list(source_warnings)
    errors = list(extraction_errors)
    if filtered_generic:
        warnings.append(
            f"filtered {filtered_generic} generic TalentConnect company lead(s) from concrete JobRecord catalog"
        )

    company_items = list(groups.items())
    for company_index, (_, items) in enumerate(company_items, start=1):
        company = items[0][1].company or "<unknown>"
        if progress:
            progress(
                f"[COMPANY {company_index:02}/{len(company_items):02}] "
                f"{company} — {len(items)} active/unknown role(s)"
            )
        outcome = research_company_jobs(
            email=email,
            source_key=record.source,
            company_items=items,
            context=context,
            progress=progress,
        )
        job_records.extend(outcome.job_records)
        warnings.extend(outcome.warnings)
        errors.extend(outcome.errors)

        if progress:
            for job in outcome.job_records:
                source = (
                    "official"
                    if job.jd_status == "fetched_official"
                    else "secondary"
                    if job.jd_status == "fetched_secondary"
                    else "closed"
                    if job.availability_status == "closed_by_official"
                    else "unresolved"
                )
                progress(f"    -> {job.title}: {source}")

    order = {
        (signal.company or "", signal.role_title or "", signal.raw_text): index
        for index, signal in enumerate(opportunities)
    }
    job_records.sort(
        key=lambda job: order.get((job.company or "", job.title or "", job.source_evidence), 10**9)
    )

    return EmailOpportunityResearchResult(
        source_key=record.source,
        source_message_id=email.message_id,
        source_subject=email.subject,
        source_documents=documents,
        opportunities=opportunities,
        company_count=len(groups),
        job_records=job_records,
        extraction_llm_calls=extraction_metrics.llm_calls,
        extraction_source_chars=extraction_metrics.source_chars,
        web_search_calls=context.search_calls,
        page_fetch_calls=context.fetch_calls,
        judge_llm_calls=0,
        warnings=list(dict.fromkeys([*warnings, *context.warnings])),
        errors=list(dict.fromkeys([*errors, *context.errors])),
    )
