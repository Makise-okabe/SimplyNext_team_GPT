from __future__ import annotations

import hashlib
from collections import defaultdict

from career_agent.all_job_extraction import extract_all_opportunities
from career_agent.batch_sources import build_source_corpus
from career_agent.job_identity.concrete_job_research import research_concrete_job_or_delegate
from career_agent.models.email import EmailMessage
from career_agent.models.inbox import CareerEmailRecord
from career_agent.models.job_identity import JobIdentity
from career_agent.models.job_record import EmailOpportunityResearchResult, JobRecord
from career_agent.models.signal import OpportunitySignal
from career_agent.nodes.normalize_email import html_to_text
from career_agent.tools.web_fetch import fetch_public_page

MAX_JD_TEXT_CHARS = 30_000


def _fingerprint(signal: OpportunitySignal) -> str:
    value = "|".join(
        [
            (signal.company or "").strip().lower(),
            (signal.role_title or "").strip().lower(),
            (signal.location or "").strip().lower(),
            signal.opportunity_type,
            "|".join(sorted(signal.urls)),
        ]
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _identity_from_signal(signal: OpportunitySignal, index: int) -> JobIdentity:
    evidence = [signal.raw_text] if signal.raw_text else []
    return JobIdentity(
        source_message_id=signal.source_message_id,
        signal_index=index,
        company=signal.company,
        title=signal.role_title,
        location=signal.location,
        opportunity_type=signal.opportunity_type,
        employment_type=(
            signal.opportunity_type
            if signal.opportunity_type in {"full_time", "internship"}
            else None
        ),
        direct_urls=signal.urls,
        evidence_snippets=evidence,
        identity_strength="moderate" if signal.company and signal.role_title else "weak",
        source_fingerprint=_fingerprint(signal),
    )


def _source_context(email: EmailMessage, signal: OpportunitySignal, radius: int = 1800) -> str:
    text = (email.body_text or "").strip()
    if email.body_html:
        html_text = html_to_text(email.body_html)
        if len(html_text) > len(text):
            text = html_text
    if email.attachment_text:
        text = f"{text}\n\n{email.attachment_text}".strip()

    anchor = signal.role_title or signal.company or ""
    if anchor:
        index = text.lower().find(anchor.lower())
        if index >= 0:
            start = max(0, index - radius)
            end = min(len(text), index + len(anchor) + radius)
            return text[start:end].strip()
    return (signal.raw_text or text[: radius * 2]).strip()


def _fetch_best_jd(package, source_context: str) -> tuple[str, str | None, str, list[str], int]:
    warnings: list[str] = []
    fetches = 0

    if package.official_job_url:
        try:
            page = fetch_public_page(package.official_job_url)
            fetches += 1
            if page.text.strip():
                return (
                    "fetched_official",
                    page.final_url or package.official_job_url,
                    page.text[:MAX_JD_TEXT_CHARS],
                    warnings,
                    fetches,
                )
            warnings.append("official job page returned no static text; retained source context")
        except Exception as exc:
            fetches += 1
            warnings.append(
                f"official JD fetch failed: {type(exc).__name__}: {exc}"
            )

    secondary_urls = package.secondary_evidence_urls or []
    for url in secondary_urls[:1]:
        try:
            page = fetch_public_page(url)
            fetches += 1
            if page.text.strip():
                return (
                    "fetched_secondary",
                    page.final_url or url,
                    page.text[:MAX_JD_TEXT_CHARS],
                    warnings,
                    fetches,
                )
        except Exception as exc:
            fetches += 1
            warnings.append(
                f"secondary JD fetch failed: {type(exc).__name__}: {exc}"
            )

    if source_context:
        return (
            "source_context_only",
            None,
            source_context[:MAX_JD_TEXT_CHARS],
            warnings,
            fetches,
        )
    return "unavailable", None, "", warnings, fetches


def research_career_email_record(
    record: CareerEmailRecord,
    *,
    fetch_linked_pdfs: bool = True,
) -> EmailOpportunityResearchResult:
    email = record.email
    corpus, source_links, documents, source_warnings = build_source_corpus(
        email,
        fetch_linked_pdfs=fetch_linked_pdfs,
    )

    opportunities, extraction_metrics, extraction_errors = extract_all_opportunities(
        source_name=email.sender_name or email.sender_email or record.source,
        source_message_id=email.message_id,
        source_date=email.received_at,
        corpus=corpus,
    )

    # Preserve all public source URLs in the research email while keeping each
    # opportunity's exact direct URLs separate.
    research_email = email.model_copy(
        update={
            "body_text": corpus,
            "links": list(dict.fromkeys([*email.links, *source_links])),
        }
    )

    groups: dict[str, list[tuple[int, OpportunitySignal]]] = defaultdict(list)
    for index, signal in enumerate(opportunities, start=1):
        key = " ".join((signal.company or "unknown").lower().split())
        groups[key].append((index, signal))

    job_records: list[JobRecord] = []
    search_calls = 0
    page_fetch_calls = 0
    judge_llm_calls = 0
    warnings = list(source_warnings)
    errors = list(extraction_errors)

    # Company grouping is explicit even before deeper cross-role company-context
    # reuse is added. This gives one stable grouping boundary for later caching.
    for _, company_items in groups.items():
        for index, signal in company_items:
            identity = _identity_from_signal(signal, index)
            package = research_concrete_job_or_delegate(identity, research_email)
            search_calls += package.metrics.search_calls
            page_fetch_calls += package.metrics.fetch_calls
            judge_llm_calls += package.metrics.judge_llm_calls

            context = _source_context(research_email, signal)
            jd_status, jd_url, jd_text, jd_warnings, extra_fetches = _fetch_best_jd(
                package,
                context,
            )
            page_fetch_calls += extra_fetches

            job_records.append(
                JobRecord(
                    source_message_id=email.message_id,
                    source_sender_email=email.sender_email,
                    source_subject=email.subject,
                    company=signal.company,
                    title=signal.role_title,
                    location=signal.location,
                    opportunity_type=signal.opportunity_type,
                    deadline_hint=signal.deadline_hint,
                    target_major=signal.target_major,
                    target_degree_level=signal.target_degree_level,
                    source_urls=signal.urls,
                    record_kind=package.record_kind,
                    research_status=package.status,
                    research_confidence=package.confidence,
                    research_basis=package.basis,
                    official_job_url=package.official_job_url,
                    application_url=package.application_url,
                    jd_status=jd_status,
                    jd_source_url=jd_url,
                    jd_text=jd_text,
                    source_evidence=signal.raw_text,
                    evidence_summary=package.evidence_summary,
                    warnings=list(dict.fromkeys([*package.warnings, *jd_warnings])),
                    errors=package.errors,
                )
            )
            warnings.extend(jd_warnings)
            errors.extend(package.errors)

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
        web_search_calls=search_calls,
        page_fetch_calls=page_fetch_calls,
        judge_llm_calls=judge_llm_calls,
        warnings=list(dict.fromkeys(warnings)),
        errors=list(dict.fromkeys(errors)),
    )
