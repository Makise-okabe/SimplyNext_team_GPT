from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from datetime import date
from urllib.parse import urlparse

from career_agent.all_job_extraction import extract_all_opportunities
from career_agent.batch_sources import build_source_corpus
from career_agent.job_identity.concrete_job_research import (
    _enhanced_job_like_url,
    research_concrete_job_or_delegate,
)
from career_agent.models.email import EmailMessage
from career_agent.models.inbox import CareerEmailRecord
from career_agent.models.job_identity import JobIdentity
from career_agent.models.job_record import EmailOpportunityResearchResult, JobRecord
from career_agent.models.signal import OpportunitySignal
from career_agent.nodes.normalize_email import html_to_text
from career_agent.talentconnect_extraction import extract_talentconnect_opportunities
from career_agent.tools.web_fetch import fetch_public_page
from career_agent.tools.web_search import search_public_web

MAX_JD_TEXT_CHARS = 30_000
MIN_USEFUL_JD_CHARS = 500
MIN_JD_TITLE_OVERLAP = 0.30
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


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


def _title_overlap(title: str | None, text: str) -> float:
    source = {
        token
        for token in TOKEN_PATTERN.findall((title or "").lower())
        if len(token) >= 3 or token.isdigit()
    }
    if not source:
        return 0.0
    target = set(TOKEN_PATTERN.findall(text.lower()))
    return len(source & target) / len(source)


def _host_matches(candidate_host: str, known_host: str) -> bool:
    candidate_host = candidate_host.lower().split(":", 1)[0]
    known_host = known_host.lower().split(":", 1)[0]
    return candidate_host == known_host or candidate_host.endswith(f".{known_host}")


def _seed_direct_url_from_company_hosts(
    signal: OpportunitySignal,
    official_hosts: list[str],
) -> tuple[OpportunitySignal, int, list[str]]:
    if signal.urls or not signal.role_title or not official_hosts:
        return signal, 0, []

    warnings: list[str] = []
    calls = 0
    location = signal.location or "Singapore"

    for host in official_hosts[:2]:
        query = f'site:{host} "{signal.role_title}" {location}'.strip()
        try:
            results = search_public_web(query, max_results=6)
            calls += 1
        except Exception as exc:
            calls += 1
            warnings.append(
                f"company-session search failed for {host}: {type(exc).__name__}: {exc}"
            )
            continue

        for result in results:
            try:
                candidate_host = urlparse(result.url).netloc.lower()
            except ValueError:
                continue
            if not _host_matches(candidate_host, host):
                continue
            if not _enhanced_job_like_url(result.url):
                continue
            metadata = f"{result.title} {result.snippet} {result.url}"
            if _title_overlap(signal.role_title, metadata) < 0.80:
                continue
            return (
                signal.model_copy(
                    update={"urls": list(dict.fromkeys([*signal.urls, result.url]))}
                ),
                calls,
                warnings,
            )

    return signal, calls, warnings


def _official_hosts_from_package(package) -> list[str]:
    hosts: list[str] = []
    if package.official_job_url:
        try:
            host = urlparse(package.official_job_url).netloc.lower()
            if host:
                hosts.append(host)
        except ValueError:
            pass

    for candidate in package.candidates:
        if candidate.tier == "official" and candidate.host:
            hosts.append(candidate.host.lower())
    return list(dict.fromkeys(hosts))


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


def _useful_jd_text(text: str, title: str | None) -> bool:
    cleaned = " ".join((text or "").split())
    if len(cleaned) < MIN_USEFUL_JD_CHARS:
        return False
    return _title_overlap(title, cleaned) >= MIN_JD_TITLE_OVERLAP


def _fetch_best_jd(package, source_context: str) -> tuple[str, str | None, str, list[str], int]:
    warnings: list[str] = []
    fetches = 0
    title = package.identity.title

    if package.official_job_url:
        try:
            page = fetch_public_page(package.official_job_url)
            fetches += 1
            if _useful_jd_text(page.text, title):
                return (
                    "fetched_official",
                    page.final_url or package.official_job_url,
                    page.text[:MAX_JD_TEXT_CHARS],
                    warnings,
                    fetches,
                )
            warnings.append(
                "official job page returned only a shell/insufficient JD text; retained fallback context"
            )
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
            if _useful_jd_text(page.text, title):
                return (
                    "fetched_secondary",
                    page.final_url or url,
                    page.text[:MAX_JD_TEXT_CHARS],
                    warnings,
                    fetches,
                )
            warnings.append(
                "secondary page returned insufficient role-specific JD text; retained source context"
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


def _is_expired(signal: OpportunitySignal, *, today: date | None = None) -> bool:
    if signal.deadline_hint is None:
        return False
    return signal.deadline_hint < (today or date.today())


def _expired_job_record(email: EmailMessage, signal: OpportunitySignal) -> JobRecord:
    return JobRecord(
        source_message_id=email.message_id,
        source_sender_email=email.sender_email,
        source_subject=email.subject,
        company=signal.company,
        title=signal.role_title,
        location=signal.location,
        opportunity_type=signal.opportunity_type,
        deadline_hint=signal.deadline_hint,
        availability_status="expired_by_source_deadline",
        research_skipped_reason="source deadline has passed; expensive current-web research skipped",
        target_major=signal.target_major,
        target_degree_level=signal.target_degree_level,
        source_urls=signal.urls,
        record_kind="job_posting",
        research_status="source_verified",
        research_confidence="medium",
        research_basis="trusted_nus_email_expired_source_deadline",
        jd_status="unavailable",
        source_evidence=signal.raw_text,
        evidence_summary=[
            "trusted NUS career source circulated this opportunity",
            f"source deadline passed: {signal.deadline_hint.isoformat()}",
        ],
    )


def research_career_email_record(
    record: CareerEmailRecord,
    *,
    fetch_linked_pdfs: bool = True,
) -> EmailOpportunityResearchResult:
    """Read email/attachments, extract all leads, then research official originals."""
    email = record.email
    corpus, source_links, documents, source_warnings = build_source_corpus(
        email,
        fetch_linked_pdfs=fetch_linked_pdfs,
    )

    extraction_fn = (
        extract_talentconnect_opportunities
        if record.source == "talentconnect"
        else extract_all_opportunities
    )
    opportunities, extraction_metrics, extraction_errors = extraction_fn(
        source_name=email.sender_name or email.sender_email or record.source,
        source_message_id=email.message_id,
        source_date=email.received_at,
        corpus=corpus,
    )

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

    for _, company_items in groups.items():
        company_hosts: list[str] = []

        for index, original_signal in company_items:
            if _is_expired(original_signal):
                job_records.append(_expired_job_record(email, original_signal))
                continue

            signal, seed_calls, seed_warnings = _seed_direct_url_from_company_hosts(
                original_signal,
                company_hosts,
            )
            search_calls += seed_calls
            warnings.extend(seed_warnings)

            identity = _identity_from_signal(signal, index)
            package = research_concrete_job_or_delegate(identity, research_email)
            search_calls += package.metrics.search_calls
            page_fetch_calls += package.metrics.fetch_calls
            judge_llm_calls += package.metrics.judge_llm_calls

            discovered_hosts = _official_hosts_from_package(package)
            company_hosts = list(dict.fromkeys([*company_hosts, *discovered_hosts]))

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
                    availability_status="active_candidate" if signal.deadline_hint else "unknown",
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

    all_company_keys = {
        " ".join((signal.company or "unknown").lower().split())
        for signal in opportunities
    }

    return EmailOpportunityResearchResult(
        source_key=record.source,
        source_message_id=email.message_id,
        source_subject=email.subject,
        source_documents=documents,
        opportunities=opportunities,
        company_count=len(all_company_keys),
        job_records=job_records,
        extraction_llm_calls=extraction_metrics.llm_calls,
        extraction_source_chars=extraction_metrics.source_chars,
        web_search_calls=search_calls,
        page_fetch_calls=page_fetch_calls,
        judge_llm_calls=judge_llm_calls,
        warnings=list(dict.fromkeys(warnings)),
        errors=list(dict.fromkeys(errors)),
    )
