from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

from career_agent.all_job_extraction import ExtractionMetrics, extract_all_opportunities
from career_agent.batch_sources import build_source_corpus
from career_agent.catalog_consolidation import consolidate_job_records
from career_agent.connectors.outlook_graph import CAREER_SOURCE_BY_SENDER, OutlookGraphConnector
from career_agent.goh_extraction import extract_goh_opportunities
from career_agent.job_catalog_pipeline import GENERIC_TALENTCONNECT_TITLES
from career_agent.job_research_quality import is_plausible_official_url, is_secondary_url
from career_agent.matching_dataset import (
    is_matching_candidate,
    matching_evidence_level,
    matching_input_text,
    sanitize_job_sources,
)
from career_agent.models.job_record import JobRecord
from ui.talentconnect_cached import ExtractionState, extract_talentconnect_cached

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class LiveInboxBuild:
    scanned_email_count: int
    email_count: int
    duplicate_email_count: int
    email_source_counts: dict[str, int]
    raw_job_count: int
    canonical_job_count: int
    candidate_count: int
    candidate_source_counts: dict[str, int]
    extraction_llm_calls: int
    source_index: dict[tuple[str, str], dict]
    extraction_warnings: list[str] = field(default_factory=list)


def _key(company: str | None, title: str | None) -> tuple[str, str]:
    return (
        " ".join(str(company or "").lower().split()),
        " ".join(str(title or "").lower().split()),
    )


def _source_key(sender_email: str | None) -> str | None:
    return CAREER_SOURCE_BY_SENDER.get((sender_email or "").strip().lower())


def _normalized_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip().lower()


def _email_fingerprint(source: str, email) -> str:
    """Deduplicate exact repeated forwards without collapsing distinct newsletters."""
    payload = "\n".join(
        [
            source,
            _normalized_text(email.subject),
            _normalized_text(email.body_text or email.body_html),
            _normalized_text(email.attachment_text),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()


def _dedupe_messages(messages) -> list:
    unique = []
    seen: set[str] = set()
    for email in messages:
        source = _source_key(email.sender_email)
        if source is None:
            continue
        fingerprint = _email_fingerprint(source, email)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(email)
    return unique


def _fresh_goh_base_extractor(**kwargs):
    """Call the frozen generic extractor directly; do not reuse saved snapshots."""
    return extract_all_opportunities(**kwargs)


def _no_llm_goh_base_extractor(**kwargs):
    """No-op base so the frozen Goh extractor returns deterministic table rows only."""
    corpus = str(kwargs.get("corpus") or "")
    return [], ExtractionMetrics(llm_calls=0, source_chars=len(corpus)), []


def _extract_email(source: str, email, *, extraction_state=None):
    corpus, _source_links, _documents, warnings = build_source_corpus(
        email,
        fetch_linked_pdfs=True,
    )
    kwargs = {
        "source_name": email.sender_name or email.sender_email or source,
        "source_message_id": email.message_id,
        "source_date": email.received_at,
        "corpus": corpus,
    }
    if source == "goh_ze_li":
        # Structured Goh circulars are deterministic and should not consume LLM quota.
        signals, metrics, errors = extract_goh_opportunities(
            **kwargs,
            base_extractor=_no_llm_goh_base_extractor,
        )
        if not signals:
            fallback_signals, fallback_metrics, fallback_errors = extract_goh_opportunities(
                **kwargs,
                base_extractor=_fresh_goh_base_extractor,
            )
            if len(fallback_signals) > len(signals):
                signals = fallback_signals
                metrics = fallback_metrics
            errors = [*errors, *fallback_errors]
    else:
        # Same TalentConnect extraction semantics, but successful chunks are memoized
        # by exact content hash and rate-limited chunks are paced/retried.
        signals, metrics, errors = extract_talentconnect_cached(**kwargs, state=extraction_state)
    return signals, metrics, [*warnings, *errors]


def _signal_to_source_job(source: str, email, signal) -> JobRecord:
    expired = bool(signal.deadline_hint and signal.deadline_hint < date.today())
    source_urls = list(dict.fromkeys(signal.urls or []))

    primary = next(
        (url for url in source_urls if is_plausible_official_url(url, signal.company)),
        None,
    )
    secondary = next((url for url in source_urls if is_secondary_url(url)), None)

    return JobRecord(
        source_key=source,
        source_message_id=email.message_id,
        source_sender_email=email.sender_email,
        source_subject=email.subject,
        company=signal.company,
        title=signal.role_title,
        industry=signal.industry,
        talentconnect_id=signal.talentconnect_id,
        remarks=signal.remarks,
        source_provenance=[{"source_key": source, "message_id": email.message_id, "subject": email.subject}],
        location=signal.location,
        opportunity_type=signal.opportunity_type,
        deadline_hint=signal.deadline_hint,
        availability_status="expired_by_source_deadline" if expired else "unknown",
        research_skipped_reason=(
            "source deadline has passed; web research skipped"
            if expired
            else "broad inbox extraction only; targeted web enrichment deferred until after rough ranking"
        ),
        target_major=list(signal.target_major or []),
        target_degree_level=list(signal.target_degree_level or []),
        source_urls=source_urls,
        record_kind="job_posting",
        research_status="source_verified",
        research_confidence="medium",
        research_basis="trusted_nus_email_source_only",
        primary_source_url=primary,
        secondary_source_url=secondary,
        jd_status="unavailable" if expired else "source_context_only",
        source_evidence=signal.raw_text or f"{signal.company or ''} | {signal.role_title or ''}",
        evidence_summary=["trusted NUS career source circulated this opportunity"],
    )


def build_live_matching_candidates(
    output_path: Path,
    *,
    progress: ProgressCallback | None = None,
) -> LiveInboxBuild:
    """Scan all trusted Outlook mail, extract broadly, then let the frozen ranker enrich selectively."""
    scan_limit = max(1, int(os.getenv("SIMPLYNEXT_UI_EMAIL_SCAN", "999")))
    connector = OutlookGraphConnector()

    if progress:
        progress("Connecting to the dedicated Outlook career inbox...")

    messages = connector.get_messages(top=scan_limit, include_attachments=True)
    trusted_messages = [message for message in messages if _source_key(message.sender_email)]
    if not trusted_messages:
        raise RuntimeError("No Goh Ze Li / TalentConnect career emails were recovered from Outlook.")

    unique_messages = _dedupe_messages(trusted_messages)
    duplicate_count = len(trusted_messages) - len(unique_messages)
    email_source_counts = Counter(_source_key(message.sender_email) or "unknown" for message in unique_messages)

    if progress:
        progress(
            f"Recovered {len(trusted_messages)} trusted career email(s); "
            f"processing {len(unique_messages)} unique email(s) after removing {duplicate_count} exact duplicate(s)."
        )
        progress(
            f"Unique sources: {email_source_counts.get('goh_ze_li', 0)} Goh · "
            f"{email_source_counts.get('talentconnect', 0)} TalentConnect."
        )
        progress("Broad pass = extraction only. No company-by-company web research before ranking.")

    raw_jobs: list[JobRecord] = []
    extraction_llm_calls = 0
    extraction_warnings = []
    extraction_state = ExtractionState()

    for email_index, email in enumerate(unique_messages, start=1):
        source = _source_key(email.sender_email)
        if source is None:
            continue
        if progress:
            progress(
                f"[EMAIL {email_index:02}/{len(unique_messages):02}] {source} — extracting opportunities"
            )

        signals, metrics, messages = _extract_email(source, email, extraction_state=extraction_state)
        extraction_llm_calls += int(metrics.llm_calls)
        extraction_warnings.extend(str(message) for message in messages if not str(message).startswith("INFO "))

        kept = 0
        for signal in signals:
            if not (signal.company or "").strip() or not (signal.role_title or "").strip():
                continue
            if source == "talentconnect" and (signal.role_title or "").strip().lower() in GENERIC_TALENTCONNECT_TITLES:
                continue
            raw_jobs.append(_signal_to_source_job(source, email, signal))
            kept += 1

        if progress:
            progress(
                f"    -> {kept} concrete opportunity signal(s); actual LLM attempts={metrics.llm_calls}"
            )
            for message in messages:
                normalized = " ".join(str(message).split())[:220]
                if normalized.startswith("INFO "):
                    progress(f"       {normalized}")
                else:
                    progress(f"       warning: {normalized}")

    canonical_jobs = consolidate_job_records(raw_jobs)
    candidates: list[dict] = []
    for raw_job in canonical_jobs:
        job = sanitize_job_sources(raw_job)
        if not is_matching_candidate(job):
            continue
        item = job.model_dump(mode="json")
        item["matching_evidence_level"] = matching_evidence_level(job)
        item["matching_input_text"] = matching_input_text(job)
        candidates.append(item)

    if not candidates:
        raise RuntimeError("Outlook career emails were found, but no active matching candidates were produced.")

    payload = {
        "schema": "simplinext.matching_candidates.v1",
        "purpose": "Ephemeral live-Outlook extraction input for the frozen career opportunity runner.",
        "extraction_complete": not extraction_warnings,
        "extraction_warnings": extraction_warnings,
        "job_count": len(candidates),
        "jobs": candidates,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    candidate_source_counts = Counter(str(item.get("source_key") or "unknown") for item in candidates)
    source_index = {_key(item.get("company"), item.get("title")): item for item in candidates}

    if progress:
        progress(
            f"Broad inbox pass ready: {len(raw_jobs)} raw jobs → "
            f"{len(canonical_jobs)} canonical jobs → {len(candidates)} active candidates."
        )
        progress(f"Total actual extraction LLM attempts this run: {extraction_llm_calls}.")
        progress("Next: frozen runner rough-ranks all candidates, then web-enriches only its Top 15 shortlist.")

    return LiveInboxBuild(
        scanned_email_count=len(trusted_messages),
        email_count=len(unique_messages),
        duplicate_email_count=duplicate_count,
        email_source_counts=dict(email_source_counts),
        raw_job_count=len(raw_jobs),
        canonical_job_count=len(canonical_jobs),
        candidate_count=len(candidates),
        candidate_source_counts=dict(candidate_source_counts),
        extraction_llm_calls=extraction_llm_calls,
        source_index=source_index,
        extraction_warnings=extraction_warnings,
    )
