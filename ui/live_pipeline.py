from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from career_agent.catalog_consolidation import consolidate_job_records
from career_agent.connectors.outlook_graph import CAREER_SOURCE_BY_SENDER, OutlookGraphConnector
from career_agent.job_catalog_pipeline import research_career_email_for_catalog
from career_agent.matching_dataset import (
    is_matching_candidate,
    matching_evidence_level,
    matching_input_text,
    sanitize_job_sources,
)
from career_agent.models.inbox import CareerEmailRecord

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class LiveInboxBuild:
    email_count: int
    email_source_counts: dict[str, int]
    raw_job_count: int
    canonical_job_count: int
    candidate_count: int
    candidate_source_counts: dict[str, int]
    source_index: dict[tuple[str, str], dict]


def _key(company: str | None, title: str | None) -> tuple[str, str]:
    return (
        " ".join(str(company or "").lower().split()),
        " ".join(str(title or "").lower().split()),
    )


def _source_key(sender_email: str | None) -> str | None:
    return CAREER_SOURCE_BY_SENDER.get((sender_email or "").strip().lower())


def build_live_matching_candidates(
    output_path: Path,
    *,
    progress: ProgressCallback | None = None,
) -> LiveInboxBuild:
    """Build the matching input from the current live Outlook inbox.

    This is deliberately an orchestration-only UI adapter. It does not replace
    any extraction, research, consolidation, or matching logic. Every trusted
    email is passed through the frozen backend functions used by Track B.

    The JSON written to ``output_path`` is ephemeral and exists only because the
    frozen career-opportunity runner accepts a ``--jobs`` path.
    """
    scan_limit = max(1, int(os.getenv("SIMPLYNEXT_UI_EMAIL_SCAN", "999")))
    connector = OutlookGraphConnector()

    if progress:
        progress("Connecting to the dedicated Outlook career inbox...")

    # get_messages() already performs forwarded-sender recovery, trusted-sender
    # filtering, and PDF attachment retrieval using the frozen connector logic.
    messages = connector.get_messages(top=scan_limit, include_attachments=True)
    trusted_messages = [message for message in messages if _source_key(message.sender_email)]

    if not trusted_messages:
        raise RuntimeError("No Goh Ze Li / TalentConnect career emails were recovered from Outlook.")

    email_source_counts = Counter(_source_key(message.sender_email) or "unknown" for message in trusted_messages)
    if progress:
        progress(
            "Recovered "
            f"{len(trusted_messages)} career email(s) from Outlook "
            f"({email_source_counts.get('goh_ze_li', 0)} Goh, "
            f"{email_source_counts.get('talentconnect', 0)} TalentConnect)."
        )

    raw_jobs = []
    for email_index, email in enumerate(trusted_messages, start=1):
        source = _source_key(email.sender_email)
        if source is None:
            continue
        if progress:
            progress(
                f"[EMAIL {email_index:02}/{len(trusted_messages):02}] "
                f"{source} — {email.subject}"
            )

        result = research_career_email_for_catalog(
            CareerEmailRecord(source=source, email=email),
            fetch_linked_pdfs=True,
            progress=progress,
        )
        raw_jobs.extend(sanitize_job_sources(job) for job in result.job_records)

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
        "purpose": "Ephemeral live-Outlook input for the frozen career opportunity runner.",
        "job_count": len(candidates),
        "jobs": candidates,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    candidate_source_counts = Counter(str(item.get("source_key") or "unknown") for item in candidates)
    source_index = {
        _key(item.get("company"), item.get("title")): item
        for item in candidates
    }
    if progress:
        progress(
            f"Live catalogue ready: {len(raw_jobs)} raw jobs → "
            f"{len(canonical_jobs)} canonical jobs → {len(candidates)} active candidates."
        )

    return LiveInboxBuild(
        email_count=len(trusted_messages),
        email_source_counts=dict(email_source_counts),
        raw_job_count=len(raw_jobs),
        canonical_job_count=len(canonical_jobs),
        candidate_count=len(candidates),
        candidate_source_counts=dict(candidate_source_counts),
        source_index=source_index,
    )
