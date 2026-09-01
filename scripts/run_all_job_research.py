from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from career_agent.connectors.outlook_graph import CAREER_SOURCE_BY_SENDER, OutlookGraphConnector
from career_agent.job_catalog_pipeline import research_career_email_for_catalog
from career_agent.matching_dataset import is_matching_ready, sanitize_job_sources
from career_agent.models.inbox import CareerEmailRecord

DEFAULT_CATALOG_OUTPUT = Path("data/job_records/latest_job_catalog.json")
DEFAULT_MATCHING_OUTPUT = Path("data/job_records/latest_job_records.json")
DEFAULT_ARCHIVE_OUTPUT = Path("data/job_records/latest_job_records_archive.json")


def _source_key(sender_email: str | None) -> str | None:
    return CAREER_SOURCE_BY_SENDER.get((sender_email or "").strip().lower())


def _short(value: str | None, limit: int = 120) -> str:
    text = " ".join((value or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _latest_per_source(messages):
    selected = {}
    for message in messages:
        source = _source_key(message.sender_email)
        if source and source not in selected:
            selected[source] = message
    return [selected[key] for key in ("goh_ze_li", "talentconnect") if key in selected]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _catalog_job(job):
    clean = sanitize_job_sources(job)
    payload = clean.model_dump(mode="json")
    payload["matching_ready"] = is_matching_ready(clean)
    payload["matching_evidence_level"] = (
        "full_jd"
        if payload["matching_ready"]
        else "source_only"
        if clean.availability_status not in {"expired_by_source_deadline", "closed_by_official"}
        else "inactive"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "SimplyNext Track B: extract the latest trusted career emails, research jobs "
            "official-first by company, then write the canonical catalog for later matching."
        )
    )
    parser.add_argument("--scan", type=int, default=30)
    parser.add_argument(
        "--source",
        choices=["goh_ze_li", "talentconnect"],
        default=None,
        help="Optional single-source diagnostic run. Omit for latest Goh + latest TalentConnect.",
    )
    parser.add_argument("--subject", default=None, help="Optional case-insensitive subject filter.")
    parser.add_argument(
        "--no-linked-pdf",
        action="store_true",
        help="Skip linked PDF retrieval only for debugging.",
    )
    parser.add_argument("--catalog-output", default=str(DEFAULT_CATALOG_OUTPUT))
    parser.add_argument("--output", default=str(DEFAULT_MATCHING_OUTPUT))
    parser.add_argument("--archive-output", default=str(DEFAULT_ARCHIVE_OUTPUT))
    args = parser.parse_args()

    connector = OutlookGraphConnector()
    messages = connector.get_messages(top=args.scan, include_attachments=True)
    messages = [message for message in messages if _source_key(message.sender_email)]

    if args.source:
        messages = [message for message in messages if _source_key(message.sender_email) == args.source][:1]
    else:
        messages = _latest_per_source(messages)

    if args.subject:
        messages = [
            message
            for message in messages
            if args.subject.lower() in (message.subject or "").lower()
        ]
    if not messages:
        raise RuntimeError("No trusted Goh Ze Li / TalentConnect email matched the request.")

    print("=" * 112)
    print("SIMPLYNEXT TRACK B — CAREER EMAILS → OFFICIAL-FIRST JOB CATALOG")
    print("=" * 112)
    print("Trusted career emails selected:", len(messages))
    print("Research strategy: group by company → official/ATS first → LinkedIn → other secondary")
    print("Judge LLM in web research: disabled")

    all_results = []
    all_jobs = []

    for email_index, email in enumerate(messages, start=1):
        source = _source_key(email.sender_email)
        if source is None:
            continue

        print("\n" + "-" * 112)
        print(f"EMAIL {email_index}/{len(messages)} | {source}")
        print("Source :", f"{email.sender_name} <{email.sender_email}>")
        print("Subject:", _short(email.subject, 170))
        print("Extracting source and researching company sessions...")

        result = research_career_email_for_catalog(
            CareerEmailRecord(source=source, email=email),
            fetch_linked_pdfs=not args.no_linked_pdf,
            progress=print,
        )
        all_results.append(result)
        all_jobs.extend(sanitize_job_sources(job) for job in result.job_records)

        print("\nEMAIL SUMMARY")
        print("  opportunities :", len(result.opportunities))
        print("  companies     :", result.company_count)
        print("  JobRecords    :", len(result.job_records))
        print("  web searches  :", result.web_search_calls)
        print("  page fetches  :", result.page_fetch_calls)
        print("  judge LLM     :", result.judge_llm_calls)
        print("  extraction LLM:", result.extraction_llm_calls)
        if result.warnings:
            print("  warnings      :", len(result.warnings))
            for warning in result.warnings[:6]:
                print("    WARN:", _short(warning, 190))
        if result.errors:
            print("  errors        :", len(result.errors))
            for error in result.errors[:6]:
                print("    ERROR:", _short(error, 190))

    matching_jobs = [job for job in all_jobs if is_matching_ready(job)]
    catalog_jobs = [_catalog_job(job) for job in all_jobs]

    source_counts = Counter(job.source_key for job in all_jobs)
    availability_counts = Counter(job.availability_status for job in all_jobs)
    jd_counts = Counter(job.jd_status for job in all_jobs)

    catalog_payload = {
        "schema": "simplinext.job_catalog.v1",
        "purpose": (
            "Canonical Goh Ze Li + TalentConnect job catalog consumed later by the "
            "resume/transcript matching and ranking agent."
        ),
        "job_count": len(catalog_jobs),
        "matching_ready_count": len(matching_jobs),
        "source_counts": dict(source_counts),
        "availability_counts": dict(availability_counts),
        "jd_counts": dict(jd_counts),
        "jobs": catalog_jobs,
    }

    matching_payload = {
        "schema": "simplinext.job_records.matching.v1",
        "job_count": len(matching_jobs),
        "jobs": [job.model_dump(mode="json") for job in matching_jobs],
    }

    archive_payload = {
        "schema": "simplinext.job_records.archive.v1",
        "job_count": len(all_jobs),
        "jobs": [job.model_dump(mode="json") for job in all_jobs],
        "source_results": [
            {
                "source_key": result.source_key,
                "source_message_id": result.source_message_id,
                "source_subject": result.source_subject,
                "opportunities": len(result.opportunities),
                "job_records": len(result.job_records),
                "web_search_calls": result.web_search_calls,
                "page_fetch_calls": result.page_fetch_calls,
                "warnings": result.warnings,
                "errors": result.errors,
            }
            for result in all_results
        ],
    }

    catalog_output = Path(args.catalog_output)
    matching_output = Path(args.output)
    archive_output = Path(args.archive_output)
    _write_json(catalog_output, catalog_payload)
    _write_json(matching_output, matching_payload)
    _write_json(archive_output, archive_payload)

    print("\n" + "=" * 112)
    print("TRACK B JOB CATALOG SUMMARY")
    print("=" * 112)
    print("Total JobRecords :", len(all_jobs))
    print("By source        :", dict(source_counts))
    print("Availability     :", dict(availability_counts))
    print("JD status        :", dict(jd_counts))
    print("Matching-ready   :", len(matching_jobs))
    print("Canonical catalog:", catalog_output)
    print("Full-JD subset   :", matching_output)
    print("Archive/provenance:", archive_output)
    print("\nTrack B output is ready for a later Resume/Transcript ranking agent; matching was NOT run here.")


if __name__ == "__main__":
    main()
