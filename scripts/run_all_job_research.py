from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from career_agent.batch_job_research import research_career_email_record
from career_agent.connectors.outlook_graph import CAREER_SOURCE_BY_SENDER, OutlookGraphConnector
from career_agent.models.inbox import CareerEmailRecord

DEFAULT_OUTPUT = Path("data/job_records/latest_job_records.json")


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SimplyNext: read latest trusted career emails/attachments, research original jobs, and write matching-ready JobRecords."
    )
    parser.add_argument("--scan", type=int, default=30)
    parser.add_argument(
        "--source",
        choices=["goh_ze_li", "talentconnect"],
        default=None,
        help="Optional single-source run. Omit to process latest Goh + latest TalentConnect email.",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Optional case-insensitive subject filter.",
    )
    parser.add_argument(
        "--no-linked-pdf",
        action="store_true",
        help="Skip linked PDF retrieval only for debugging.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="JSON output path for the matching agent.",
    )
    args = parser.parse_args()

    connector = OutlookGraphConnector()
    messages = connector.get_messages(top=args.scan, include_attachments=True)
    messages = [message for message in messages if _source_key(message.sender_email)]

    if args.source:
        messages = [message for message in messages if _source_key(message.sender_email) == args.source]
        messages = messages[:1]
    else:
        messages = _latest_per_source(messages)

    if args.subject:
        messages = [
            message for message in messages
            if args.subject.lower() in (message.subject or "").lower()
        ]

    if not messages:
        raise RuntimeError("No trusted Goh Ze Li / TalentConnect email matched the request.")

    print("=" * 112)
    print("SIMPLYNEXT — CAREER EMAILS → ORIGINAL SOURCES/JD → MATCHING DATASET")
    print("=" * 112)
    print("Trusted career emails selected:", len(messages))

    all_results = []
    all_job_records = []
    grand_searches = grand_fetches = grand_extract_calls = grand_judges = 0

    for email_index, email in enumerate(messages, start=1):
        source = _source_key(email.sender_email)
        if source is None:
            continue
        result = research_career_email_record(
            CareerEmailRecord(source=source, email=email),
            fetch_linked_pdfs=not args.no_linked_pdf,
        )
        all_results.append(result)
        all_job_records.extend(result.job_records)

        print("\n" + "-" * 112)
        print(f"EMAIL {email_index}/{len(messages)}")
        print("Source :", f"{email.sender_name} <{email.sender_email}>")
        print("Subject:", _short(email.subject, 170))
        print("Source documents:")
        for doc in result.source_documents:
            suffix = f" | {doc.url}" if doc.url else ""
            print(f"  - {doc.source_type:11} {doc.text_chars:7} chars | {doc.label}{suffix}")

        print("\nEXTRACTION")
        print("  opportunities :", len(result.opportunities))
        print("  companies     :", result.company_count)
        print("  JobRecords    :", len(result.job_records))

        grouped: dict[str, list] = defaultdict(list)
        for job in result.job_records:
            grouped[job.company or "<unknown>"].append(job)

        for company, jobs in grouped.items():
            print("\n  " + company)
            for job in jobs:
                mark = "✓" if job.jd_text.strip() and job.jd_source_url else "?"
                if job.availability_status == "expired_by_source_deadline":
                    mark = "×"
                print(
                    f"    {mark} {_short(job.title, 92)}"
                    f" | {job.opportunity_type}"
                    f" | research={job.research_status}"
                    f" | jd={job.jd_status} ({len(job.jd_text)} chars)"
                )
                if job.primary_source_url:
                    print("      primary  :", job.primary_source_url)
                if job.secondary_source_url:
                    print("      secondary:", job.secondary_source_url)
                if job.jd_source_url:
                    print("      JD source:", job.jd_source_url)
                if job.research_skipped_reason:
                    print("      skipped  :", job.research_skipped_reason)

        print("\nEMAIL METRICS")
        print("  web searches  :", result.web_search_calls)
        print("  page fetches  :", result.page_fetch_calls)
        print("  judge LLM     :", result.judge_llm_calls)
        print("  extraction LLM:", result.extraction_llm_calls)
        print("  warnings      :", len(result.warnings))
        print("  errors        :", len(result.errors))

        grand_searches += result.web_search_calls
        grand_fetches += result.page_fetch_calls
        grand_extract_calls += result.extraction_llm_calls
        grand_judges += result.judge_llm_calls

    payload = {
        "schema": "simplinext.job_records.v1",
        "job_count": len(all_job_records),
        "jobs": [job.model_dump(mode="json") for job in all_job_records],
        "source_results": [
            {
                "source_key": result.source_key,
                "source_message_id": result.source_message_id,
                "source_subject": result.source_subject,
                "opportunities": len(result.opportunities),
                "job_records": len(result.job_records),
                "warnings": result.warnings,
                "errors": result.errors,
            }
            for result in all_results
        ],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    usable_jd = sum(1 for job in all_job_records if job.jd_text.strip() and job.jd_source_url)
    primary = sum(1 for job in all_job_records if job.primary_source_url)
    secondary = sum(1 for job in all_job_records if job.secondary_source_url)

    print("\n" + "=" * 112)
    print("MATCHING DATASET SUMMARY")
    print("=" * 112)
    print("JobRecords       :", len(all_job_records))
    print("With JD+source   :", usable_jd)
    print("Primary links    :", primary)
    print("Secondary links  :", secondary)
    print("Web searches     :", grand_searches)
    print("Page fetches     :", grand_fetches)
    print("Judge LLM calls  :", grand_judges)
    print("Extraction LLM   :", grand_extract_calls)
    print("Dataset written  :", output)
    print("\nReady for Resume/Transcript Match Agent consumption.")


if __name__ == "__main__":
    main()
