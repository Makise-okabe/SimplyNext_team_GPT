from __future__ import annotations

import argparse
from collections import defaultdict

from career_agent.all_job_extraction import extract_all_opportunities
from career_agent.batch_job_research import research_career_email_record
from career_agent.batch_sources import build_source_corpus
from career_agent.connectors.outlook_graph import CAREER_SOURCE_BY_SENDER, OutlookGraphConnector
from career_agent.models.inbox import CareerEmailRecord


def _source_key(sender_email: str | None) -> str | None:
    return CAREER_SOURCE_BY_SENDER.get((sender_email or "").strip().lower())


def _short(value: str | None, limit: int = 120) -> str:
    text = " ".join((value or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _extract_only(record: CareerEmailRecord, fetch_linked_pdfs: bool):
    email = record.email
    corpus, _, documents, warnings = build_source_corpus(
        email,
        fetch_linked_pdfs=fetch_linked_pdfs,
    )
    opportunities, metrics, errors = extract_all_opportunities(
        source_name=email.sender_name or email.sender_email or record.source,
        source_message_id=email.message_id,
        source_date=email.received_at,
        corpus=corpus,
    )
    return corpus, documents, opportunities, metrics, warnings, errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SimplyNext: extract every job from recent career emails and research official JDs."
    )
    parser.add_argument("--scan", type=int, default=30)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument(
        "--subject",
        default=None,
        help="Optional case-insensitive subject filter for repeatable testing.",
    )
    parser.add_argument(
        "--no-linked-pdf",
        action="store_true",
        help="Skip public PDF retrieval while debugging extraction.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Stop after exhaustive email/PDF opportunity extraction; no web job research.",
    )
    args = parser.parse_args()

    connector = OutlookGraphConnector()
    messages = connector.get_messages(top=args.scan, include_attachments=True)
    if args.subject:
        messages = [
            message
            for message in messages
            if args.subject.lower() in (message.subject or "").lower()
        ]
    messages = messages[: args.limit]

    print("=" * 112)
    print(
        "SIMPLYNEXT — ALL-JOB EXTRACTION"
        if args.extract_only
        else "SIMPLYNEXT — ALL-JOB EMAIL RESEARCH → JOB RECORDS"
    )
    print("=" * 112)
    print("Trusted career emails selected:", len(messages))

    if not messages:
        raise RuntimeError("No trusted Goh Ze Li / TalentConnect email matched the request.")

    if args.extract_only:
        total_opportunities = 0
        total_companies: set[str] = set()
        total_llm_calls = 0
        total_chars = 0

        for email_index, email in enumerate(messages, start=1):
            source = _source_key(email.sender_email)
            if source is None:
                continue
            record = CareerEmailRecord(source=source, email=email)
            corpus, documents, opportunities, metrics, warnings, errors = _extract_only(
                record,
                fetch_linked_pdfs=not args.no_linked_pdf,
            )

            print("\n" + "-" * 112)
            print(f"EMAIL {email_index}/{len(messages)}")
            print("Source :", f"{email.sender_name} <{email.sender_email}>")
            print("Subject:", _short(email.subject, 170))
            print("Source documents:")
            for doc in documents:
                suffix = f" | {doc.url}" if doc.url else ""
                print(f"  - {doc.source_type:11} {doc.text_chars:7} chars | {doc.label}{suffix}")

            companies: dict[str, list] = defaultdict(list)
            for opportunity in opportunities:
                companies[opportunity.company or "<unknown>"].append(opportunity)

            print("\nEXTRACTED OPPORTUNITIES")
            for company, roles in companies.items():
                print("\n  " + company)
                for role in roles:
                    print(
                        f"    - {_short(role.role_title, 96)}"
                        f" | {role.opportunity_type}"
                        f" | {role.location or 'location unknown'}"
                    )
                    if role.urls:
                        print("      direct URLs:", len(role.urls))
                        for url in role.urls[:3]:
                            print("        ", url)

            print("\nEMAIL EXTRACTION METRICS")
            print("  source chars  :", len(corpus))
            print("  opportunities :", len(opportunities))
            print("  companies     :", len(companies))
            print("  extraction LLM:", metrics.llm_calls)
            print("  warnings      :", len(warnings))
            print("  errors        :", len(errors))
            for warning in warnings[:8]:
                print("    WARN:", _short(warning, 180))
            for error in errors[:8]:
                print("    ERROR:", _short(error, 180))

            total_opportunities += len(opportunities)
            total_companies.update(companies)
            total_llm_calls += metrics.llm_calls
            total_chars += metrics.source_chars

        print("\n" + "=" * 112)
        print("EXTRACTION SUMMARY")
        print("=" * 112)
        print("Opportunities    :", total_opportunities)
        print("Companies        :", len(total_companies))
        print("Source chars     :", total_chars)
        print("Extraction LLM   :", total_llm_calls)
        print("\nStopped after extraction. No web research or matching was run.")
        return

    grand_jobs = 0
    grand_companies: set[str] = set()
    grand_verified = 0
    grand_jd = 0
    grand_searches = 0
    grand_fetches = 0
    grand_judges = 0
    grand_extract_calls = 0

    for email_index, email in enumerate(messages, start=1):
        source = _source_key(email.sender_email)
        if source is None:
            continue
        record = CareerEmailRecord(source=source, email=email)
        result = research_career_email_record(
            record,
            fetch_linked_pdfs=not args.no_linked_pdf,
        )

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
        print("  extraction LLM:", result.extraction_llm_calls)
        print("  source chars  :", result.extraction_source_chars)

        grouped: dict[str, list] = defaultdict(list)
        for job in result.job_records:
            grouped[job.company or "<unknown>"].append(job)

        for company, jobs in grouped.items():
            print("\n  " + company)
            for job in jobs:
                status_mark = "✓" if job.research_status == "verified_exact_job" else "?"
                print(
                    f"    {status_mark} {_short(job.title, 92)}"
                    f" | {job.opportunity_type}"
                    f" | research={job.research_status}"
                    f" | jd={job.jd_status} ({len(job.jd_text)} chars)"
                )
                if job.official_job_url:
                    print("      official:", job.official_job_url)
                elif job.application_url:
                    print("      apply   :", job.application_url)
                if job.jd_source_url:
                    print("      JD source:", job.jd_source_url)

        verified = sum(
            1 for job in result.job_records if job.research_status == "verified_exact_job"
        )
        jds = sum(1 for job in result.job_records if job.jd_text.strip())
        print("\nEMAIL METRICS")
        print("  JobRecords    :", len(result.job_records))
        print("  exact verified:", verified)
        print("  JD available  :", jds)
        print("  web searches  :", result.web_search_calls)
        print("  page fetches  :", result.page_fetch_calls)
        print("  judge LLM     :", result.judge_llm_calls)
        print("  extraction LLM:", result.extraction_llm_calls)
        print("  warnings      :", len(result.warnings))
        print("  errors        :", len(result.errors))
        for warning in result.warnings[:8]:
            print("    WARN:", _short(warning, 180))
        for error in result.errors[:8]:
            print("    ERROR:", _short(error, 180))

        grand_jobs += len(result.job_records)
        grand_companies.update((job.company or "<unknown>") for job in result.job_records)
        grand_verified += verified
        grand_jd += jds
        grand_searches += result.web_search_calls
        grand_fetches += result.page_fetch_calls
        grand_judges += result.judge_llm_calls
        grand_extract_calls += result.extraction_llm_calls

    print("\n" + "=" * 112)
    print("BATCH SUMMARY")
    print("=" * 112)
    print("JobRecords       :", grand_jobs)
    print("Companies        :", len(grand_companies))
    print("Exact verified   :", grand_verified)
    print("JD available     :", grand_jd)
    print("Web searches     :", grand_searches)
    print("Page fetches     :", grand_fetches)
    print("Judge LLM calls  :", grand_judges)
    print("Extraction LLM   :", grand_extract_calls)
    print("\nStopped at JobRecord. Resume/transcript matching is intentionally not run.")


if __name__ == "__main__":
    main()
