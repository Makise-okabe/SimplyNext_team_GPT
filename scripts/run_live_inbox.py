from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from career_agent.connectors.outlook_graph import OutlookGraphConnector
from career_agent.graph.workflow import career_agent_workflow
from career_agent.nodes.normalize_email import normalize_email
from career_agent.storage.sqlite import OpportunityStore


def _short(value: str, limit: int = 95) -> str:
    value = " ".join((value or "").split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SimplyNext directly on the dedicated live Outlook inbox."
    )
    parser.add_argument("--scan", type=int, default=20)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--subject", default=None)
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Validate Graph/forwarding/PDF ingestion without LLM/web calls.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional raw diagnostic JSON output. Off by default.",
    )
    parser.add_argument(
        "--store",
        nargs="?",
        const=Path("private_data/simplynext.db"),
        type=Path,
        default=None,
        help=(
            "Persist only normalized job records in SQLite. Raw email/PDF text is never stored. "
            "Optional path; default when flag is present: private_data/simplynext.db"
        ),
    )
    args = parser.parse_args()

    connector = OutlookGraphConnector()
    account = connector.get_account()
    account_email = account.get("mail") or account.get("userPrincipalName")

    messages = connector.get_messages(top=args.scan, include_attachments=True)
    if args.subject:
        needle = args.subject.lower()
        messages = [message for message in messages if needle in message.subject.lower()]
    messages = messages[: args.limit]

    print("=" * 88)
    print("SIMPLYNEXT LIVE INBOX")
    print("=" * 88)
    print("Account :", account_email)
    print("Scanned :", args.scan)
    print("Career emails recovered:", len(messages))

    if not messages:
        raise RuntimeError("No matching career emails were recovered from the live inbox.")

    store = OpportunityStore(args.store) if args.store else None
    outputs: list[dict] = []
    all_jobs: list[dict] = []
    stored_new = 0
    stored_updated = 0

    for index, message in enumerate(messages, start=1):
        normalized = normalize_email(message)
        print("\n" + "-" * 88)
        print(f"EMAIL {index}/{len(messages)}")
        print("Source     :", f"{normalized.sender_name} <{normalized.sender_email}>")
        if normalized.transport_sender_email:
            print(
                "Forwarded by:",
                f"{normalized.transport_sender_name} <{normalized.transport_sender_email}>",
            )
        print("Subject    :", _short(normalized.subject, 120))
        print("Attachments:", ", ".join(normalized.attachments) or "none")
        print("Body chars :", len(normalized.body_text))
        print("Links      :", len(normalized.links))
        if normalized.attachment_text:
            print("PDF text   :", len(normalized.attachment_text), "chars")

        if args.ingest_only:
            outputs.append({"email": normalized.model_dump(mode="json"), "mode": "ingest_only"})
            continue

        result = career_agent_workflow.invoke(
            {"email": message.model_dump(mode="json"), "errors": []}
        )

        signals = result.get("opportunity_signals", [])
        jobs = result.get("verified_jobs", [])
        errors = result.get("errors", [])
        all_jobs.extend(jobs)

        print("Signals     :", len(signals))
        print("Job records :", len(jobs))
        print("Errors      :", len(errors))

        for job_index, job in enumerate(jobs, start=1):
            print(
                f"  JOB {job_index}: "
                f"{job.get('company') or 'Unknown'} — {job.get('title') or 'Unknown'}"
            )
            print("    status :", job.get("verification_status"))
            print("    basis  :", job.get("verification_basis"))
            print("    official:", job.get("official_url"))
            print("    apply  :", job.get("application_url"))

        if store and jobs:
            inserted, updated = store.upsert_jobs(
                jobs,
                source_email=normalized.model_dump(mode="json"),
            )
            stored_new += inserted
            stored_updated += updated

        if errors:
            print("  Pipeline errors:")
            for error in errors:
                print("   -", error)

        outputs.append(
            {
                "email": normalized.model_dump(mode="json"),
                "opportunity_signals": signals,
                "resolved_pages": result.get("resolved_pages", []),
                "candidate_jobs": result.get("candidate_jobs", []),
                "verified_jobs": jobs,
                "errors": errors,
            }
        )

    if not args.ingest_only:
        counts = Counter(job.get("verification_status") or "unknown" for job in all_jobs)
        print("\n" + "=" * 88)
        print("PIPELINE SUMMARY")
        print("=" * 88)
        print("Job records      :", len(all_jobs))
        print("Official verified:", counts.get("verified", 0))
        print("Source verified  :", counts.get("source_verified", 0))
        print("Partial          :", counts.get("partial", 0))
        print("Unresolved       :", counts.get("unresolved", 0))

    if store:
        print("Structured memory:", args.store)
        print("New records      :", stored_new)
        print("Updated records  :", stored_updated)
        print("Total in memory  :", store.count())
        print("Raw email/PDF text persisted: NO")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(outputs, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print("Diagnostic JSON saved:", args.output)
    else:
        print("No raw diagnostic JSON was persisted.")


if __name__ == "__main__":
    main()
