from __future__ import annotations

import argparse
import json
from pathlib import Path

from career_agent.connectors.outlook_graph import OutlookGraphConnector
from career_agent.graph.workflow import career_agent_workflow
from career_agent.nodes.normalize_email import normalize_email


def _short(value: str, limit: int = 95) -> str:
    value = " ".join((value or "").split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SimplyNext directly on the dedicated live Outlook inbox."
    )
    parser.add_argument(
        "--scan",
        type=int,
        default=20,
        help="How many recent inbox messages Graph should inspect.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Maximum recovered career emails to process.",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Optional case-insensitive subject substring.",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Validate Graph/forwarding/PDF ingestion without calling the LLM/web pipeline.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Nothing is persisted unless you set this.",
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

    outputs: list[dict] = []

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
            outputs.append(
                {
                    "email": normalized.model_dump(mode="json"),
                    "mode": "ingest_only",
                }
            )
            continue

        result = career_agent_workflow.invoke(
            {
                "email": message.model_dump(mode="json"),
                "errors": [],
            }
        )

        signals = result.get("opportunity_signals", [])
        jobs = result.get("verified_jobs", [])
        errors = result.get("errors", [])

        print("Signals     :", len(signals))
        print("Verified jobs:", len(jobs))
        print("Errors      :", len(errors))

        for job_index, job in enumerate(jobs, start=1):
            print(
                f"  JOB {job_index}: "
                f"{job.get('company') or 'Unknown'} — {job.get('title') or 'Unknown'}"
            )
            print("    status:", job.get("verification_status"))
            print("    url   :", job.get("official_url"))

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

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(outputs, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print("\nSaved:", args.output)
    else:
        print("\nNo local email/JD data was persisted.")


if __name__ == "__main__":
    main()
