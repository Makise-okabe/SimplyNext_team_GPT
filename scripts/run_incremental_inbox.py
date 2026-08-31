from __future__ import annotations

import argparse

from career_agent.connectors.outlook_graph import OutlookGraphConnector
from career_agent.incremental_inbox import bootstrap_inbox, scan_new_career_emails


def _short(value: str | None, limit: int = 140) -> str:
    text = " ".join((value or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SimplyNext incremental inbox checkpoint + trusted career-email intake."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bootstrap", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--scan", type=int, default=200)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the real scan/filter logic without advancing the checkpoint.",
    )
    args = parser.parse_args()

    connector = OutlookGraphConnector()

    if args.bootstrap:
        checkpoint = bootstrap_inbox(connector, scan=args.scan)
        print("=" * 88)
        print("SIMPLYNEXT — INCREMENTAL INBOX BASELINE")
        print("=" * 88)
        print("Current inbox messages marked as seen:", len(checkpoint.seen_message_ids))
        print("Checkpoint updated at                :", checkpoint.updated_at)
        print("No email content was processed.")
        print("\nNow send your test emails, then run:")
        print("uv run python scripts/run_incremental_inbox.py --run")
        return

    result = scan_new_career_emails(
        connector,
        scan=args.scan,
        include_attachments=True,
        commit=not args.dry_run,
    )

    print("=" * 88)
    print("SIMPLYNEXT — NEW EMAIL INTAKE")
    print("=" * 88)
    print("Recent inbox window :", result.scanned_recent)
    print("New emails          :", result.unseen_total)
    print("Filtered out        :", result.filtered_out)
    print("Career emails       :", len(result.records))
    print("Checkpoint advanced :", "NO (dry-run)" if args.dry_run else "YES")

    if not result.records:
        print("\nNo new Goh Ze Li / TalentConnect email was found.")
        return

    for index, record in enumerate(result.records, start=1):
        email = record.email
        print("\n" + "-" * 88)
        print(f"CAREER EMAIL RECORD {index}/{len(result.records)}")
        print("source key      :", record.source)
        print("original sender :", f"{email.sender_name} <{email.sender_email}>")
        print("subject         :", _short(email.subject))
        print("received        :", email.received_at)
        print("message id      :", email.message_id)
        print("transport sender:", email.transport_sender_email or "None")
        print("attachments     :", "; ".join(email.attachments) or "None")
        print("attachment text :", len(email.attachment_text), "chars")
        print("body text       :", len(email.body_text), "chars")
        print("links           :", len(email.links))


if __name__ == "__main__":
    main()
