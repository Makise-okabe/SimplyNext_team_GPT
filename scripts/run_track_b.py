from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from career_agent.graph.workflow import career_agent_workflow
from career_agent.models.email import EmailMessage

RAW_EMAIL_PATH = Path("data/raw_email.json")
OUTPUT_PATH = Path("data/track_b_results.json")
TRUSTED_SENDERS = {"zeli.goh@nus.edu.sg", "no-reply@kinobi.asia"}
CAREER_KEYWORDS = ("career", "job", "intern", "opportun", "talentconnect", "enews", "industry")


def graph_item_to_email(item: dict) -> EmailMessage:
    sender = item.get("from", {}).get("emailAddress", {})
    body = item.get("body", {})
    received_raw = item.get("receivedDateTime")
    received_at = None
    if received_raw:
        received_at = datetime.fromisoformat(received_raw.replace("Z", "+00:00"))

    body_content = body.get("content") or ""
    content_type = (body.get("contentType") or "").lower()

    return EmailMessage(
        message_id=item.get("id") or "unknown",
        sender_name=sender.get("name"),
        sender_email=(sender.get("address") or "").lower() or None,
        subject=item.get("subject") or "",
        received_at=received_at,
        body_text=body_content if content_type == "text" else "",
        body_html=body_content if content_type == "html" else "",
        links=[item["webLink"]] if item.get("webLink") else [],
        attachments=["present"] if item.get("hasAttachments") else [],
    )


def load_graph_messages(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("value"), list):
        return payload["value"]
    raise ValueError("Expected a Graph response containing a 'value' list.")


def choose_messages(raw_messages: list[dict], limit: int, subject_hint: str | None) -> list[EmailMessage]:
    candidates: list[EmailMessage] = []
    for raw in raw_messages:
        message = graph_item_to_email(raw)
        if message.sender_email not in TRUSTED_SENDERS:
            continue
        subject_lower = message.subject.lower()
        if subject_hint and subject_hint.lower() not in subject_lower:
            continue
        if not subject_hint and not any(word in subject_lower for word in CAREER_KEYWORDS):
            continue
        candidates.append(message)

    candidates.sort(key=lambda item: item.received_at or datetime.min, reverse=True)
    return candidates[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SimplyNext Track B on local Graph email JSON.")
    parser.add_argument("--limit", type=int, default=1, help="Number of career emails to process.")
    parser.add_argument("--subject", default=None, help="Optional case-insensitive subject substring.")
    args = parser.parse_args()

    if not RAW_EMAIL_PATH.exists():
        raise FileNotFoundError(f"Missing {RAW_EMAIL_PATH}")

    messages = choose_messages(load_graph_messages(RAW_EMAIL_PATH), args.limit, args.subject)
    if not messages:
        raise RuntimeError("No matching trusted career emails found.")

    all_results: list[dict] = []
    for index, message in enumerate(messages, start=1):
        print("=" * 96)
        print(f"EMAIL {index}/{len(messages)}")
        print(f"Sender : {message.sender_name} <{message.sender_email}>")
        print(f"Subject: {message.subject}")

        result = career_agent_workflow.invoke(
            {
                "email": message.model_dump(mode="json"),
                "errors": [],
            }
        )

        signals = result.get("opportunity_signals", [])
        pages = result.get("resolved_pages", [])
        jobs = result.get("verified_jobs", [])
        errors = result.get("errors", [])

        print(f"Signals       : {len(signals)}")
        print(f"Pages resolved: {len(pages)}")
        print(f"Verified jobs : {len(jobs)}")
        print(f"Errors        : {len(errors)}")

        if errors:
            print("\nPIPELINE ERRORS")
            for error_index, error in enumerate(errors, start=1):
                print(f"  {error_index}. {error}")

        for job_index, job in enumerate(jobs, start=1):
            print(f"\n  JOB {job_index}: {job.get('company')} — {job.get('title')}")
            print(f"    status : {job.get('verification_status')}")
            print(f"    url    : {job.get('official_url')}")

        all_results.append(
            {
                "source_email": {
                    "message_id": message.message_id,
                    "sender_email": message.sender_email,
                    "subject": message.subject,
                    "received_at": message.received_at.isoformat() if message.received_at else None,
                },
                "opportunity_signals": signals,
                "resolved_pages": pages,
                "candidate_jobs": result.get("candidate_jobs", []),
                "verified_jobs": jobs,
                "errors": errors,
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "=" * 96)
    print(f"Saved full Track B output to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
