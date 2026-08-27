from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from career_agent.models.email import EmailMessage
from career_agent.nodes.normalize_email import normalize_email

RAW_EMAIL_PATH = Path("data/raw_email.json")
ALLOWED_SENDERS = {
    "zeli.goh@nus.edu.sg",
    "no-reply@kinobi.asia",
}


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

    raise ValueError(
        "Unsupported raw_email.json format. Expected a Graph response with a "
        "'value' list or a top-level list of messages."
    )


def main() -> None:
    if not RAW_EMAIL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {RAW_EMAIL_PATH}. Put your Graph Explorer JSON at "
            "data/raw_email.json and run this script again."
        )

    raw_messages = load_graph_messages(RAW_EMAIL_PATH)
    print(f"Loaded {len(raw_messages)} raw Graph messages.\n")

    matched = 0
    for raw in raw_messages:
        message = graph_item_to_email(raw)
        if message.sender_email not in ALLOWED_SENDERS:
            continue

        matched += 1
        normalized = normalize_email(message)
        preview = normalized.body_text[:700].replace("\r", "")

        print("=" * 88)
        print(f"[{matched}] {normalized.sender_name} <{normalized.sender_email}>")
        print(f"Subject: {normalized.subject}")
        print(f"Received: {normalized.received_at}")
        print(f"Clean text chars: {len(normalized.body_text)}")
        print(f"Links found: {len(normalized.links)}")
        print("\nTEXT PREVIEW")
        print(preview or "<empty>")
        print("\nURLS")
        if normalized.links:
            for index, url in enumerate(normalized.links, start=1):
                print(f"  {index}. {url}")
        else:
            print("  <none>")
        print()

    print(f"Finished. Matched {matched} allowed career emails.")


if __name__ == "__main__":
    main()
