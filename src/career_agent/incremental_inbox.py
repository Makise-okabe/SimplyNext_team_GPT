from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from career_agent.connectors.outlook_graph import OutlookGraphConnector
from career_agent.models.inbox import (
    CareerEmailRecord,
    InboxCheckpoint,
    IncrementalInboxResult,
)
from career_agent.parsers.forwarded_email import recover_forwarded_email

DEFAULT_CHECKPOINT_PATH = Path(".cache/inbox_checkpoint.json")
MAX_SEEN_IDS = 5000


def load_checkpoint(path: str | Path = DEFAULT_CHECKPOINT_PATH) -> InboxCheckpoint:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return InboxCheckpoint()
    try:
        return InboxCheckpoint.model_validate_json(
            checkpoint_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return InboxCheckpoint()


def save_checkpoint(
    checkpoint: InboxCheckpoint,
    path: str | Path = DEFAULT_CHECKPOINT_PATH,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        checkpoint.model_dump_json(indent=2),
        encoding="utf-8",
    )


def _source_key(sender_email: str | None) -> str | None:
    sender = (sender_email or "").strip().lower()
    if sender == "zeli.goh@nus.edu.sg":
        return "goh_ze_li"
    if sender in {"talentconnect@se.nus.edu.sg", "no-reply@kinobi.asia"}:
        return "talentconnect"
    return None


def _recent_items(connector: OutlookGraphConnector, scan: int) -> tuple[str, list[dict]]:
    token = connector._get_access_token()
    return token, connector._list_recent_items(token, top=scan)


def bootstrap_inbox(
    connector: OutlookGraphConnector,
    *,
    scan: int = 200,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
) -> InboxCheckpoint:
    """Mark the current inbox window as already seen without processing it."""
    _, items = _recent_items(connector, scan)
    ids = [item.get("id") for item in items if item.get("id")]
    checkpoint = InboxCheckpoint(
        seen_message_ids=ids[-MAX_SEEN_IDS:],
        updated_at=datetime.now(timezone.utc),
    )
    save_checkpoint(checkpoint, checkpoint_path)
    return checkpoint


def scan_new_career_emails(
    connector: OutlookGraphConnector,
    *,
    scan: int = 200,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    include_attachments: bool = True,
    commit: bool = True,
) -> IncrementalInboxResult:
    """Return only unseen trusted career emails from the dedicated inbox.

    All unseen message IDs are committed only after the scan succeeds. Irrelevant
    unseen emails are also marked seen so they do not reappear next run. Attachment
    bytes are downloaded only for relevant career emails.
    """
    checkpoint = load_checkpoint(checkpoint_path)
    seen = set(checkpoint.seen_message_ids)
    token, items = _recent_items(connector, scan)

    unseen_items = [
        item
        for item in reversed(items)
        if item.get("id") and item["id"] not in seen
    ]

    records: list[CareerEmailRecord] = []
    unseen_ids: list[str] = []

    for item in unseen_items:
        message_id = item["id"]
        unseen_ids.append(message_id)

        message = connector._graph_item_to_message(item)
        message = recover_forwarded_email(message)
        source = _source_key(message.sender_email)
        if source is None:
            continue

        if include_attachments and item.get("hasAttachments"):
            message = connector._enrich_attachments(message, token)

        records.append(CareerEmailRecord(source=source, email=message))

    result = IncrementalInboxResult(
        scanned_recent=len(items),
        unseen_total=len(unseen_items),
        filtered_out=len(unseen_items) - len(records),
        records=records,
        unseen_message_ids=unseen_ids,
    )

    if commit and unseen_ids:
        merged = [*checkpoint.seen_message_ids, *unseen_ids]
        deduped = list(dict.fromkeys(merged))[-MAX_SEEN_IDS:]
        save_checkpoint(
            InboxCheckpoint(
                seen_message_ids=deduped,
                updated_at=datetime.now(timezone.utc),
            ),
            checkpoint_path,
        )

    return result
