from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from career_agent.connectors.outlook_graph import OutlookGraphConnector
from career_agent.models.inbox import InboxCheckpoint, IncrementalInboxResult

DEFAULT_CHECKPOINT_PATH = Path(".cache/inbox_checkpoint.json")
MAX_SEEN_IDS = 5000


def load_checkpoint(path: str | Path = DEFAULT_CHECKPOINT_PATH) -> InboxCheckpoint:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Inbox checkpoint does not exist at {checkpoint_path}. "
            "Run scripts/run_incremental_inbox.py --bootstrap first."
        )
    return InboxCheckpoint.model_validate_json(
        checkpoint_path.read_text(encoding="utf-8")
    )


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


def bootstrap_inbox(
    connector: OutlookGraphConnector,
    *,
    scan: int = 200,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
) -> InboxCheckpoint:
    """Mark the current inbox as the baseline without processing its content.

    The timestamp is captured before the Graph snapshot. Therefore a message that
    arrives during bootstrap is safe either way: if it appears in the snapshot its
    ID is already seen; if it does not, its received timestamp is after baseline
    and the next run can still pick it up.
    """
    baseline_at = datetime.now(timezone.utc)
    token = connector._get_access_token()
    items = connector._list_recent_items(token, top=scan)
    ids = [item.get("id") for item in items if item.get("id")]

    checkpoint = InboxCheckpoint(
        baseline_at=baseline_at,
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
    """Return only genuinely new Goh/TalentConnect career emails.

    All newly inspected message IDs, including irrelevant mail, are committed only
    after the scan succeeds. The connector performs sender recovery/filtering
    before downloading attachments, so irrelevant mail stays cheap.
    """
    checkpoint = load_checkpoint(checkpoint_path)
    result = connector.scan_incremental(
        baseline_at=checkpoint.baseline_at,
        seen_message_ids=set(checkpoint.seen_message_ids),
        top=scan,
        include_attachments=include_attachments,
    )

    if not commit:
        return result

    merged = list(
        dict.fromkeys([*checkpoint.seen_message_ids, *result.unseen_message_ids])
    )[-MAX_SEEN_IDS:]
    next_checkpoint = checkpoint.model_copy(
        update={
            "seen_message_ids": merged,
            "updated_at": datetime.now(timezone.utc),
        }
    )
    save_checkpoint(next_checkpoint, checkpoint_path)
    return result.model_copy(update={"checkpoint_committed": True})
