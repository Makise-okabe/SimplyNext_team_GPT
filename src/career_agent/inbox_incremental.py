from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from career_agent.connectors.outlook_graph import OutlookGraphConnector
from career_agent.models.inbox import InboxCheckpoint, IncrementalInboxResult

DEFAULT_CHECKPOINT_PATH = Path(".cache/incremental_inbox_checkpoint.json")
MAX_SEEN_IDS = 5000


def checkpoint_path() -> Path:
    configured = os.getenv("INBOX_CHECKPOINT_PATH")
    return Path(configured) if configured else DEFAULT_CHECKPOINT_PATH


def load_checkpoint(path: Path | None = None) -> InboxCheckpoint:
    target = path or checkpoint_path()
    if not target.exists():
        raise FileNotFoundError(
            f"Inbox checkpoint does not exist at {target}. "
            "Run the incremental inbox script with --bootstrap first."
        )
    return InboxCheckpoint.model_validate_json(target.read_text(encoding="utf-8"))


def save_checkpoint(checkpoint: InboxCheckpoint, path: Path | None = None) -> None:
    target = path or checkpoint_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        checkpoint.model_dump_json(indent=2),
        encoding="utf-8",
    )


def bootstrap_checkpoint(
    *,
    path: Path | None = None,
    overwrite: bool = False,
    now: datetime | None = None,
) -> InboxCheckpoint:
    """Create the manual baseline used by prototype acceptance testing.

    Bootstrap intentionally does not query Outlook. The instant this function is
    called becomes the boundary: mail with a Graph received timestamp after the
    boundary is eligible on the next --run. This makes the test deterministic and
    avoids downloading any existing inbox content just to establish a baseline.
    """
    target = path or checkpoint_path()
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"Checkpoint already exists at {target}. Use --reset with --bootstrap "
            "only when you intentionally want a new baseline."
        )

    boundary = now or datetime.now(timezone.utc)
    if boundary.tzinfo is None:
        boundary = boundary.replace(tzinfo=timezone.utc)

    checkpoint = InboxCheckpoint(
        baseline_at=boundary,
        seen_message_ids=[],
        updated_at=boundary,
    )
    save_checkpoint(checkpoint, target)
    return checkpoint


def commit_result(
    checkpoint: InboxCheckpoint,
    result: IncrementalInboxResult,
    *,
    path: Path | None = None,
    now: datetime | None = None,
) -> InboxCheckpoint:
    """Commit every newly inspected message ID, including irrelevant mail."""
    ordered = list(
        dict.fromkeys([*checkpoint.seen_message_ids, *result.unseen_message_ids])
    )
    if len(ordered) > MAX_SEEN_IDS:
        ordered = ordered[-MAX_SEEN_IDS:]

    updated = now or datetime.now(timezone.utc)
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)

    next_checkpoint = checkpoint.model_copy(
        update={
            "seen_message_ids": ordered,
            "updated_at": updated,
        }
    )
    save_checkpoint(next_checkpoint, path)
    return next_checkpoint


def run_incremental_inbox(
    connector: OutlookGraphConnector,
    *,
    scan: int = 100,
    commit: bool = True,
    path: Path | None = None,
) -> IncrementalInboxResult:
    checkpoint = load_checkpoint(path)
    result = connector.scan_incremental(
        baseline_at=checkpoint.baseline_at,
        seen_message_ids=set(checkpoint.seen_message_ids),
        top=scan,
        include_attachments=True,
    )

    if commit:
        commit_result(checkpoint, result, path=path)
        result = result.model_copy(update={"checkpoint_committed": True})
    return result
