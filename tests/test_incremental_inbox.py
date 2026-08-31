from __future__ import annotations

from datetime import datetime, timezone

from career_agent.incremental_inbox import (
    bootstrap_inbox,
    load_checkpoint,
    scan_new_career_emails,
)
from career_agent.models.email import EmailMessage


class FakeConnector:
    def __init__(self, items: list[dict]) -> None:
        self.items = items
        self.attachment_calls: list[str] = []

    def _get_access_token(self) -> str:
        return "token"

    def _list_recent_items(self, token: str, top: int) -> list[dict]:
        assert token == "token"
        return self.items[:top]

    @staticmethod
    def _graph_item_to_message(item: dict) -> EmailMessage:
        return EmailMessage(
            message_id=item["id"],
            sender_name=item.get("sender_name"),
            sender_email=item.get("sender_email"),
            subject=item.get("subject", ""),
            received_at=item.get("received_at"),
            body_text=item.get("body_text", ""),
            transport_sender_name=item.get("transport_sender_name"),
            transport_sender_email=item.get("transport_sender_email"),
        )

    def _enrich_attachments(self, message: EmailMessage, token: str) -> EmailMessage:
        self.attachment_calls.append(message.message_id)
        return message.model_copy(
            update={
                "attachments": ["JD.pdf"],
                "attachment_text": "sample JD text",
            }
        )


def _item(
    message_id: str,
    sender: str,
    *,
    subject: str = "mail",
    attachments: bool = False,
) -> dict:
    return {
        "id": message_id,
        "sender_name": sender,
        "sender_email": sender,
        "subject": subject,
        "received_at": datetime(2026, 8, 31, tzinfo=timezone.utc),
        "body_text": "body",
        "hasAttachments": attachments,
    }


def test_bootstrap_marks_current_window_seen_without_processing(tmp_path) -> None:
    path = tmp_path / "checkpoint.json"
    connector = FakeConnector(
        [
            _item("old-3", "friend@example.com"),
            _item("old-2", "zeli.goh@nus.edu.sg"),
            _item("old-1", "talentconnect@se.nus.edu.sg"),
        ]
    )

    checkpoint = bootstrap_inbox(connector, checkpoint_path=path)

    assert set(checkpoint.seen_message_ids) == {"old-1", "old-2", "old-3"}
    assert connector.attachment_calls == []
    assert set(load_checkpoint(path).seen_message_ids) == {"old-1", "old-2", "old-3"}


def test_five_new_emails_filter_to_one_relevant_record(tmp_path) -> None:
    path = tmp_path / "checkpoint.json"
    connector = FakeConnector([_item("old", "friend@example.com")])
    bootstrap_inbox(connector, checkpoint_path=path)

    connector.items = [
        _item("new-5", "newsletter@example.com"),
        _item("new-4", "shopping@example.com"),
        _item("new-3", "zeli.goh@nus.edu.sg", subject="Industry Opportunities", attachments=True),
        _item("new-2", "microsoft@example.com"),
        _item("new-1", "friend@example.com"),
        _item("old", "friend@example.com"),
    ]

    result = scan_new_career_emails(connector, checkpoint_path=path, commit=True)

    assert result.unseen_total == 5
    assert result.filtered_out == 4
    assert len(result.records) == 1
    assert result.records[0].source == "goh_ze_li"
    assert result.records[0].email.subject == "Industry Opportunities"
    assert result.records[0].email.attachments == ["JD.pdf"]
    assert connector.attachment_calls == ["new-3"]
    assert set(load_checkpoint(path).seen_message_ids) >= {
        "old",
        "new-1",
        "new-2",
        "new-3",
        "new-4",
        "new-5",
    }


def test_dry_run_does_not_advance_checkpoint(tmp_path) -> None:
    path = tmp_path / "checkpoint.json"
    connector = FakeConnector([_item("old", "friend@example.com")])
    bootstrap_inbox(connector, checkpoint_path=path)

    connector.items = [
        _item("new", "talentconnect@se.nus.edu.sg", subject="TalentConnect"),
        _item("old", "friend@example.com"),
    ]

    result = scan_new_career_emails(connector, checkpoint_path=path, commit=False)

    assert result.unseen_total == 1
    assert len(result.records) == 1
    assert result.records[0].source == "talentconnect"
    assert "new" not in load_checkpoint(path).seen_message_ids


def test_irrelevant_new_mail_is_committed_as_seen(tmp_path) -> None:
    path = tmp_path / "checkpoint.json"
    connector = FakeConnector([_item("old", "friend@example.com")])
    bootstrap_inbox(connector, checkpoint_path=path)

    connector.items = [
        _item("irrelevant", "random@example.com"),
        _item("old", "friend@example.com"),
    ]
    first = scan_new_career_emails(connector, checkpoint_path=path, commit=True)
    second = scan_new_career_emails(connector, checkpoint_path=path, commit=True)

    assert first.unseen_total == 1
    assert first.filtered_out == 1
    assert second.unseen_total == 0
