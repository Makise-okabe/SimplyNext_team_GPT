from __future__ import annotations

from career_agent.connectors.outlook_graph import OutlookGraphConnector
from career_agent.incremental_inbox import (
    bootstrap_inbox,
    load_checkpoint,
    scan_new_career_emails,
)


class FakeConnector(OutlookGraphConnector):
    def __init__(self, items: list[dict]) -> None:
        self.items = items
        self.attachment_calls: list[str] = []

    def _get_access_token(self) -> str:
        return "token"

    def _list_recent_items(self, token: str, top: int) -> list[dict]:
        assert token == "token"
        return self.items[:top]

    def _enrich_attachments(self, message, token: str):
        self.attachment_calls.append(message.message_id)
        return message.model_copy(
            update={
                "attachments": ["JD.pdf"],
                "attachment_text": "sample JD text",
            }
        )


def _item(
    message_id: str,
    sender_email: str,
    *,
    subject: str = "mail",
    body: str = "body",
    received: str = "2099-08-31T09:00:00Z",
    attachments: bool = False,
) -> dict:
    return {
        "id": message_id,
        "subject": subject,
        "from": {
            "emailAddress": {
                "name": sender_email,
                "address": sender_email,
            }
        },
        "receivedDateTime": received,
        "body": {"contentType": "text", "content": body},
        "bodyPreview": body[:80],
        "webLink": f"https://outlook.live.com/owa/?ItemID={message_id}",
        "hasAttachments": attachments,
    }


def _goh_forward_body() -> str:
    return (
        "Get Outlook for iOS\n"
        "________________________________\n"
        "From: Goh Ze Li <zeli.goh@nus.edu.sg>\n"
        "Sent: Monday, 31 August 2099 17:00:00\n"
        "Subject: From Your CDE Career Advisors: Industry Opportunities\n\n"
        "Dear Students, career opportunities here."
    )


def test_bootstrap_marks_current_window_seen_without_processing(tmp_path) -> None:
    path = tmp_path / "checkpoint.json"
    connector = FakeConnector(
        [
            _item("old-3", "friend@example.com"),
            _item("old-2", "student@u.nus.edu", body=_goh_forward_body()),
            _item("old-1", "talentconnect@se.nus.edu.sg"),
        ]
    )

    checkpoint = bootstrap_inbox(connector, checkpoint_path=path)

    assert set(checkpoint.seen_message_ids) == {"old-1", "old-2", "old-3"}
    assert checkpoint.baseline_at is not None
    assert connector.attachment_calls == []
    assert set(load_checkpoint(path).seen_message_ids) == {"old-1", "old-2", "old-3"}


def test_five_new_emails_filter_to_one_forwarded_goh_record(tmp_path) -> None:
    path = tmp_path / "checkpoint.json"
    connector = FakeConnector([_item("old", "friend@example.com")])
    bootstrap_inbox(connector, checkpoint_path=path)

    connector.items = [
        _item("new-5", "newsletter@example.com", attachments=True),
        _item("new-4", "shopping@example.com", attachments=True),
        _item(
            "new-3",
            "student@u.nus.edu",
            subject="Fw: From Your CDE Career Advisors: Industry Opportunities",
            body=_goh_forward_body(),
            attachments=True,
        ),
        _item("new-2", "microsoft@example.com", attachments=True),
        _item("new-1", "friend@example.com", attachments=True),
        _item("old", "friend@example.com"),
    ]

    result = scan_new_career_emails(connector, checkpoint_path=path, commit=True)

    assert result.unseen_total == 5
    assert result.filtered_out == 4
    assert len(result.records) == 1
    assert result.records[0].source == "goh_ze_li"
    assert result.records[0].email.sender_email == "zeli.goh@nus.edu.sg"
    assert result.records[0].email.transport_sender_email == "student@u.nus.edu"
    assert result.records[0].email.subject == "From Your CDE Career Advisors: Industry Opportunities"
    assert result.records[0].email.attachments == ["JD.pdf"]
    # All five messages claim to have attachments, but only the relevant career
    # email is allowed to trigger attachment retrieval.
    assert connector.attachment_calls == ["new-3"]
    assert result.checkpoint_committed is True
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
    assert result.checkpoint_committed is False
    assert "new" not in load_checkpoint(path).seen_message_ids


def test_irrelevant_new_mail_is_committed_and_not_returned_again(tmp_path) -> None:
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
    assert second.filtered_out == 0
