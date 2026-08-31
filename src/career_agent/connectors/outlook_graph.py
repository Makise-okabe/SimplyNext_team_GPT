from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import msal
import requests
from dotenv import load_dotenv

from career_agent.config import Settings
from career_agent.connectors.base import EmailConnector
from career_agent.models.email import EmailMessage
from career_agent.models.inbox import CareerEmailRecord, IncrementalInboxResult
from career_agent.parsers.attachments import extract_pdf_text, format_attachment_text
from career_agent.parsers.forwarded_email import recover_forwarded_email

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
DEFAULT_SCOPES = ["User.Read", "Mail.Read"]

CAREER_SOURCE_BY_SENDER = {
    "zeli.goh@nus.edu.sg": "goh_ze_li",
    "talentconnect@se.nus.edu.sg": "talentconnect",
    # Keep compatibility with older TalentConnect messages seen during prototype work.
    "no-reply@kinobi.asia": "talentconnect",
}


class OutlookGraphConnector(EmailConnector):
    """Live Microsoft Graph connector for the dedicated SimplyNext inbox.

    Raw PDFs are parsed in memory and are not persisted to the repository.
    Forwarded emails are normalized back to the original career source while
    retaining the forwarding account in ``transport_sender_*``.
    """

    def __init__(
        self,
        client_id: str | None = None,
        authority: str | None = None,
        token_cache_path: str | Path | None = None,
        timeout: int = 30,
    ) -> None:
        load_dotenv()

        self.client_id = client_id or os.getenv("MS_CLIENT_ID")
        self.authority = authority or os.getenv(
            "MS_AUTHORITY",
            "https://login.microsoftonline.com/consumers",
        )
        self.timeout = timeout
        self.token_cache_path = Path(
            token_cache_path
            or os.getenv("MS_TOKEN_CACHE_PATH", ".cache/msal_token_cache.json")
        )

        if not self.client_id:
            raise ValueError("MS_CLIENT_ID is missing from .env")

        self.cache = msal.SerializableTokenCache()
        if self.token_cache_path.exists():
            try:
                self.cache.deserialize(
                    self.token_cache_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                # A damaged local cache should never block a fresh login.
                pass

        self.app = msal.PublicClientApplication(
            client_id=self.client_id,
            authority=self.authority,
            token_cache=self.cache,
        )

    def _persist_cache(self) -> None:
        if not self.cache.has_state_changed:
            return

        self.token_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_cache_path.write_text(self.cache.serialize(), encoding="utf-8")

    def _get_access_token(self) -> str:
        result = None
        accounts = self.app.get_accounts()

        if accounts:
            result = self.app.acquire_token_silent(
                DEFAULT_SCOPES,
                account=accounts[0],
            )

        if not result:
            result = self.app.acquire_token_interactive(scopes=DEFAULT_SCOPES)

        self._persist_cache()

        if not result or "access_token" not in result:
            error = (result or {}).get("error", "unknown_error")
            description = (result or {}).get(
                "error_description",
                "No error description returned.",
            )
            raise RuntimeError(
                f"Microsoft authentication failed: {error}: {description}"
            )

        return result["access_token"]

    def _request_json(
        self,
        endpoint: str,
        token: str,
        params: dict[str, str] | None = None,
    ) -> dict:
        response = requests.get(
            f"{GRAPH_BASE_URL}{endpoint}",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "Prefer": 'outlook.body-content-type="html"',
            },
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _request_bytes(self, endpoint: str, token: str) -> bytes:
        response = requests.get(
            f"{GRAPH_BASE_URL}{endpoint}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.content

    def get_account(self) -> dict:
        token = self._get_access_token()
        return self._request_json(
            "/me",
            token,
            params={"$select": "displayName,userPrincipalName,mail"},
        )

    def _list_recent_items(self, token: str, top: int) -> list[dict]:
        payload = self._request_json(
            "/me/mailFolders/inbox/messages",
            token,
            params={
                "$top": str(top),
                "$select": (
                    "id,subject,from,receivedDateTime,body,bodyPreview,"
                    "webLink,hasAttachments"
                ),
                "$orderby": "receivedDateTime desc",
            },
        )
        return payload.get("value", [])

    def _list_attachments(self, message_id: str, token: str) -> list[dict]:
        payload = self._request_json(
            f"/me/messages/{message_id}/attachments",
            token,
            params={"$select": "id,name,contentType,size,isInline"},
        )
        return payload.get("value", [])

    def get_attachment_bytes(
        self,
        message_id: str,
        attachment_id: str,
        token: str | None = None,
    ) -> bytes:
        access_token = token or self._get_access_token()
        return self._request_bytes(
            f"/me/messages/{message_id}/attachments/{attachment_id}/$value",
            access_token,
        )

    @staticmethod
    def _graph_item_to_message(item: dict) -> EmailMessage:
        sender = item.get("from", {}).get("emailAddress", {})
        body = item.get("body", {})
        received_raw = item.get("receivedDateTime")
        received_at = None
        if received_raw:
            received_at = datetime.fromisoformat(
                received_raw.replace("Z", "+00:00")
            )

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
        )

    def _enrich_attachments(
        self,
        message: EmailMessage,
        token: str,
    ) -> EmailMessage:
        attachment_names: list[str] = []
        text_blocks: list[str] = []

        for attachment in self._list_attachments(message.message_id, token):
            if attachment.get("isInline"):
                continue

            name = (attachment.get("name") or "unnamed attachment").strip()
            attachment_names.append(name)

            content_type = (attachment.get("contentType") or "").lower()
            is_pdf = (
                content_type == "application/pdf"
                or name.lower().endswith(".pdf")
            )
            if not is_pdf:
                continue

            attachment_id = attachment.get("id")
            if not attachment_id:
                continue

            try:
                raw = self.get_attachment_bytes(
                    message.message_id,
                    attachment_id,
                    token=token,
                )
                text = extract_pdf_text(raw)
            except Exception as exc:
                text_blocks.append(
                    f"ATTACHMENT: {name}\n"
                    f"[PDF extraction failed: {type(exc).__name__}: {exc}]"
                )
                continue

            block = format_attachment_text(name, text)
            if block:
                text_blocks.append(block)

        return message.model_copy(
            update={
                "attachments": attachment_names,
                "attachment_text": "\n\n".join(text_blocks),
            }
        )

    def scan_incremental(
        self,
        *,
        baseline_at: datetime,
        seen_message_ids: set[str],
        top: int = 100,
        include_attachments: bool = True,
    ) -> IncrementalInboxResult:
        """Scan only mail newer than a manual bootstrap boundary.

        Every unseen message is counted so irrelevant mail can be committed as
        seen too. Forwarded sender recovery and trusted-source filtering happen
        before attachment retrieval, so irrelevant mail stays cheap.
        """
        token = self._get_access_token()
        items = self._list_recent_items(token, top=top)
        unseen_items: list[dict] = []
        records: list[CareerEmailRecord] = []

        for item in items:
            message_id = item.get("id") or ""
            if not message_id or message_id in seen_message_ids:
                continue

            message = self._graph_item_to_message(item)
            if not message.received_at or message.received_at <= baseline_at:
                continue

            unseen_items.append(item)
            normalized = recover_forwarded_email(message)
            sender = (normalized.sender_email or "").strip().lower()
            source = CAREER_SOURCE_BY_SENDER.get(sender)
            if source is None:
                continue

            if include_attachments and item.get("hasAttachments"):
                normalized = self._enrich_attachments(normalized, token)

            records.append(CareerEmailRecord(source=source, email=normalized))

        return IncrementalInboxResult(
            scanned_recent=len(items),
            unseen_total=len(unseen_items),
            filtered_out=len(unseen_items) - len(records),
            records=records,
            unseen_message_ids=[item.get("id") for item in unseen_items if item.get("id")],
            checkpoint_committed=False,
        )

    def get_messages(
        self,
        top: int = 20,
        include_attachments: bool = True,
    ) -> list[EmailMessage]:
        """Return normalized career emails from the live dedicated inbox."""
        token = self._get_access_token()
        trusted = Settings().trusted_senders
        messages: list[EmailMessage] = []

        for item in self._list_recent_items(token, top=top):
            message = self._graph_item_to_message(item)
            message = recover_forwarded_email(message)

            sender = (message.sender_email or "").lower()
            if sender not in trusted:
                continue

            if include_attachments and item.get("hasAttachments"):
                message = self._enrich_attachments(message, token)

            messages.append(message)

        return messages
