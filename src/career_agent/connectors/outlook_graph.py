from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

import httpx
import msal
from dotenv import load_dotenv

from career_agent.connectors.base import EmailConnector
from career_agent.models.email import EmailMessage

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
MAIL_SCOPES = ["Mail.Read"]
DEFAULT_SENDERS = (
    "zeli.goh@nus.edu.sg",
    "no-reply@kinobi.asia",
)


class OutlookGraphConnector(EmailConnector):
    """Read selected NUS career emails through delegated Microsoft Graph access.

    Authentication uses MSAL device-code flow. The user's password is entered only
    on Microsoft's login page. Access/refresh tokens are cached locally and must
    never be committed to Git.
    """

    def __init__(
        self,
        client_id: str | None = None,
        tenant_id: str | None = None,
        token_cache_path: str | Path = "token_cache.json",
        senders: Iterable[str] = DEFAULT_SENDERS,
    ) -> None:
        load_dotenv()

        self.client_id = client_id or os.getenv("MS_CLIENT_ID")
        self.tenant_id = tenant_id or os.getenv("MS_TENANT_ID", "organizations")
        self.token_cache_path = Path(token_cache_path)
        self.senders = tuple(sender.lower() for sender in senders)

        if not self.client_id:
            raise ValueError(
                "MS_CLIENT_ID is missing. Create a Microsoft Entra app registration "
                "and put its Application (client) ID in your local .env file."
            )

        self.cache = msal.SerializableTokenCache()
        if self.token_cache_path.exists():
            self.cache.deserialize(self.token_cache_path.read_text(encoding="utf-8"))

        authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        self.app = msal.PublicClientApplication(
            client_id=self.client_id,
            authority=authority,
            token_cache=self.cache,
        )

    def _persist_cache(self) -> None:
        if self.cache.has_state_changed:
            self.token_cache_path.write_text(self.cache.serialize(), encoding="utf-8")

    def _get_access_token(self) -> str:
        accounts = self.app.get_accounts()
        result = None

        if accounts:
            result = self.app.acquire_token_silent(MAIL_SCOPES, account=accounts[0])

        if not result:
            flow = self.app.initiate_device_flow(scopes=MAIL_SCOPES)
            if "user_code" not in flow:
                raise RuntimeError(f"Could not start device-code flow: {json.dumps(flow, indent=2)}")

            print(flow["message"])
            result = self.app.acquire_token_by_device_flow(flow)

        self._persist_cache()

        if "access_token" not in result:
            error = result.get("error", "unknown_error")
            description = result.get("error_description", "No error description returned.")
            raise RuntimeError(f"Microsoft authentication failed: {error}: {description}")

        return result["access_token"]

    def _search_expression(self) -> str:
        return " OR ".join(f"from:{sender}" for sender in self.senders)

    def _request_messages(self, top: int = 50) -> list[dict]:
        token = self._get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        params = {
            "$search": f'"{self._search_expression()}"',
            "$top": str(top),
            "$select": "id,subject,from,receivedDateTime,body,webLink,hasAttachments",
        }

        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.get(f"{GRAPH_BASE_URL}/me/messages", headers=headers, params=params)
            response.raise_for_status()
            payload = response.json()

        return payload.get("value", [])

    @staticmethod
    def _to_email_message(item: dict) -> EmailMessage:
        sender = item.get("from", {}).get("emailAddress", {})
        body = item.get("body", {})
        received_raw = item.get("receivedDateTime")
        received_at = None
        if received_raw:
            received_at = datetime.fromisoformat(received_raw.replace("Z", "+00:00"))

        body_content = body.get("content") or ""
        content_type = (body.get("contentType") or "").lower()

        return EmailMessage(
            message_id=item["id"],
            sender_name=sender.get("name"),
            sender_email=(sender.get("address") or "").lower() or None,
            subject=item.get("subject") or "",
            received_at=received_at,
            body_text=body_content if content_type == "text" else "",
            body_html=body_content if content_type == "html" else "",
            links=[item["webLink"]] if item.get("webLink") else [],
            attachments=["present"] if item.get("hasAttachments") else [],
        )

    def get_messages(self, top: int = 50) -> list[EmailMessage]:
        raw_messages = self._request_messages(top=top)
        allowed = set(self.senders)

        messages: list[EmailMessage] = []
        for item in raw_messages:
            message = self._to_email_message(item)
            if message.sender_email in allowed:
                messages.append(message)

        return messages
