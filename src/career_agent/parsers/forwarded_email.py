from __future__ import annotations

import re
from html import unescape

from bs4 import BeautifulSoup

from career_agent.models.email import EmailMessage

FORWARD_PREFIX = re.compile(r"^(?:fw|fwd)\s*:\s*", re.IGNORECASE)
FROM_HEADER = re.compile(
    r"(?im)^From:\s*(?P<name>[^\r\n<]*?)\s*<(?P<email>[^>\s]+@[^>\s]+)>[ \t]*$"
)
SUBJECT_HEADER = re.compile(r"(?im)^Subject:\s*(?P<subject>[^\r\n]+)")


def _html_to_text(html: str) -> str:
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()

    return "\n".join(
        line.strip()
        for line in unescape(soup.get_text("\n")).splitlines()
        if line.strip()
    )


def strip_forward_prefix(subject: str) -> str:
    cleaned = subject.strip()
    while True:
        updated = FORWARD_PREFIX.sub("", cleaned, count=1).strip()
        if updated == cleaned:
            return cleaned
        cleaned = updated


def _forwarded_payload(text: str, subject_match: re.Match[str] | None) -> str:
    if not text:
        return ""

    if subject_match:
        payload = text[subject_match.end() :]
    else:
        payload = text

    payload = payload.lstrip(" \t\r\n_-")
    return payload.strip()


def recover_forwarded_email(message: EmailMessage) -> EmailMessage:
    """Recover the original sender/subject from an Outlook-style forwarded email.

    The forwarding account remains available as ``transport_sender_*`` while
    ``sender_*`` becomes the actual career source used by the downstream filter.
    """
    if not FORWARD_PREFIX.match(message.subject or ""):
        return message

    text = (message.body_text or "").strip()
    if not text and message.body_html:
        text = _html_to_text(message.body_html)

    # Only inspect the forwarded header area. This avoids accidentally treating
    # a quoted "From:" much later in a long newsletter as the transport header.
    header_window = text[:2500]
    from_match = FROM_HEADER.search(header_window)
    subject_match = SUBJECT_HEADER.search(header_window)

    original_subject = (
        subject_match.group("subject").strip()
        if subject_match
        else strip_forward_prefix(message.subject)
    )

    if not from_match:
        return message.model_copy(update={"subject": original_subject})

    original_name = from_match.group("name").strip().strip('"') or None
    original_email = from_match.group("email").strip().lower()

    payload = _forwarded_payload(text, subject_match)
    update = {
        "transport_sender_name": message.transport_sender_name or message.sender_name,
        "transport_sender_email": message.transport_sender_email or message.sender_email,
        "sender_name": original_name,
        "sender_email": original_email,
        "subject": original_subject,
    }

    if payload:
        update["body_text"] = payload

    return message.model_copy(update=update)
