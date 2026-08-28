from __future__ import annotations

import re
from html import unescape
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from career_agent.models.email import EmailMessage

URL_PATTERN = re.compile(r"https?://[^\s<>\"']+")
TRAILING_URL_PUNCTUATION = ".,;:!?)]}"


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)

    return result


def html_to_text(html: str) -> str:
    """Convert HTML email content into readable plain text."""
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text("\n", strip=True)
    lines = [line.strip() for line in unescape(text).splitlines() if line.strip()]
    return "\n".join(lines)


def extract_links_from_html(html: str, base_url: str | None = None) -> list[str]:
    """Extract href URLs from HTML while preserving their original order."""
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []

    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "").strip()
        if not href:
            continue

        if base_url:
            href = urljoin(base_url, href)

        links.append(href)

    return _dedupe_keep_order(links)


def extract_links_from_text(text: str) -> list[str]:
    """Extract HTTP(S) URLs written directly in plain-text email bodies."""
    if not text:
        return []

    links: list[str] = []
    for match in URL_PATTERN.findall(text):
        cleaned = match.rstrip(TRAILING_URL_PUNCTUATION)
        if cleaned:
            links.append(cleaned)

    return _dedupe_keep_order(links)


def normalize_email(message: EmailMessage) -> EmailMessage:
    """Normalize body, links, and in-memory attachment text deterministically."""
    body_text = (message.body_text or "").strip()

    if not body_text and message.body_html:
        body_text = html_to_text(message.body_html)

    attachment_text = (message.attachment_text or "").strip()
    if attachment_text:
        body_text = (
            f"{body_text}\n\n{attachment_text}".strip()
            if body_text
            else attachment_text
        )

    html_links = extract_links_from_html(message.body_html)
    text_links = extract_links_from_text(body_text)
    links = _dedupe_keep_order([*message.links, *html_links, *text_links])

    return message.model_copy(
        update={
            "body_text": body_text,
            "links": links,
        }
    )
