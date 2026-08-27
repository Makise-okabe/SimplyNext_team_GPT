from __future__ import annotations

from html import unescape
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from career_agent.models.email import EmailMessage


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


def normalize_email(message: EmailMessage) -> EmailMessage:
    """Return an EmailMessage with normalized text and links.

    The function intentionally stays deterministic: no LLM is needed for basic
    HTML cleanup or URL extraction.
    """
    body_text = (message.body_text or "").strip()

    if not body_text and message.body_html:
        body_text = html_to_text(message.body_html)

    html_links = extract_links_from_html(message.body_html)
    links = _dedupe_keep_order([*message.links, *html_links])

    return message.model_copy(
        update={
            "body_text": body_text,
            "links": links,
        }
    )
