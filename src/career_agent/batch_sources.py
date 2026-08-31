from __future__ import annotations

from io import BytesIO
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

from career_agent.models.email import EmailMessage
from career_agent.models.job_record import SourceDocument
from career_agent.nodes.normalize_email import (
    extract_links_from_html,
    extract_links_from_text,
)

MAX_LINKED_PDFS = 8
MAX_PDF_BYTES = 12 * 1024 * 1024
MAX_PDF_TEXT_CHARS = 120_000


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _is_http(url: str) -> bool:
    try:
        return urlparse(url).scheme.lower() in {"http", "https"}
    except ValueError:
        return False


def _looks_like_pdf(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return parsed.path.lower().endswith(".pdf")


def _html_to_text_with_links(html: str) -> str:
    """Convert HTML to readable text without throwing away anchor destinations."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "").strip()
        label = " ".join(anchor.get_text(" ", strip=True).split())
        if not href:
            continue
        replacement = f"{label} <{href}>" if label else f"<{href}>"
        anchor.replace_with(replacement)

    return "\n".join(
        line.strip()
        for line in soup.get_text("\n").splitlines()
        if line.strip()
    )


def _pdf_text(raw: bytes) -> str:
    reader = PdfReader(BytesIO(raw))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    return text[:MAX_PDF_TEXT_CHARS].strip()


def _fetch_linked_pdf(url: str, timeout_seconds: float = 15.0) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SimplyNextCareerAgent/0.1)"
    }
    with httpx.Client(follow_redirects=True, timeout=timeout_seconds, headers=headers) as client:
        response = client.get(url)
        response.raise_for_status()
        raw = response.content
        if len(raw) > MAX_PDF_BYTES:
            raise ValueError(f"linked PDF exceeds {MAX_PDF_BYTES} bytes")
        content_type = response.headers.get("content-type", "").lower()
        if "pdf" not in content_type and not raw.startswith(b"%PDF"):
            raise ValueError("linked resource is not a PDF")
    return _pdf_text(raw)


def build_source_corpus(
    email: EmailMessage,
    *,
    fetch_linked_pdfs: bool = True,
) -> tuple[str, list[str], list[SourceDocument], list[str]]:
    """Build one extraction corpus from every useful representation of an email.

    The richest email representation is used once, avoiding duplicate token spend
    on nearly identical plain/HTML bodies. Public PDF links are appended as extra
    source documents when available.
    """
    warnings: list[str] = []
    blocks: list[str] = []
    documents: list[SourceDocument] = []

    plain = (email.body_text or "").strip()
    html_text = _html_to_text_with_links(email.body_html or "").strip()
    attachment_text = (email.attachment_text or "").strip()

    # Forwarded HTML often contains everything in the recovered plain payload plus
    # richer tables and href targets. Prefer it when it is comparably large.
    if html_text and len(html_text) >= max(300, int(len(plain) * 0.8)):
        email_text = html_text
        label = "full email html"
    else:
        email_text = plain or html_text
        label = "recovered email text" if plain else "full email html"

    if email_text:
        blocks.append(f"SOURCE: EMAIL\n{email_text}")
        documents.append(
            SourceDocument(label=label, source_type="email", text_chars=len(email_text))
        )

    if attachment_text:
        blocks.append(f"SOURCE: EMAIL ATTACHMENTS\n{attachment_text}")
        documents.append(
            SourceDocument(label="email attachments", source_type="attachment", text_chars=len(attachment_text))
        )

    links = _dedupe(
        [
            *email.links,
            *extract_links_from_html(email.body_html or ""),
            *extract_links_from_text(plain),
            *extract_links_from_text(html_text),
        ]
    )

    if fetch_linked_pdfs:
        pdf_urls = [url for url in links if _is_http(url) and _looks_like_pdf(url)][:MAX_LINKED_PDFS]
        for url in pdf_urls:
            try:
                text = _fetch_linked_pdf(url)
            except Exception as exc:
                warnings.append(
                    f"linked PDF unavailable: {url}: {type(exc).__name__}: {exc}"
                )
                continue
            if not text:
                warnings.append(f"linked PDF contained no extractable text: {url}")
                continue
            blocks.append(f"SOURCE: LINKED PDF\nURL: {url}\n{text}")
            documents.append(
                SourceDocument(
                    label=urlparse(url).path.rsplit("/", 1)[-1] or "linked PDF",
                    source_type="linked_pdf",
                    url=url,
                    text_chars=len(text),
                )
            )

    return "\n\n================ SOURCE DOCUMENT ================\n\n".join(blocks), links, documents, warnings
