from __future__ import annotations

from io import BytesIO
from urllib.parse import urlparse

import httpx
from pypdf import PdfReader

from career_agent.models.email import EmailMessage
from career_agent.models.job_record import SourceDocument
from career_agent.nodes.normalize_email import (
    extract_links_from_html,
    extract_links_from_text,
    html_to_text,
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

    Forwarded mail can expose a short recovered plain-text payload while still
    retaining a much richer HTML body. We deliberately consider both. PDF
    attachments parsed by the Graph connector are included, as are public PDF
    links such as NUS CFG eNews booklets.
    """
    warnings: list[str] = []
    blocks: list[str] = []
    documents: list[SourceDocument] = []

    plain = (email.body_text or "").strip()
    html_text = html_to_text(email.body_html or "").strip()
    attachment_text = (email.attachment_text or "").strip()

    if plain:
        blocks.append(f"SOURCE: RECOVERED EMAIL TEXT\n{plain}")
        documents.append(
            SourceDocument(label="recovered email text", source_type="email", text_chars=len(plain))
        )

    if html_text and html_text != plain:
        blocks.append(f"SOURCE: FULL EMAIL HTML AS TEXT\n{html_text}")
        documents.append(
            SourceDocument(label="full email html", source_type="email", text_chars=len(html_text))
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
