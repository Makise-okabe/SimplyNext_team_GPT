from __future__ import annotations

from io import BytesIO
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup, NavigableString
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
TABLE_START = "[[SIMPLYNEXT_TABLE_START]]"
TABLE_END = "[[SIMPLYNEXT_TABLE_END]]"
SOURCE_DOCUMENT_SEPARATOR = "================ SOURCE DOCUMENT ================"
PDF_PAGE_START = "[[SIMPLYNEXT_PDF_PAGE_START:"
PDF_PAGE_LINKS = "[[SIMPLYNEXT_PDF_PAGE_LINKS]]"
PDF_PAGE_END = "[[SIMPLYNEXT_PDF_PAGE_END]]"


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


def _fragment_text_with_links(fragment) -> str:
    clone = BeautifulSoup(str(fragment), "html.parser")
    for tag in clone(["script", "style", "noscript", "svg"]):
        tag.decompose()

    # Outlook often encodes multiple roles/TC IDs as HTML ordered lists. Plain
    # BeautifulSoup get_text() drops the visible numbering, which makes one table
    # row impossible to deterministically expand back into one role per record.
    for ordered in clone.find_all("ol"):
        for index, item in enumerate(ordered.find_all("li", recursive=False), start=1):
            item.insert(0, NavigableString(f"{index}. "))

    for anchor in clone.find_all("a", href=True):
        href = anchor.get("href", "").strip()
        label = " ".join(anchor.get_text(" ", strip=True).split())
        if href:
            anchor.replace_with(f"{label} <{href}>" if label else f"<{href}>")
    return " ".join(clone.get_text(" ", strip=True).split())


def _rows_belonging_to_table(table) -> list:
    rows = []
    for row in table.find_all("tr"):
        if row.find_parent("table") is table:
            rows.append(row)
    return rows


def _html_to_text_with_links(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    for table in reversed(soup.find_all("table")):
        rows: list[str] = []
        for row in _rows_belonging_to_table(table):
            cells = row.find_all(["th", "td"], recursive=False)
            if not cells:
                cells = [
                    cell
                    for cell in row.find_all(["th", "td"])
                    if cell.find_parent("tr") is row
                ]
            if not cells:
                continue
            values = [_fragment_text_with_links(cell) for cell in cells]
            if any(values):
                rows.append(" | ".join(values))
        if rows:
            table.replace_with(
                NavigableString(
                    "\n" + TABLE_START + "\n" + "\n".join(rows) + "\n" + TABLE_END + "\n"
                )
            )

    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "").strip()
        label = " ".join(anchor.get_text(" ", strip=True).split())
        if href:
            anchor.replace_with(f"{label} <{href}>" if label else f"<{href}>")

    return "\n".join(
        line.strip() for line in soup.get_text("\n").splitlines() if line.strip()
    )


def _page_uri_links(page) -> list[str]:
    urls: list[str] = []
    for annotation_ref in page.get("/Annots") or []:
        try:
            annotation = annotation_ref.get_object()
            action = annotation.get("/A")
            if action is None:
                continue
            if hasattr(action, "get_object"):
                action = action.get_object()
            uri = action.get("/URI") if hasattr(action, "get") else None
            if uri:
                value = str(uri).strip()
                if _is_http(value):
                    urls.append(value)
        except Exception:
            continue
    return _dedupe(urls)


def _extract_pdf_page_text(page) -> str:
    """Prefer layout-aware extraction; fall back to normal pypdf text."""
    try:
        text = page.extract_text(extraction_mode="layout") or ""
    except (TypeError, ValueError, NotImplementedError):
        text = page.extract_text() or ""
    return text.strip()


def _pdf_pages_and_links(raw: bytes) -> tuple[list[str], list[str]]:
    """Read the whole PDF page by page and retain any embedded URLs as evidence."""
    reader = PdfReader(BytesIO(raw))
    pages: list[str] = []
    all_links: list[str] = []
    text_budget = MAX_PDF_TEXT_CHARS

    for page_number, page in enumerate(reader.pages, start=1):
        visible = _extract_pdf_page_text(page)
        page_links = _page_uri_links(page)
        all_links.extend(page_links)

        if text_budget <= 0 and not page_links:
            continue
        visible = visible[: max(0, text_budget)]
        text_budget -= len(visible)

        lines = [f"{PDF_PAGE_START}{page_number}]]", visible]
        if page_links:
            lines.append(PDF_PAGE_LINKS)
            lines.extend(f"<{url}>" for url in page_links)
        lines.append(PDF_PAGE_END)
        pages.append("\n".join(line for line in lines if line))

    return pages, _dedupe(all_links)


def _fetch_linked_pdf(
    url: str,
    timeout_seconds: float = 15.0,
) -> tuple[list[str], list[str]]:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SimplyNextCareerAgent/0.1)"}
    with httpx.Client(follow_redirects=True, timeout=timeout_seconds, headers=headers) as client:
        response = client.get(url)
        response.raise_for_status()
        raw = response.content
        if len(raw) > MAX_PDF_BYTES:
            raise ValueError(f"linked PDF exceeds {MAX_PDF_BYTES} bytes")
        content_type = response.headers.get("content-type", "").lower()
        if "pdf" not in content_type and not raw.startswith(b"%PDF"):
            raise ValueError("linked resource is not a PDF")
    return _pdf_pages_and_links(raw)


def build_source_corpus(
    email: EmailMessage,
    *,
    fetch_linked_pdfs: bool = True,
) -> tuple[str, list[str], list[SourceDocument], list[str]]:
    """Read the email and its attachments/linked PDFs. Research happens later."""
    warnings: list[str] = []
    blocks: list[str] = []
    documents: list[SourceDocument] = []

    plain = (email.body_text or "").strip()
    html_text = _html_to_text_with_links(email.body_html or "").strip()
    attachment_text = (email.attachment_text or "").strip()

    if html_text and len(html_text) > len(plain):
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
            SourceDocument(
                label="email attachments",
                source_type="attachment",
                text_chars=len(attachment_text),
            )
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
        for pdf_url in pdf_urls:
            try:
                pages, embedded_links = _fetch_linked_pdf(pdf_url)
            except Exception as exc:
                warnings.append(
                    f"linked PDF unavailable: {pdf_url}: {type(exc).__name__}: {exc}"
                )
                continue
            if not pages:
                warnings.append(f"linked PDF contained no extractable text: {pdf_url}")
                continue

            links = _dedupe([*links, *embedded_links])
            for page_number, page_text in enumerate(pages, start=1):
                blocks.append(
                    "SOURCE: LINKED PDF PAGE\n"
                    f"PDF URL: {pdf_url}\n"
                    f"PAGE: {page_number}\n"
                    f"{page_text}"
                )

            documents.append(
                SourceDocument(
                    label=urlparse(pdf_url).path.rsplit("/", 1)[-1] or "linked PDF",
                    source_type="linked_pdf",
                    url=pdf_url,
                    text_chars=sum(len(page) for page in pages),
                )
            )

    return f"\n\n{SOURCE_DOCUMENT_SEPARATOR}\n\n".join(blocks), links, documents, warnings
