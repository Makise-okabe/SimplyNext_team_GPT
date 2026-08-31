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
MAX_PDF_EMPLOYMENT_LINKS = 30
MAX_LINKED_JOB_PAGE_CHARS = 12_000
TABLE_START = "[[SIMPLYNEXT_TABLE_START]]"
TABLE_END = "[[SIMPLYNEXT_TABLE_END]]"
SOURCE_DOCUMENT_SEPARATOR = "================ SOURCE DOCUMENT ================"
PDF_PAGE_START = "[[SIMPLYNEXT_PDF_PAGE_START:"
PDF_PAGE_LINKS = "[[SIMPLYNEXT_PDF_PAGE_LINKS]]"
PDF_PAGE_END = "[[SIMPLYNEXT_PDF_PAGE_END]]"

ATS_HOST_MARKERS = (
    "myworkdayjobs.com",
    "workdayjobs.com",
    "greenhouse.io",
    "lever.co",
    "smartrecruiters.com",
    "successfactors.com",
    "taleo.net",
    "icims.com",
    "jobvite.com",
)
EMPLOYMENT_URL_MARKERS = (
    "/job",
    "jobs/",
    "career",
    "position",
    "requisition",
    "jobdetail",
    "jobcode=",
    "intern",
    "graduate",
    "opportunit",
    "apply",
)


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


def _looks_like_employment_link(url: str) -> bool:
    if not _is_http(url) or _looks_like_pdf(url):
        return False
    lowered = url.lower()
    try:
        host = urlparse(url).netloc.lower()
    except ValueError:
        host = ""
    return any(marker in host for marker in ATS_HOST_MARKERS) or any(
        marker in lowered for marker in EMPLOYMENT_URL_MARKERS
    )


def _fragment_text_with_links(fragment) -> str:
    clone = BeautifulSoup(str(fragment), "html.parser")
    for tag in clone(["script", "style", "noscript", "svg"]):
        tag.decompose()
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
    """Prefer pypdf's layout mode so newsletter columns stay in reading order."""
    try:
        text = page.extract_text(extraction_mode="layout") or ""
    except (TypeError, ValueError, NotImplementedError):
        text = page.extract_text() or ""
    return text.strip()


def _pdf_pages_and_links(raw: bytes) -> tuple[list[str], list[str]]:
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


def _fetch_linked_job_page(url: str, timeout_seconds: float = 12.0) -> str:
    """Fetch readable static text from a careers/job URL found inside a PDF."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SimplyNextCareerAgent/0.1)"}
    with httpx.Client(follow_redirects=True, timeout=timeout_seconds, headers=headers) as client:
        response = client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if "html" not in content_type and "text" not in content_type:
            return ""
        html = response.text

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    title = " ".join((soup.title.get_text(" ", strip=True) if soup.title else "").split())
    body = "\n".join(
        line.strip() for line in soup.get_text("\n").splitlines() if line.strip()
    )
    text = f"PAGE TITLE: {title}\n{body}" if title else body
    return text[:MAX_LINKED_JOB_PAGE_CHARS].strip()


def build_source_corpus(
    email: EmailMessage,
    *,
    fetch_linked_pdfs: bool = True,
) -> tuple[str, list[str], list[SourceDocument], list[str]]:
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
        documents.append(SourceDocument(label=label, source_type="email", text_chars=len(email_text)))

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

    crawled_job_urls: set[str] = set()
    if fetch_linked_pdfs:
        pdf_urls = [url for url in links if _is_http(url) and _looks_like_pdf(url)][:MAX_LINKED_PDFS]
        for url in pdf_urls:
            try:
                pages, embedded_links = _fetch_linked_pdf(url)
            except Exception as exc:
                warnings.append(f"linked PDF unavailable: {url}: {type(exc).__name__}: {exc}")
                continue
            if not pages:
                warnings.append(f"linked PDF contained no extractable text: {url}")
                continue

            embedded_links = _dedupe(embedded_links)
            links = _dedupe([*links, *embedded_links])
            for page_number, page_text in enumerate(pages, start=1):
                blocks.append(
                    f"SOURCE: LINKED PDF PAGE\nPDF URL: {url}\nPAGE: {page_number}\n{page_text}"
                )

                page_links = extract_links_from_text(page_text)
                for employment_url in page_links:
                    if len(crawled_job_urls) >= MAX_PDF_EMPLOYMENT_LINKS:
                        break
                    if employment_url in crawled_job_urls or not _looks_like_employment_link(employment_url):
                        continue
                    crawled_job_urls.add(employment_url)
                    try:
                        job_text = _fetch_linked_job_page(employment_url)
                    except Exception as exc:
                        warnings.append(
                            f"PDF employment link unavailable: {employment_url}: {type(exc).__name__}: {exc}"
                        )
                        continue
                    if not job_text:
                        continue
                    blocks.append(
                        "SOURCE: PDF-LINKED EMPLOYMENT PAGE\n"
                        f"FROM PDF: {url}\nFROM PDF PAGE: {page_number}\n"
                        f"URL: {employment_url}\n{job_text}"
                    )

            documents.append(
                SourceDocument(
                    label=urlparse(url).path.rsplit("/", 1)[-1] or "linked PDF",
                    source_type="linked_pdf",
                    url=url,
                    text_chars=sum(len(page) for page in pages),
                )
            )

    return f"\n\n{SOURCE_DOCUMENT_SEPARATOR}\n\n".join(blocks), links, documents, warnings
