from __future__ import annotations

import re
from io import BytesIO

from pypdf import PdfReader


def normalize_attachment_text(text: str) -> str:
    """Clean common PDF extraction artifacts without changing meaning."""
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Join words split by layout hyphenation, e.g. "manu-\nfacturing".
    text = re.sub(r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract searchable text directly from PDF bytes held in memory."""
    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("Attachment does not look like a PDF.")

    reader = PdfReader(BytesIO(pdf_bytes))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return normalize_attachment_text("\n".join(pages))


def format_attachment_text(name: str, text: str) -> str:
    if not text.strip():
        return ""
    return f"ATTACHMENT: {name}\n{text.strip()}"
