from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from career_agent.batch_sources import (
    SOURCE_DOCUMENT_SEPARATOR,
    TABLE_END,
    TABLE_START,
)
from career_agent.models.signal import ExtractedOpportunityBatch, OpportunitySignal
from career_agent.nodes.normalize_email import extract_links_from_text

CHUNK_CHARS = 6500
CHUNK_OVERLAP = 900
MAX_LLM_CHUNKS = 24
MAX_RETRY_SPLIT_DEPTH = 2
MIN_RETRY_CHARS = 1400

SECTION_JOBS = "jobs"
SECTION_INTERNSHIPS = "internships"
SECTION_EVENTS = "events"


@dataclass(frozen=True)
class ExtractionMetrics:
    llm_calls: int = 0
    source_chars: int = 0


def _normalize(value: str | None) -> str:
    return " ".join((value or "").lower().split())


def _clean_cell(value: str) -> str:
    value = value.replace("**", "").replace("__", "")
    value = re.sub(r"\s+", " ", value)
    return value.strip(" |\t\r\n")


def _clean_company_cell(value: str) -> str:
    value = _clean_cell(value).strip("* _")
    match = re.search(r"(?i)\s+\*?\s*for\s+the\s+roles\s+in\b", value)
    if match:
        value = value[: match.start()].strip("* _")
    return value


def _chunks(text: str) -> list[str]:
    if not text:
        return []
    values: list[str] = []
    start = 0
    while start < len(text) and len(values) < MAX_LLM_CHUNKS:
        end = min(len(text), start + CHUNK_CHARS)
        values.append(text[start:end])
        if end >= len(text):
            break
        start = max(start + 1, end - CHUNK_OVERLAP)
    return values


def _split_retry_chunk(text: str) -> tuple[str, str] | None:
    if len(text) < MIN_RETRY_CHARS * 2:
        return None
    midpoint = len(text) // 2
    search_radius = min(900, midpoint - 1, len(text) - midpoint - 1)
    split_at = midpoint
    for offset in range(search_radius + 1):
        left = midpoint - offset
        right = midpoint + offset
        if left > 0 and text[left] == "\n":
            split_at = left
            break
        if right < len(text) and text[right] == "\n":
            split_at = right
            break
    left = text[:split_at].strip()
    right = text[split_at:].strip()
    if len(left) < MIN_RETRY_CHARS or len(right) < MIN_RETRY_CHARS:
        split_at = midpoint
        left = text[:split_at].strip()
        right = text[split_at:].strip()
    if not left or not right:
        return None
    return left, right


def _build_llm():
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is missing from .env")
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
    ).with_structured_output(ExtractedOpportunityBatch)


def _invoke(chunk: str) -> ExtractedOpportunityBatch:
    prompt = f"""
You exhaustively extract EMPLOYMENT OPPORTUNITIES from one chunk of an NUS career email/newsletter.

Return every concrete opportunity explicitly named in this chunk, including:
- full-time jobs and graduate jobs
- internships
- graduate/academy/management-associate programmes that lead to employment

Do NOT return:
- events, career fairs, talks, workshops, seminars, competitions, or challenges
- generic career advice/articles
- company names without a role/programme title
- social links or generic home pages by themselves

Grounding rules:
- Never invent company, title, location, deadline, degree, major, or URL.
- Preserve the role/programme title as written, including cohort/year when present.
- If multiple roles are listed for one company, return one object PER ROLE.
- Keep only URLs that belong to that exact role/programme when the association is clear.
- opportunity_type: internship, full_time, or unknown.
- evidence_text: concise source text that identifies the company + role.

SOURCE CHUNK:
{chunk}
""".strip()
    return _build_llm().invoke(prompt)


def _invoke_with_adaptive_retry(
    chunk: str,
    *,
    depth: int = 0,
) -> tuple[list, int, list[str]]:
    calls = 1
    try:
        batch = _invoke(chunk)
        return list(batch.opportunities), calls, []
    except Exception as exc:
        split = _split_retry_chunk(chunk) if depth < MAX_RETRY_SPLIT_DEPTH else None
        if split is None:
            return [], calls, [f"{type(exc).__name__}: {exc}"]
        opportunities: list = []
        terminal_errors: list[str] = []
        for child in split:
            child_items, child_calls, child_errors = _invoke_with_adaptive_retry(
                child,
                depth=depth + 1,
            )
            calls += child_calls
            opportunities.extend(child_items)
            terminal_errors.extend(child_errors)
        return opportunities, calls, terminal_errors


def _parse_date_hint(text: str):
    patterns = (
        r"(?i)deadline\s*:\s*(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(20\d{2})",
        r"(?i)deadline\s*:\s*(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{2})",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        day, month, year = match.groups()
        if len(year) == 2:
            year = "20" + year
        try:
            return datetime.strptime(f"{day} {month} {year}", "%d %b %Y").date()
        except ValueError:
            continue
    return None


def _location_from_remarks(text: str) -> str | None:
    match = re.search(
        r"(?i)Location\s*:\s*([^|\n]+?)(?=(?:Deadline|UG|PG|Note|$))",
        text,
    )
    if not match:
        return None
    value = _clean_cell(match.group(1))
    return value or None


def _looks_like_identifier(value: str) -> bool:
    cleaned = re.sub(r"(?i)^TC\s*:\s*", "", _clean_cell(value))
    cleaned = cleaned.replace(" ", "")
    return bool(re.fullmatch(r"(?:\d{5,8}|NA|N/A)", cleaned, flags=re.IGNORECASE))


def _row_fields(
    cells: list[str],
    *,
    carried_company: str | None,
) -> tuple[str | None, str | None, str] | None:
    cells = [_clean_cell(cell) for cell in cells if _clean_cell(cell)]
    if not cells:
        return None
    joined = " | ".join(cells)
    lowered = joined.lower()
    if "company" in lowered and ("role" in lowered or "event title" in lowered):
        return None
    if len(cells) >= 5 and _looks_like_identifier(cells[3]):
        return _clean_company_cell(cells[1]), cells[2], " | ".join(cells[4:])
    if len(cells) >= 4 and _looks_like_identifier(cells[2]):
        return _clean_company_cell(cells[0]), cells[1], " | ".join(cells[3:])
    if len(cells) >= 3 and _looks_like_identifier(cells[1]) and carried_company:
        return carried_company, cells[0], " | ".join(cells[2:])
    if len(cells) >= 4 and _looks_like_identifier(cells[2]) and carried_company:
        return carried_company, cells[1], " | ".join(cells[3:])
    return None


def _table_row_signal(
    *,
    source_name: str,
    source_message_id: str,
    source_date,
    section: str,
    cells: list[str],
    carried_company: str | None = None,
) -> tuple[OpportunitySignal | None, str | None]:
    resolved = _row_fields(cells, carried_company=carried_company)
    if resolved is None:
        return None, carried_company
    company, role, remarks = resolved
    if not company or not role:
        return None, carried_company
    company = _clean_company_cell(company)
    role = _clean_cell(role)
    if company.lower() in {"company", "-", "—"} or role.lower().startswith("role"):
        return None, carried_company
    urls = extract_links_from_text(role)
    role_without_urls = re.sub(r"<https?://[^>]+>", "", role).strip()
    role_without_urls = re.sub(r"\s+", " ", role_without_urls)
    if not role_without_urls:
        return None, company
    opportunity_type = "internship" if section == SECTION_INTERNSHIPS else "full_time"
    if "intern" in role_without_urls.lower() or "internship" in remarks.lower():
        opportunity_type = "internship"
    raw_cells = [_clean_cell(cell) for cell in cells]
    signal = OpportunitySignal(
        source_type="outlook",
        source_name=source_name,
        source_message_id=source_message_id,
        source_date=source_date,
        company=company,
        role_title=role_without_urls,
        location=_location_from_remarks(remarks),
        opportunity_type=opportunity_type,
        deadline_hint=_parse_date_hint(remarks),
        urls=urls,
        raw_text=" | ".join(raw_cells),
        resolution_status="unresolved",
    )
    return signal, company


def _split_source_documents(corpus: str) -> list[str]:
    return [part.strip() for part in corpus.split(SOURCE_DOCUMENT_SEPARATOR) if part.strip()]


def _extract_structured_email_document(
    *,
    source_name: str,
    source_message_id: str,
    source_date,
    document: str,
) -> tuple[list[OpportunitySignal], str, bool]:
    """Parse a single EMAIL document; structured job tables become authoritative."""
    lines = document.splitlines()
    signals: list[OpportunitySignal] = []
    residual: list[str] = []
    in_table = False
    section = ""
    carried_company: str | None = None
    saw_structured_job_table = False

    for raw_line in lines:
        line = raw_line.strip()
        upper = re.sub(r"[^A-Z]", "", line.upper())
        if upper == "JOBS":
            section = SECTION_JOBS
            carried_company = None
            continue
        if upper == "INTERNSHIPS":
            section = SECTION_INTERNSHIPS
            carried_company = None
            continue
        if upper == "EVENTS":
            section = SECTION_EVENTS
            carried_company = None
            continue
        if line == TABLE_START:
            in_table = True
            carried_company = None
            if section in {SECTION_JOBS, SECTION_INTERNSHIPS}:
                saw_structured_job_table = True
            continue
        if line == TABLE_END:
            in_table = False
            carried_company = None
            continue
        if in_table:
            if section not in {SECTION_JOBS, SECTION_INTERNSHIPS}:
                continue
            cells = [cell.strip() for cell in raw_line.split("|")]
            signal, carried_company = _table_row_signal(
                source_name=source_name,
                source_message_id=source_message_id,
                source_date=source_date,
                section=section,
                cells=cells,
                carried_company=carried_company,
            )
            if signal is not None:
                signals.append(signal)
            continue
        if section == SECTION_EVENTS:
            continue
        residual.append(raw_line)

    # Once a trusted email contains explicit JOBS/INTERNSHIPS tables, those rows
    # are the authoritative employment list. Suppress surrounding prose so guides,
    # challenges and marketing copy cannot cause flaky tool calls or false jobs.
    residual_text = "" if saw_structured_job_table else "\n".join(residual)
    return signals, residual_text, saw_structured_job_table


def _key(item) -> tuple[str, str]:
    company = _normalize(item.company)
    title = _normalize(item.role_title)
    return company, re.sub(r"\s+", " ", title)


def _merge_signal(
    merged: dict[tuple[str, str], OpportunitySignal],
    signal: OpportunitySignal,
) -> None:
    key = (_normalize(signal.company), _normalize(signal.role_title))
    if not all(key):
        return
    existing = merged.get(key)
    if existing is None:
        merged[key] = signal
        return
    merged[key] = existing.model_copy(
        update={
            "urls": list(dict.fromkeys([*existing.urls, *signal.urls])),
            "location": existing.location or signal.location,
            "opportunity_type": (
                existing.opportunity_type
                if existing.opportunity_type != "unknown"
                else signal.opportunity_type
            ),
            "deadline_hint": existing.deadline_hint or signal.deadline_hint,
            "target_major": list(dict.fromkeys([*existing.target_major, *signal.target_major])),
            "target_degree_level": list(
                dict.fromkeys([*existing.target_degree_level, *signal.target_degree_level])
            ),
            "raw_text": existing.raw_text or signal.raw_text,
        }
    )


def extract_all_opportunities(
    *,
    source_name: str,
    source_message_id: str,
    source_date,
    corpus: str,
) -> tuple[list[OpportunitySignal], ExtractionMetrics, list[str]]:
    """Structured email tables first; LLM only for independent non-table documents."""
    errors: list[str] = []
    merged: dict[tuple[str, str], OpportunitySignal] = {}
    llm_inputs: list[str] = []

    documents = _split_source_documents(corpus)
    for document in documents:
        if document.startswith("SOURCE: EMAIL\n"):
            table_signals, residual, _ = _extract_structured_email_document(
                source_name=source_name,
                source_message_id=source_message_id,
                source_date=source_date,
                document=document,
            )
            for signal in table_signals:
                _merge_signal(merged, signal)
            if residual.strip():
                llm_inputs.append(residual)
        else:
            # Attachments and linked PDFs remain eligible for LLM extraction even
            # when the email itself had authoritative structured tables.
            llm_inputs.append(document)

    llm_calls = 0
    extracted = []
    for llm_input in llm_inputs:
        for index, chunk in enumerate(_chunks(llm_input), start=1):
            if len(chunk.strip()) < 180:
                continue
            chunk_items, chunk_calls, chunk_errors = _invoke_with_adaptive_retry(chunk)
            llm_calls += chunk_calls
            extracted.extend(chunk_items)
            for error in chunk_errors:
                errors.append(
                    f"all-job non-table chunk {index} failed after adaptive retry: {error}"
                )

    for item in extracted:
        key = _key(item)
        if not all(key):
            continue
        signal = OpportunitySignal(
            source_type="outlook",
            source_name=source_name,
            source_message_id=source_message_id,
            source_date=source_date,
            company=item.company,
            role_title=item.role_title,
            location=item.location,
            opportunity_type=item.opportunity_type,
            target_major=item.target_major or [],
            target_degree_level=item.target_degree_level or [],
            urls=list(dict.fromkeys(item.urls or [])),
            raw_text=item.evidence_text or "",
            resolution_status="unresolved",
        )
        _merge_signal(merged, signal)

    return (
        list(merged.values()),
        ExtractionMetrics(llm_calls=llm_calls, source_chars=len(corpus)),
        errors,
    )
