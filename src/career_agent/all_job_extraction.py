from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import unquote

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
LLM_TIMEOUT_SECONDS = 15.0
LLM_MAX_RETRIES = 1

SECTION_JOBS = "jobs"
SECTION_INTERNSHIPS = "internships"
SECTION_EVENTS = "events"
URL_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


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
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
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
        r"(?i)deadline\s*:\s*(\d{1,2})(?:st|nd|rd|th)?\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(20\d{2})",
        r"(?i)deadline\s*:\s*(\d{1,2})(?:st|nd|rd|th)?\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{2})",
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


def _table_signals(
    *,
    source_name: str,
    source_message_id: str,
    source_date,
    corpus: str,
) -> list[OpportunitySignal]:
    signals: list[OpportunitySignal] = []
    section = ""
    in_table = False
    last_company: str | None = None

    for raw_line in corpus.splitlines():
        line = raw_line.strip()
        normalized_section = re.sub(r"[^A-Z]", "", line.upper())
        if normalized_section == "JOBS":
            section = SECTION_JOBS
            last_company = None
            continue
        if normalized_section == "INTERNSHIPS":
            section = SECTION_INTERNSHIPS
            last_company = None
            continue
        if normalized_section == "EVENTS":
            section = SECTION_EVENTS
            last_company = None
            continue
        if line == TABLE_START:
            in_table = True
            last_company = None
            continue
        if line == TABLE_END:
            in_table = False
            last_company = None
            continue
        if not in_table or section not in {SECTION_JOBS, SECTION_INTERNSHIPS}:
            continue

        cells = [_clean_cell(cell) for cell in raw_line.split("|")]
        if len(cells) < 3:
            continue
        joined = " | ".join(cells).lower()
        if "company" in joined and ("role" in joined or "tc id" in joined):
            continue

        if len(cells) >= 5:
            company = _clean_company_cell(cells[1])
            role = _clean_cell(cells[2])
            remarks = " | ".join(cells[4:])
            if company:
                last_company = company
        elif len(cells) == 3 and last_company:
            # HTML rowspan continuation: company/industry cells exist only on the
            # first physical row, while following rows carry role | TC ID | remarks.
            company = last_company
            role = _clean_cell(cells[0])
            remarks = cells[2]
        elif len(cells) >= 4:
            company = _clean_company_cell(cells[0])
            role = _clean_cell(cells[1])
            remarks = " | ".join(cells[3:])
            if company:
                last_company = company
        else:
            continue

        if not company or not role:
            continue

        lowered_role = role.lower()
        lowered_remarks = remarks.lower()
        if "intern" in lowered_role or "internship" in lowered_remarks:
            opportunity_type = "internship"
        elif "full time job" in lowered_remarks or "full-time job" in lowered_remarks:
            opportunity_type = "full_time"
        else:
            opportunity_type = "internship" if section == SECTION_INTERNSHIPS else "full_time"

        signals.append(
            OpportunitySignal(
                source_type="outlook",
                source_name=source_name,
                source_message_id=source_message_id,
                source_date=source_date,
                company=company,
                role_title=role,
                location=_location_from_remarks(remarks),
                opportunity_type=opportunity_type,
                deadline_hint=_parse_date_hint(remarks),
                urls=list(dict.fromkeys(extract_links_from_text(" | ".join(cells)))),
                raw_text=" | ".join(cells),
                resolution_status="unresolved",
            )
        )
    return signals


def _strip_table_blocks(corpus: str) -> str:
    """Remove structured HTML table blocks before optional LLM extraction."""
    kept: list[str] = []
    in_table = False
    for raw_line in corpus.splitlines():
        line = raw_line.strip()
        if line == TABLE_START:
            in_table = True
            continue
        if line == TABLE_END:
            in_table = False
            continue
        if not in_table:
            kept.append(raw_line)
    return "\n".join(kept).strip()


def _residual_needs_llm(text: str, *, table_signals_found: bool) -> bool:
    lines: list[str] = []
    ignored = {"jobs", "internships", "events"}
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split()).strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith("source:") or lowered in ignored:
            continue
        if line == SOURCE_DOCUMENT_SEPARATOR:
            continue
        lines.append(line)

    residual = " ".join(lines).strip()
    if not residual:
        return False
    if not table_signals_found:
        return True

    # When deterministic job tables already cover the email, only ask the LLM to
    # inspect residual prose that actually looks employment-related. Event/challenge
    # blurbs should not create a gratuitous LLM call.
    return bool(
        re.search(
            r"(?i)\b(job|jobs|role|roles|intern|internship|hiring|graduate|associate|engineer|analyst|developer|manager|programme|program)\b",
            residual,
        )
    )


def _item_value(item, key: str):
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _from_llm_item(
    item,
    *,
    source_name: str,
    source_message_id: str,
    source_date,
) -> OpportunitySignal | None:
    company = _clean_cell(str(_item_value(item, "company") or ""))
    role_title = _clean_cell(str(_item_value(item, "role_title") or ""))
    if not company or not role_title:
        return None
    opportunity_type = _item_value(item, "opportunity_type") or "unknown"
    if opportunity_type not in {"internship", "full_time", "unknown"}:
        opportunity_type = "unknown"
    urls = _item_value(item, "urls") or []
    evidence = _item_value(item, "evidence_text") or f"{company} | {role_title}"
    deadline_hint = _parse_date_hint(str(evidence))
    location = _item_value(item, "location") or None
    target_major = list(_item_value(item, "target_major") or [])
    target_degree_level = list(_item_value(item, "target_degree_level") or [])
    return OpportunitySignal(
        source_type="outlook",
        source_name=source_name,
        source_message_id=source_message_id,
        source_date=source_date,
        company=company,
        role_title=role_title,
        location=location,
        opportunity_type=opportunity_type,
        deadline_hint=deadline_hint,
        target_major=target_major,
        target_degree_level=target_degree_level,
        urls=list(dict.fromkeys(urls)),
        raw_text=str(evidence),
        resolution_status="unresolved",
    )


def _signal_key(signal: OpportunitySignal) -> tuple[str, str]:
    return _normalize(signal.company), _normalize(signal.role_title)


def _dedupe_signals(signals: list[OpportunitySignal]) -> list[OpportunitySignal]:
    merged: dict[tuple[str, str], OpportunitySignal] = {}
    for signal in signals:
        key = _signal_key(signal)
        previous = merged.get(key)
        if previous is None:
            merged[key] = signal
            continue
        merged[key] = previous.model_copy(
            update={
                "deadline_hint": previous.deadline_hint or signal.deadline_hint,
                "location": previous.location or signal.location,
                "target_major": list(dict.fromkeys([*previous.target_major, *signal.target_major])),
                "target_degree_level": list(
                    dict.fromkeys([*previous.target_degree_level, *signal.target_degree_level])
                ),
                "urls": list(dict.fromkeys([*previous.urls, *signal.urls])),
                "raw_text": previous.raw_text or signal.raw_text,
            }
        )
    return list(merged.values())


def _reattach_direct_urls(signals: list[OpportunitySignal], corpus: str) -> list[OpportunitySignal]:
    """Attach each loose direct URL only to its strongest company+role match."""
    all_urls = list(dict.fromkeys(extract_links_from_text(corpus)))
    if not all_urls or not signals:
        return signals

    assignments: dict[int, list[str]] = {index: [] for index in range(len(signals))}
    for url in all_urls:
        normalized_url = _normalize(unquote(url))
        url_tokens = set(URL_TOKEN_PATTERN.findall(normalized_url))
        scored: list[tuple[float, int]] = []

        for index, signal in enumerate(signals):
            if signal.urls:
                continue
            role_tokens = set(URL_TOKEN_PATTERN.findall(_normalize(signal.role_title)))
            company_tokens = set(URL_TOKEN_PATTERN.findall(_normalize(signal.company)))
            if not role_tokens or not company_tokens:
                continue
            if not (company_tokens & url_tokens):
                continue
            title_overlap = len(role_tokens & url_tokens) / max(1, len(role_tokens))
            if title_overlap >= 0.5:
                scored.append((title_overlap, index))

        if not scored:
            continue
        scored.sort(reverse=True)
        best_score, best_index = scored[0]
        # If two different roles match an URL almost equally, leave it unattached
        # rather than contaminating both records.
        if len(scored) > 1 and abs(best_score - scored[1][0]) < 0.05:
            continue
        assignments[best_index].append(url)

    updated: list[OpportunitySignal] = []
    for index, signal in enumerate(signals):
        if signal.urls or not assignments[index]:
            updated.append(signal)
            continue
        updated.append(
            signal.model_copy(update={"urls": list(dict.fromkeys(assignments[index]))})
        )
    return updated


def extract_all_opportunities(
    *,
    source_name: str,
    source_message_id: str,
    source_date,
    corpus: str,
) -> tuple[list[OpportunitySignal], ExtractionMetrics, list[str]]:
    table_signals = _table_signals(
        source_name=source_name,
        source_message_id=source_message_id,
        source_date=source_date,
        corpus=corpus,
    )

    llm_corpus = _strip_table_blocks(corpus) if table_signals else corpus
    llm_items: list = []
    llm_calls = 0
    errors: list[str] = []
    if _residual_needs_llm(llm_corpus, table_signals_found=bool(table_signals)):
        for chunk in _chunks(llm_corpus):
            chunk_items, chunk_calls, chunk_errors = _invoke_with_adaptive_retry(chunk)
            llm_calls += chunk_calls
            llm_items.extend(chunk_items)
            errors.extend(chunk_errors)

    llm_signals = [
        signal
        for item in llm_items
        if (
            signal := _from_llm_item(
                item,
                source_name=source_name,
                source_message_id=source_message_id,
                source_date=source_date,
            )
        )
    ]
    signals = _dedupe_signals([*table_signals, *llm_signals])
    signals = _reattach_direct_urls(signals, corpus)
    return signals, ExtractionMetrics(llm_calls=llm_calls, source_chars=len(corpus)), errors
