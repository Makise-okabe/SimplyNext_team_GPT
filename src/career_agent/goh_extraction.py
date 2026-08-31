from __future__ import annotations

import re
from datetime import datetime

from career_agent.all_job_extraction import ExtractionMetrics, extract_all_opportunities
from career_agent.batch_sources import SOURCE_DOCUMENT_SEPARATOR, TABLE_END, TABLE_START
from career_agent.models.signal import OpportunitySignal
from career_agent.nodes.normalize_email import extract_links_from_text

SECTION_JOBS = "jobs"
SECTION_INTERNSHIPS = "internships"
SECTION_EVENTS = "events"


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").replace("**", "").replace("__", "")).strip(" |")


def _parse_deadline(text: str, source_date=None):
    value = text or ""
    explicit = re.search(
        r"(?i)deadline\s*:\s*(\d{1,2})\s*(?:st|nd|rd|th)?\s*"
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*"
        r"(20\d{2}|\d{2})",
        value,
    )
    if explicit:
        day, month, year = explicit.groups()
        if len(year) == 2:
            year = "20" + year
        try:
            return datetime.strptime(f"{day} {month} {year}", "%d %b %Y").date()
        except ValueError:
            return None

    implicit = re.search(
        r"(?i)deadline\s*:\s*(\d{1,2})\s*(?:st|nd|rd|th)?\s*"
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b",
        value,
    )
    if not implicit or source_date is None:
        return None
    day, month = implicit.groups()
    try:
        source_year = source_date.year
        parsed = datetime.strptime(f"{day} {month} {source_year}", "%d %b %Y").date()
        if (parsed - source_date.date()).days < -120:
            parsed = parsed.replace(year=source_year + 1)
        return parsed
    except (ValueError, AttributeError):
        return None


def _location(text: str) -> str | None:
    match = re.search(
        r"(?i)Location\s*:\s*([^|]+?)(?=(?:Deadline|UG|PG|Note|$))",
        text or "",
    )
    return _clean(match.group(1)) if match else None


def _split_numbered_roles(text: str) -> list[str]:
    value = _clean(text)
    if not value:
        return []
    matches = list(re.finditer(r"(?:^|\s)(\d{1,2})\.\s*", value))
    if len(matches) < 2:
        return [value]
    roles: list[str] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(value)
        role = _clean(value[start:end])
        if role:
            roles.append(role)
    return roles or [value]


def _opportunity_type(section: str, role: str, remarks: str) -> str:
    lowered_role = role.lower()
    lowered_remarks = remarks.lower()
    if "intern" in lowered_role or "internship" in lowered_remarks:
        return "internship"
    if "full time job" in lowered_remarks or "full-time job" in lowered_remarks:
        return "full_time"
    return "internship" if section == SECTION_INTERNSHIPS else "full_time"


def _structured_signals(
    *,
    source_name: str,
    source_message_id: str,
    source_date,
    corpus: str,
) -> list[OpportunitySignal]:
    signals: list[OpportunitySignal] = []
    for document in corpus.split(SOURCE_DOCUMENT_SEPARATOR):
        if not document.strip().startswith("SOURCE: EMAIL\n"):
            continue
        section = ""
        in_table = False
        for raw_line in document.splitlines():
            line = raw_line.strip()
            upper = re.sub(r"[^A-Z]", "", line.upper())
            if upper == "JOBS":
                section = SECTION_JOBS
                continue
            if upper == "INTERNSHIPS":
                section = SECTION_INTERNSHIPS
                continue
            if upper == "EVENTS":
                section = SECTION_EVENTS
                continue
            if line == TABLE_START:
                in_table = True
                continue
            if line == TABLE_END:
                in_table = False
                continue
            if not in_table or section not in {SECTION_JOBS, SECTION_INTERNSHIPS}:
                continue

            cells = [_clean(cell) for cell in raw_line.split("|")]
            if len(cells) < 4:
                continue
            joined = " | ".join(cells).lower()
            if "company" in joined and ("role" in joined or "tc id" in joined):
                continue

            if len(cells) >= 5:
                company, role_cell, remarks = cells[1], cells[2], " | ".join(cells[4:])
            else:
                company, role_cell, remarks = cells[0], cells[1], " | ".join(cells[3:])
            company = _clean(company)
            if not company or not role_cell:
                continue

            urls = list(dict.fromkeys(extract_links_from_text(" | ".join(cells))))
            for role in _split_numbered_roles(role_cell):
                if not role or role.lower().startswith("role"):
                    continue
                signals.append(
                    OpportunitySignal(
                        source_type="outlook",
                        source_name=source_name,
                        source_message_id=source_message_id,
                        source_date=source_date,
                        company=company,
                        role_title=role,
                        location=_location(remarks),
                        opportunity_type=_opportunity_type(section, role, remarks),
                        deadline_hint=_parse_deadline(remarks, source_date),
                        urls=urls,
                        raw_text=" | ".join(cells),
                        resolution_status="unresolved",
                    )
                )
    return signals


def _key(signal: OpportunitySignal) -> tuple[str, str]:
    return (
        " ".join((signal.company or "").lower().split()),
        " ".join((signal.role_title or "").lower().split()),
    )


def extract_goh_opportunities(
    *,
    source_name: str,
    source_message_id: str,
    source_date,
    corpus: str,
    base_extractor=extract_all_opportunities,
) -> tuple[list[OpportunitySignal], ExtractionMetrics, list[str]]:
    """Keep the proven extractor, then deterministically repair/expand Goh tables."""
    base, metrics, errors = base_extractor(
        source_name=source_name,
        source_message_id=source_message_id,
        source_date=source_date,
        corpus=corpus,
    )
    robust = _structured_signals(
        source_name=source_name,
        source_message_id=source_message_id,
        source_date=source_date,
        corpus=corpus,
    )

    merged: dict[tuple[str, str], OpportunitySignal] = {}
    for signal in base:
        if len(_split_numbered_roles(signal.role_title or "")) > 1:
            continue
        merged[_key(signal)] = signal

    for signal in robust:
        key = _key(signal)
        previous = merged.get(key)
        if previous is None:
            merged[key] = signal
            continue
        merged[key] = previous.model_copy(
            update={
                "deadline_hint": signal.deadline_hint or previous.deadline_hint,
                "location": signal.location or previous.location,
                "opportunity_type": signal.opportunity_type,
                "urls": list(dict.fromkeys([*previous.urls, *signal.urls])),
                "raw_text": signal.raw_text or previous.raw_text,
            }
        )

    return list(merged.values()), metrics, errors
