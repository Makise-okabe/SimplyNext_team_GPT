from __future__ import annotations

import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from career_agent.models.signal import ExtractedOpportunityBatch, OpportunitySignal

CHUNK_CHARS = 6500
CHUNK_OVERLAP = 900
MAX_LLM_CHUNKS = 24


@dataclass(frozen=True)
class ExtractionMetrics:
    llm_calls: int = 0
    source_chars: int = 0


def _normalize(value: str | None) -> str:
    return " ".join((value or "").lower().split())


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
- generic career advice/articles
- career fairs/workshops/seminars unless an actual named job/programme is stated
- company names without a role/programme title
- social links or generic home pages by themselves

Grounding rules:
- Never invent company, title, location, deadline, degree, major, or URL.
- Preserve the role/programme title as written, including cohort/year when present.
- If multiple roles are listed for one company, return one object PER ROLE.
- If one row has multiple explicit role titles, split them into separate opportunities.
- Keep only URLs that belong to that exact role/programme when the association is clear.
- opportunity_type: internship, full_time, or unknown.
- evidence_text: concise source text that identifies the company + role.
- It is okay for the same opportunity to appear again in overlapping chunks; deterministic deduplication happens later.

SOURCE CHUNK:
{chunk}
""".strip()
    return _build_llm().invoke(prompt)


def _key(item) -> tuple[str, str]:
    company = _normalize(item.company)
    title = _normalize(item.role_title)
    title = re.sub(r"\s+", " ", title)
    return company, title


def extract_all_opportunities(
    *,
    source_name: str,
    source_message_id: str,
    source_date,
    corpus: str,
) -> tuple[list[OpportunitySignal], ExtractionMetrics, list[str]]:
    """Chunk one rich source corpus, extract all roles, then deduplicate."""
    errors: list[str] = []
    extracted = []
    chunks = _chunks(corpus)

    for index, chunk in enumerate(chunks, start=1):
        try:
            batch = _invoke(chunk)
            extracted.extend(batch.opportunities)
        except Exception as exc:
            errors.append(
                f"all-job extraction chunk {index}/{len(chunks)} failed: "
                f"{type(exc).__name__}: {exc}"
            )

    merged: dict[tuple[str, str], OpportunitySignal] = {}
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

        existing = merged.get(key)
        if existing is None:
            merged[key] = signal
            continue

        merged[key] = existing.model_copy(
            update={
                "urls": list(dict.fromkeys([*existing.urls, *signal.urls])),
                "location": existing.location or signal.location,
                "opportunity_type": (
                    existing.opportunity_type
                    if existing.opportunity_type != "unknown"
                    else signal.opportunity_type
                ),
                "target_major": list(dict.fromkeys([*existing.target_major, *signal.target_major])),
                "target_degree_level": list(
                    dict.fromkeys([*existing.target_degree_level, *signal.target_degree_level])
                ),
                "raw_text": existing.raw_text or signal.raw_text,
            }
        )

    return (
        list(merged.values()),
        ExtractionMetrics(llm_calls=len(chunks), source_chars=len(corpus)),
        errors,
    )
