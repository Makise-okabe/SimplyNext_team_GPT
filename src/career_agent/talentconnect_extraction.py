from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from career_agent.batch_sources import SOURCE_DOCUMENT_SEPARATOR
from career_agent.models.signal import ExtractedOpportunityBatch, OpportunitySignal

CHUNK_CHARS = 4200
CHUNK_OVERLAP = 500


@dataclass(frozen=True)
class ExtractionMetrics:
    llm_calls: int = 0
    source_chars: int = 0


def _chunks(text: str) -> list[str]:
    if not text:
        return []
    values: list[str] = []
    start = 0
    while start < len(text):
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
Read this NUS TalentConnect career-newsletter text as a practical job-search assistant.

Goal: DO NOT MISS EMPLOYMENT LEADS. Extract every company-specific employment item that a student could reasonably follow up on at the employer's official careers site.

Include:
- named full-time jobs
- named internships
- graduate programmes, academy programmes, management associate programmes
- company-specific hiring / career / internship / graduate opportunities even when the PDF does not state one exact job title

For a company-specific employment lead with no exact role title:
- set role_title to the shortest useful employment label supported by the source, such as "Career opportunities", "Graduate opportunities", "Internship opportunities", or "Hiring opportunities"
- this is a SEARCH SEED for later official-site research, so do not drop the company merely because the newsletter is brief

Exclude only things that are clearly NOT employment:
- competitions
- talks, webinars, workshops, fairs, networking events
- generic articles/news with no recruiting angle
- scholarships with no employment programme

Grounding:
- never invent a company
- preserve exact role/programme names when the source provides them
- if exact title is absent, use a generic employment seed label as described above
- use internship/full_time when obvious, otherwise unknown
- keep URLs only when the source clearly associates them with that company/item
- evidence_text should contain the source wording that caused you to include the item

SOURCE:
{chunk}
""".strip()
    return _build_llm().invoke(prompt)


def _normalize(value: str | None) -> str:
    return " ".join((value or "").lower().split())


def extract_talentconnect_opportunities(
    *,
    source_name: str,
    source_message_id: str,
    source_date,
    corpus: str,
) -> tuple[list[OpportunitySignal], ExtractionMetrics, list[str]]:
    """Simple MVP: read every email/PDF document and keep broad employment seeds."""
    merged: dict[tuple[str, str], OpportunitySignal] = {}
    errors: list[str] = []
    calls = 0

    documents = [
        part.strip()
        for part in corpus.split(SOURCE_DOCUMENT_SEPARATOR)
        if part.strip()
    ]

    for document_index, document in enumerate(documents, start=1):
        for chunk_index, chunk in enumerate(_chunks(document), start=1):
            if len(chunk.strip()) < 120:
                continue
            calls += 1
            try:
                batch = _invoke(chunk)
            except Exception as exc:
                errors.append(
                    f"TalentConnect document {document_index} chunk {chunk_index} failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

            for item in batch.opportunities:
                company = (item.company or "").strip()
                title = (item.role_title or "").strip()
                if not company:
                    continue
                if not title:
                    title = "Career opportunities"

                key = (_normalize(company), _normalize(title))
                if not all(key):
                    continue

                signal = OpportunitySignal(
                    source_type="outlook",
                    source_name=source_name,
                    source_message_id=source_message_id,
                    source_date=source_date,
                    company=company,
                    role_title=title,
                    location=item.location,
                    opportunity_type=item.opportunity_type,
                    target_major=item.target_major or [],
                    target_degree_level=item.target_degree_level or [],
                    urls=list(dict.fromkeys(item.urls or [])),
                    raw_text=item.evidence_text or f"{company} | {title}",
                    resolution_status="unresolved",
                )

                existing = merged.get(key)
                if existing is None:
                    merged[key] = signal
                else:
                    merged[key] = existing.model_copy(
                        update={
                            "urls": list(dict.fromkeys([*existing.urls, *signal.urls])),
                            "location": existing.location or signal.location,
                            "opportunity_type": (
                                existing.opportunity_type
                                if existing.opportunity_type != "unknown"
                                else signal.opportunity_type
                            ),
                            "raw_text": existing.raw_text or signal.raw_text,
                        }
                    )

    return (
        list(merged.values()),
        ExtractionMetrics(llm_calls=calls, source_chars=len(corpus)),
        errors,
    )
