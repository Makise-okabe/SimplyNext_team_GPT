from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from career_agent.config import Settings
from career_agent.models.signal import (
    ExtractedOpportunityBatch,
    OpportunitySignal,
)

MAX_CANDIDATES_PER_LLM_CALL = 8
MAX_CONTEXT_CHARS = 900
MAX_DIRECT_TEXT_CHARS = 7000

ASSET_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".webp",
)

NOISE_HOSTS = {
    "outlook.office365.com",
    "www.facebook.com",
    "facebook.com",
    "www.instagram.com",
    "instagram.com",
    "www.youtube.com",
    "youtube.com",
    "www.linkedin.com",
    "linkedin.com",
}

GENERIC_NUS_PATHS = {
    "/cfg",
    "/cfg/",
    "/cfg/events",
    "/cfg/students/career-resources",
    "/cfg/students/jobs-internships/nus-career-plus",
}


@dataclass(frozen=True)
class CandidateChunk:
    url: str
    context: str


def should_extract_signal(state: dict) -> bool:
    email = state.get("email") or {}
    sender = (email.get("sender_email") or "").strip().lower()
    return sender in Settings().trusted_senders


def is_candidate_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False

    if parsed.scheme not in {"http", "https"}:
        return False

    host = parsed.netloc.lower()
    path = parsed.path.lower()

    if not host or host in NOISE_HOSTS:
        return False
    if path.endswith(ASSET_SUFFIXES):
        return False
    if "stripocdn.email" in host and "/images/" in path:
        return False
    if host.endswith("nus.edu.sg") and path in GENERIC_NUS_PATHS:
        return False

    return True


def _context_for_url(text: str, url: str, radius: int = MAX_CONTEXT_CHARS // 2) -> str:
    if not text:
        return ""

    index = text.find(url)
    if index < 0:
        return text[:MAX_CONTEXT_CHARS].strip()

    start = max(0, index - radius)
    end = min(len(text), index + len(url) + radius)
    return text[start:end].strip()


def build_candidate_chunks(text: str, urls: list[str]) -> list[CandidateChunk]:
    seen: set[str] = set()
    candidates: list[CandidateChunk] = []

    for url in urls:
        if url in seen or not is_candidate_url(url):
            continue
        seen.add(url)
        candidates.append(CandidateChunk(url=url, context=_context_for_url(text, url)))

    return candidates


def _direct_text_candidate(email: dict) -> CandidateChunk | None:
    """Create one candidate from a parsed attachment when no URL is required.

    This is what lets a forwarded Goh Ze Li email containing a JD PDF enter
    Track B even when the PDF itself has no public application URL.
    """
    attachment_text = (email.get("attachment_text") or "").strip()
    if not attachment_text:
        return None

    return CandidateChunk(
        url="",
        context=attachment_text[:MAX_DIRECT_TEXT_CHARS],
    )


def _chunk_batches(values: list[CandidateChunk], size: int) -> list[list[CandidateChunk]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _format_candidates(candidates: list[CandidateChunk]) -> str:
    blocks: list[str] = []
    for index, candidate in enumerate(candidates, start=1):
        url_text = candidate.url or "<none: extracted directly from email/attachment>"
        blocks.append(
            f"CANDIDATE {index}\n"
            f"URL: {url_text}\n"
            f"EMAIL CONTEXT:\n{candidate.context}"
        )
    return "\n\n---\n\n".join(blocks)


def _build_llm() -> ChatGroq:
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is missing from .env")

    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
    ).with_structured_output(ExtractedOpportunityBatch)


def _extract_batch(candidates: list[CandidateChunk]) -> ExtractedOpportunityBatch:
    llm = _build_llm()
    prompt = f"""
You extract career opportunities from NUS career-email fragments.

Rules:
- Return an opportunity only when the fragment describes a concrete job, internship,
  recruitment programme, career event, workshop, challenge, or similar actionable opportunity.
- Do not invent company, title, location, deadline, target major, or degree level.
- Use null/empty values when the email fragment does not state something.
- opportunity_type must be internship, full_time, event, or unknown.
- Keep only URLs that belong to that opportunity. A candidate may have no URL when it
  came directly from a PDF attachment; do not invent one.
- Generic NUS navigation, social-media, image and mailbox links are not opportunities.
- Multiple candidates may refer to one opportunity; avoid duplicate opportunities.
- evidence_text must be a short verbatim-or-near-verbatim excerpt supporting the extraction.
- deadline_hint may be natural language such as '13 September 2026'.

Candidates:\n{_format_candidates(candidates)}
""".strip()

    return llm.invoke(prompt)


def _source_name(email: dict) -> str:
    return email.get("sender_name") or email.get("sender_email") or "NUS career email"


def _normalize_deadline(value: str | None) -> date | None:
    if not value:
        return None

    cleaned = " ".join(value.strip().replace(",", "").split())
    formats = (
        "%Y-%m-%d",
        "%d %B %Y",
        "%d %b %Y",
        "%B %d %Y",
        "%b %d %Y",
    )
    for fmt in formats:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def extract_signal(state: dict) -> dict:
    if not should_extract_signal(state):
        return {"opportunity_signals": []}

    email = state.get("email") or {}
    text = state.get("normalized_text") or email.get("body_text") or ""
    urls = state.get("extracted_links") or email.get("links") or []

    candidates = build_candidate_chunks(text, urls)

    direct_candidate = _direct_text_candidate(email)
    if direct_candidate:
        candidates.append(direct_candidate)

    if not candidates:
        return {
            "opportunity_signals": [],
            "errors": state.get("errors", []),
        }

    extracted = []
    try:
        for batch in _chunk_batches(candidates, MAX_CANDIDATES_PER_LLM_CALL):
            result = _extract_batch(batch)
            extracted.extend(result.opportunities)
    except Exception as exc:
        errors = [*state.get("errors", []), f"signal extraction failed: {exc}"]
        return {"opportunity_signals": [], "errors": errors}

    signals: list[dict] = []
    seen_keys: set[tuple[str, str, str]] = set()

    for item in extracted:
        signal_urls = item.urls or []
        key = (
            (item.company or "").strip().lower(),
            (item.role_title or "").strip().lower(),
            "|".join(sorted(signal_urls)),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)

        signal = OpportunitySignal(
            source_type="outlook",
            source_name=_source_name(email),
            source_message_id=email.get("message_id") or "unknown",
            source_date=email.get("received_at"),
            company=item.company,
            role_title=item.role_title,
            location=item.location,
            opportunity_type=item.opportunity_type,
            deadline_hint=_normalize_deadline(item.deadline_hint),
            target_major=item.target_major or [],
            target_degree_level=item.target_degree_level or [],
            urls=signal_urls,
            raw_text=item.evidence_text,
            resolution_status="unresolved",
        )
        signals.append(signal.model_dump(mode="json"))

    return {
        "opportunity_signals": signals,
        "errors": state.get("errors", []),
    }
