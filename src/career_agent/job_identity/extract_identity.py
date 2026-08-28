from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from career_agent.models.email import EmailMessage
from career_agent.models.job_identity import (
    ExtractedJobIdentity,
    ExtractedJobIdentityBatch,
    IdentityExtractionMetrics,
    JobIdentifier,
    JobIdentity,
    JobIdentityExtractionResult,
)

MAX_CONTEXT_CHARS_PER_SIGNAL = 4200
MAX_SIGNALS_PER_LLM_CALL = 4
MAX_BATCH_SOURCE_CHARS = 15000
CONTEXT_RADIUS = 1200
MAX_DISTINCTIVE_PHRASES = 8

URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
IDENTIFIER_PATTERN = re.compile(
    r"(?ix)\b"
    r"(?P<label>"
    r"job\s*(?:req(?:uisition)?\s*)?(?:id|no\.?|number)|"
    r"requisition\s*(?:id|no\.?|number)|"
    r"req\s*(?:id|no\.?|number)|"
    r"posting\s*(?:id|no\.?|number)|"
    r"reference\s*(?:id|no\.?|number)|"
    r"job\s*code|"
    r"position\s*(?:id|no\.?|number)"
    r")"
    r"\s*[:#\-]?\s*"
    r"(?P<value>[A-Z0-9][A-Z0-9._/\-]{2,})"
)

NOISE_URL_HOSTS = {
    "outlook.office.com",
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


@dataclass(frozen=True)
class IdentityInput:
    source_index: int
    signal: dict
    context: str


def _normalize_ws(value: str | None) -> str:
    return " ".join((value or "").split())


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _normalize_ws(value)
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def _identifier_kind(label: str) -> str:
    cleaned = _normalize_ws(label).lower()
    if "requisition" in cleaned or cleaned.startswith("req") or "job req" in cleaned:
        return "requisition_id"
    if "posting" in cleaned:
        return "posting_id"
    if "reference" in cleaned:
        return "reference_number"
    if "job code" in cleaned:
        return "job_code"
    if "position" in cleaned:
        return "position_id"
    if "job" in cleaned:
        return "job_id"
    return "other"


def extract_identifiers(text: str) -> list[JobIdentifier]:
    """Extract explicit job/requisition identifiers without an LLM."""
    identifiers: list[JobIdentifier] = []
    seen: set[tuple[str, str]] = set()

    for match in IDENTIFIER_PATTERN.finditer(text or ""):
        label = _normalize_ws(match.group("label"))
        value = match.group("value").strip().rstrip(".,;:)")
        key = (_identifier_kind(label), value.lower())
        if key in seen:
            continue
        seen.add(key)
        identifiers.append(
            JobIdentifier(
                kind=key[0],
                label=label,
                value=value,
            )
        )

    return identifiers


def _context_window(text: str, anchor: str, radius: int = CONTEXT_RADIUS) -> str:
    if not text or not anchor:
        return ""

    index = text.lower().find(anchor.lower())
    if index < 0:
        return ""

    start = max(0, index - radius)
    end = min(len(text), index + len(anchor) + radius)
    return text[start:end].strip()


def _signal_anchors(signal: dict) -> list[str]:
    anchors: list[str] = []
    for key in ("role_title", "company"):
        value = _normalize_ws(signal.get(key))
        if len(value) >= 4:
            anchors.append(value)

    raw_text = _normalize_ws(signal.get("raw_text"))
    if raw_text:
        anchors.append(raw_text[:220])

    for url in signal.get("urls") or []:
        if url:
            anchors.append(url)

    return anchors


def build_signal_context(email: EmailMessage, signal: dict) -> str:
    """Build a bounded, role-specific context instead of sending whole newsletters.

    The source email/PDF may be tens of thousands of characters. V1 only sends
    the subject, signal evidence and windows around role-specific anchors.
    """
    text = email.body_text or ""
    segments: list[str] = []

    if email.subject:
        segments.append(f"EMAIL SUBJECT: {email.subject}")

    raw_text = _normalize_ws(signal.get("raw_text"))
    if raw_text:
        segments.append(f"SIGNAL EVIDENCE: {raw_text}")

    for anchor in _signal_anchors(signal):
        window = _context_window(text, anchor)
        if window:
            segments.append(window)

    # Attachment text is especially valuable for attachment-only roles. Search
    # it independently so a JD near the end of a long email is not lost.
    attachment_text = email.attachment_text or ""
    if attachment_text:
        for anchor in _signal_anchors(signal):
            window = _context_window(attachment_text, anchor)
            if window:
                segments.append(f"ATTACHMENT CONTEXT:\n{window}")

    if len(segments) <= 2 and text:
        segments.append(text[:1800])

    unique_segments = _dedupe(segments)
    context = "\n\n---\n\n".join(unique_segments)
    return context[:MAX_CONTEXT_CHARS_PER_SIGNAL]


def _candidate_urls(signal: dict) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()

    for url in signal.get("urls") or []:
        cleaned = (url or "").strip()
        if not cleaned or cleaned in seen:
            continue
        try:
            parsed = urlparse(cleaned)
        except ValueError:
            continue
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        if parsed.netloc.lower() in NOISE_URL_HOSTS:
            continue
        seen.add(cleaned)
        urls.append(cleaned)

    return urls


def _grounded_value(value: str | None, context: str) -> str | None:
    cleaned = _normalize_ws(value)
    if not cleaned:
        return None

    normalized_context = _normalize_ws(context).lower()
    if cleaned.lower() in normalized_context:
        return cleaned
    return None


def _grounded_list(values: list[str] | None, context: str, limit: int) -> list[str]:
    if not values:
        return []

    normalized_context = _normalize_ws(context).lower()
    grounded: list[str] = []
    for value in values:
        cleaned = _normalize_ws(value)
        if not cleaned or cleaned.lower() not in normalized_context:
            continue
        grounded.append(cleaned)
        if len(grounded) >= limit:
            break
    return _dedupe(grounded)


def _identity_strength(
    company: str | None,
    title: str | None,
    identifiers: list[JobIdentifier],
    distinctive_phrases: list[str],
) -> str:
    if identifiers:
        return "strong"
    if company and title and len(distinctive_phrases) >= 2:
        return "strong"
    if company and title:
        return "moderate"
    return "weak"


def _fingerprint(
    company: str | None,
    title: str | None,
    location: str | None,
    opportunity_type: str,
    identifiers: list[JobIdentifier],
) -> str:
    canonical = "|".join(
        [
            _normalize_ws(company).lower(),
            _normalize_ws(title).lower(),
            _normalize_ws(location).lower(),
            opportunity_type.lower(),
            ",".join(sorted(item.value.lower() for item in identifiers)),
        ]
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_llm():
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is missing from .env")

    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
    ).with_structured_output(ExtractedJobIdentityBatch)


def _format_batch(batch: list[IdentityInput]) -> str:
    blocks: list[str] = []
    for item in batch:
        signal = item.signal
        blocks.append(
            f"SOURCE_INDEX: {item.source_index}\n"
            f"KNOWN COMPANY: {signal.get('company')}\n"
            f"KNOWN TITLE: {signal.get('role_title')}\n"
            f"KNOWN LOCATION: {signal.get('location')}\n"
            f"KNOWN TYPE: {signal.get('opportunity_type')}\n"
            f"SOURCE CONTEXT:\n{item.context}"
        )
    return "\n\n====================\n\n".join(blocks)


def _invoke_batch(batch: list[IdentityInput]) -> ExtractedJobIdentityBatch:
    prompt = f"""
You extract JOB IDENTITY attributes for later same-job verification.
You are NOT searching the web and you are NOT deciding whether a job is verified.

Return exactly one identity for every SOURCE_INDEX supplied.

Rules:
- Never invent a fact that is absent from SOURCE CONTEXT.
- company/title/location may repeat the KNOWN fields when supported by the context.
- business_unit and team are only for explicitly named organisational/team context.
- duration/start_period/end_period must preserve the source wording when possible.
- target_cohort means explicit student/year/degree cohort requirements.
- distinctive_phrases: return 3-8 exact phrases from the source that are useful for
  distinguishing this role on the web. Prefer product names, team names, unusual
  responsibilities, named systems/technologies, programme names, and role-specific
  wording. Avoid generic phrases such as 'teamwork', 'communication skills',
  'Python', or 'data analysis' unless the phrase itself is unusually specific.
- Do not create job IDs. Explicit IDs are extracted deterministically elsewhere.
- Use null/empty lists whenever evidence is missing.

CANDIDATES:
{_format_batch(batch)}
""".strip()
    return _build_llm().invoke(prompt)


def _chunk_inputs(values: list[IdentityInput]) -> list[list[IdentityInput]]:
    batches: list[list[IdentityInput]] = []
    current: list[IdentityInput] = []
    current_chars = 0

    for value in values:
        size = len(value.context)
        if current and (
            len(current) >= MAX_SIGNALS_PER_LLM_CALL
            or current_chars + size > MAX_BATCH_SOURCE_CHARS
        ):
            batches.append(current)
            current = []
            current_chars = 0

        current.append(value)
        current_chars += size

    if current:
        batches.append(current)
    return batches


def _build_identity(
    email: EmailMessage,
    item: IdentityInput,
    extracted: ExtractedJobIdentity | None,
) -> JobIdentity:
    signal = item.signal
    context = item.context

    company = (
        _grounded_value(extracted.company, context) if extracted else None
    ) or signal.get("company")
    title = (
        _grounded_value(extracted.title, context) if extracted else None
    ) or signal.get("role_title")
    location = (
        _grounded_value(extracted.location, context) if extracted else None
    ) or signal.get("location")

    identifiers = extract_identifiers(context)
    distinctive_phrases = _grounded_list(
        extracted.distinctive_phrases if extracted else None,
        context,
        MAX_DISTINCTIVE_PHRASES,
    )

    business_unit = _grounded_value(extracted.business_unit, context) if extracted else None
    team = _grounded_value(extracted.team, context) if extracted else None
    duration = _grounded_value(extracted.duration, context) if extracted else None
    start_period = _grounded_value(extracted.start_period, context) if extracted else None
    end_period = _grounded_value(extracted.end_period, context) if extracted else None
    target_cohort = _grounded_list(
        extracted.target_cohort if extracted else None,
        context,
        8,
    )

    employment_type = None
    if extracted:
        employment_type = _grounded_value(extracted.employment_type, context)
    if not employment_type and signal.get("opportunity_type") in {"internship", "full_time"}:
        employment_type = signal.get("opportunity_type")

    opportunity_type = signal.get("opportunity_type") or "unknown"
    evidence = _dedupe(
        [
            signal.get("raw_text") or "",
            *distinctive_phrases,
            *(f"{identifier.label}: {identifier.value}" for identifier in identifiers),
        ]
    )[:12]

    return JobIdentity(
        source_message_id=email.message_id,
        signal_index=item.source_index,
        company=company,
        title=title,
        identifiers=identifiers,
        location=location,
        opportunity_type=opportunity_type,
        business_unit=business_unit,
        team=team,
        employment_type=employment_type,
        duration=duration,
        start_period=start_period,
        end_period=end_period,
        target_cohort=target_cohort,
        distinctive_phrases=distinctive_phrases,
        direct_urls=_candidate_urls(signal),
        evidence_snippets=evidence,
        identity_strength=_identity_strength(
            company,
            title,
            identifiers,
            distinctive_phrases,
        ),
        source_fingerprint=_fingerprint(
            company,
            title,
            location,
            opportunity_type,
            identifiers,
        ),
    )


def extract_job_identities(
    email: EmailMessage,
    signals: list[dict],
) -> JobIdentityExtractionResult:
    """Build bounded, grounded JobIdentity objects for all opportunity signals.

    V1 intentionally does no web search. The output is the compact identity state
    consumed by V2 progressive search and V3 same-job comparison.
    """
    started = time.perf_counter()
    inputs = [
        IdentityInput(
            source_index=index,
            signal=signal,
            context=build_signal_context(email, signal),
        )
        for index, signal in enumerate(signals, start=1)
    ]

    extracted_by_index: dict[int, ExtractedJobIdentity] = {}
    errors: list[str] = []
    llm_calls = 0
    batches = _chunk_inputs(inputs)

    for batch_number, batch in enumerate(batches, start=1):
        try:
            result = _invoke_batch(batch)
            llm_calls += 1
            valid_indexes = {item.source_index for item in batch}
            for extracted in result.identities:
                if extracted.source_index in valid_indexes:
                    extracted_by_index[extracted.source_index] = extracted
        except Exception as exc:
            llm_calls += 1
            errors.append(
                f"identity extraction batch {batch_number} failed: "
                f"{type(exc).__name__}: {exc}"
            )

    identities = [
        _build_identity(
            email,
            item,
            extracted_by_index.get(item.source_index),
        )
        for item in inputs
    ]

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return JobIdentityExtractionResult(
        identities=identities,
        metrics=IdentityExtractionMetrics(
            signals_seen=len(signals),
            identities_built=len(identities),
            llm_calls=llm_calls,
            batches=len(batches),
            source_chars_sent=sum(len(item.context) for item in inputs),
            elapsed_ms=elapsed_ms,
        ),
        errors=errors,
    )
