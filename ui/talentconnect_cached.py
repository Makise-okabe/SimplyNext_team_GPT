from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from pathlib import Path

from career_agent.batch_sources import SOURCE_DOCUMENT_SEPARATOR
from career_agent.models.signal import ExtractedOpportunityBatch, OpportunitySignal
from career_agent.talentconnect_extraction import ExtractionMetrics, _chunks, _invoke

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / ".cache" / "ui_talentconnect_chunks_v1"
CACHE_VERSION = "talentconnect-prompt-v1"


@dataclass
class ExtractionState:
    rate_limited: bool = False


def _retry_delay(exc: Exception, fallback: float) -> float:
    headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        return fallback
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            return max(0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError):
            return fallback


def _normalize(value: str | None) -> str:
    return " ".join((value or "").lower().split())


def _cache_path(chunk: str) -> Path:
    digest = hashlib.sha256(
        (CACHE_VERSION + "\n" + chunk).encode("utf-8", errors="ignore")
    ).hexdigest()
    return CACHE_DIR / f"{digest}.json"


def _load_cached_batch(chunk: str) -> ExtractedOpportunityBatch | None:
    path = _cache_path(chunk)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ExtractedOpportunityBatch.model_validate(payload)
    except Exception:
        # Corrupt/stale cache entry is ignored and replaced on next success.
        return None


def _save_cached_batch(chunk: str, batch: ExtractedOpportunityBatch) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(chunk)
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(batch.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp.replace(path)


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "ratelimit" in text


def _invoke_with_retry(chunk: str) -> tuple[ExtractedOpportunityBatch | None, int, str | None]:
    """Pace new chunks and retry rate limits without redoing successful chunks."""
    max_attempts = max(1, int(os.getenv("SIMPLYNEXT_TC_MAX_ATTEMPTS", "2")))
    base_wait = max(1.0, float(os.getenv("SIMPLYNEXT_TC_RETRY_SECONDS", "8")))
    pace = max(0.0, float(os.getenv("SIMPLYNEXT_TC_PACE_SECONDS", "0")))

    attempts = 0
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        if pace:
            time.sleep(pace)
        attempts += 1
        try:
            return _invoke(chunk), attempts, None
        except Exception as exc:
            last_error = exc
            if not _is_rate_limit(exc) or attempt >= max_attempts:
                break
            delay = _retry_delay(exc, base_wait * (2 ** (attempt - 1)))
            if delay > 10.0:
                break
            time.sleep(delay)

    if last_error is None:
        return None, attempts, "unknown extraction failure"
    return None, attempts, f"{type(last_error).__name__}: {last_error}"


def extract_talentconnect_cached(
    *,
    source_name: str,
    source_message_id: str,
    source_date,
    corpus: str,
    state: ExtractionState | None = None,
) -> tuple[list[OpportunitySignal], ExtractionMetrics, list[str]]:
    """Frozen TalentConnect extraction semantics with UI-only per-chunk memoization.

    Outlook remains the live source of truth. Cache keys are the exact chunk content
    plus a prompt-version token, so changed/new mail is automatically re-extracted.
    Only successful structured LLM outputs are cached; failed/429 chunks are never cached.
    """
    state = state if state is not None else ExtractionState()
    merged: dict[tuple[str, str], OpportunitySignal] = {}
    errors: list[str] = []
    actual_calls = 0
    cache_hits = 0
    total_chunks = 0

    documents = [
        part.strip()
        for part in corpus.split(SOURCE_DOCUMENT_SEPARATOR)
        if part.strip()
    ]

    for document_index, document in enumerate(documents, start=1):
        for chunk_index, chunk in enumerate(_chunks(document), start=1):
            if len(chunk.strip()) < 120:
                continue
            total_chunks += 1

            batch = _load_cached_batch(chunk)
            if batch is not None:
                cache_hits += 1
            else:
                if state.rate_limited:
                    errors.append(f"TalentConnect document {document_index} chunk {chunk_index} deferred: provider rate limit; retry scan later.")
                    continue
                batch, attempts, error = _invoke_with_retry(chunk)
                actual_calls += attempts
                if batch is None:
                    if error and _is_rate_limit(RuntimeError(error)):
                        state.rate_limited = True
                    errors.append(
                        f"TalentConnect document {document_index} chunk {chunk_index} failed after "
                        f"{attempts} attempt(s): {error}"
                    )
                    continue
                _save_cached_batch(chunk, batch)

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

    errors.insert(
        0,
        f"INFO TalentConnect chunk cache: {cache_hits}/{total_chunks} reused; "
        f"actual LLM attempts this run={actual_calls}.",
    )
    return (
        list(merged.values()),
        ExtractionMetrics(llm_calls=actual_calls, source_chars=len(corpus)),
        errors,
    )
