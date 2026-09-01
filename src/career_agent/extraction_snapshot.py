from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from career_agent.job_normalization import canonical_company_text
from career_agent.models.signal import OpportunitySignal

SNAPSHOT_DIR = Path("data/extraction_cache")
LEGACY_BOOTSTRAP_PATHS = (
    Path("data/job_records/latest_job_records_archive.json"),
    Path("data/job_records/latest_job_catalog.json"),
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _snapshot_path(source_key: str, source_message_id: str) -> Path:
    digest = hashlib.sha256(source_message_id.encode("utf-8")).hexdigest()[:24]
    return SNAPSHOT_DIR / f"{source_key}_{digest}.json"


def load_snapshot(source_key: str, source_message_id: str) -> list[OpportunitySignal]:
    path = _snapshot_path(source_key, source_message_id)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [OpportunitySignal.model_validate(item) for item in payload.get("opportunities", [])]
    except Exception:
        return []


def save_snapshot(
    source_key: str,
    source_message_id: str,
    opportunities: list[OpportunitySignal],
) -> None:
    if not opportunities:
        return
    path = _snapshot_path(source_key, source_message_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "simplinext.extraction_snapshot.v1",
        "source_key": source_key,
        "source_message_id": source_message_id,
        "opportunity_count": len(opportunities),
        "opportunities": [item.model_dump(mode="json") for item in opportunities],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _signal_key(signal: OpportunitySignal) -> tuple[str, str]:
    return (
        canonical_company_text(signal.company),
        " ".join((signal.role_title or "").lower().split()),
    )


def _merge_signals(*groups: list[OpportunitySignal]) -> list[OpportunitySignal]:
    merged: dict[tuple[str, str], OpportunitySignal] = {}
    for group in groups:
        for signal in group:
            key = _signal_key(signal)
            if not all(key):
                continue
            previous = merged.get(key)
            if previous is None:
                merged[key] = signal
                continue
            merged[key] = previous.model_copy(
                update={
                    "location": previous.location or signal.location,
                    "opportunity_type": (
                        previous.opportunity_type
                        if previous.opportunity_type != "unknown"
                        else signal.opportunity_type
                    ),
                    "deadline_hint": previous.deadline_hint or signal.deadline_hint,
                    "target_major": list(dict.fromkeys([*previous.target_major, *signal.target_major])),
                    "target_degree_level": list(
                        dict.fromkeys([*previous.target_degree_level, *signal.target_degree_level])
                    ),
                    "urls": list(dict.fromkeys([*previous.urls, *signal.urls])),
                    "raw_text": (
                        previous.raw_text
                        if len(previous.raw_text or "") >= len(signal.raw_text or "")
                        else signal.raw_text
                    ),
                }
            )
    return list(merged.values())


def _normalized_tokens(value: str | None) -> list[str]:
    return TOKEN_PATTERN.findall((value or "").lower().replace("&", " and "))


def _company_supported_by_corpus(company: str | None, corpus: str) -> bool:
    canonical = canonical_company_text(company)
    if not canonical or canonical == "unknown":
        return False
    lowered = corpus.lower()
    compact = re.sub(r"[^a-z0-9]", "", lowered)

    aliases = {
        canonical,
        canonical.replace(" ", ""),
        (company or "").lower(),
        re.sub(r"[^a-z0-9]", "", (company or "").lower()),
    }
    if canonical == "procter gamble":
        aliases.update({"p&g", "p and g", "pg", "procter & gamble", "procter and gamble"})
    if canonical == "ernst young":
        aliases.add("ey")
    if canonical == "boston consulting group":
        aliases.update({"bcg", "the boston consulting group"})

    for alias in aliases:
        if not alias:
            continue
        alias_lower = alias.lower()
        alias_compact = re.sub(r"[^a-z0-9]", "", alias_lower)
        if len(alias_compact) <= 3:
            if re.search(rf"(?<![a-z0-9]){re.escape(alias_lower)}(?![a-z0-9])", lowered):
                return True
        elif alias_lower in lowered or alias_compact in compact:
            return True
    return False


def _role_supported_by_corpus(role_title: str | None, corpus: str) -> bool:
    role_tokens = [token for token in _normalized_tokens(role_title) if len(token) >= 2]
    if not role_tokens:
        return False
    corpus_tokens = set(_normalized_tokens(corpus))
    overlap = len(set(role_tokens) & corpus_tokens) / max(1, len(set(role_tokens)))

    normalized_role = " ".join(role_tokens)
    normalized_corpus = " ".join(_normalized_tokens(corpus))
    if normalized_role and normalized_role in normalized_corpus:
        return True

    # Require strong token support for titles whose punctuation/HTML formatting
    # changed between the old catalog and the current Outlook corpus.
    threshold = 1.0 if len(set(role_tokens)) <= 2 else 0.80
    return overlap >= threshold


def signal_supported_by_current_corpus(signal: OpportunitySignal, corpus: str) -> bool:
    return _company_supported_by_corpus(signal.company, corpus) and _role_supported_by_corpus(
        signal.role_title,
        corpus,
    )


def _jobs_from_payload(
    payload: dict,
    *,
    source_key: str,
    source_name: str,
    source_date,
    source_message_id: str | None = None,
    current_corpus: str | None = None,
) -> list[OpportunitySignal]:
    recovered: list[OpportunitySignal] = []
    for job in payload.get("jobs", []):
        job_message_id = job.get("source_message_id")
        if source_message_id is not None and job_message_id != source_message_id:
            continue
        if job.get("source_key") not in {source_key, None}:
            continue
        if job.get("record_kind") not in {"job_posting", None}:
            continue
        company = job.get("company")
        title = job.get("title")
        if not company or not title:
            continue
        try:
            signal = OpportunitySignal(
                source_type="outlook",
                source_name=source_name,
                source_message_id=source_message_id or job_message_id or "catalog-recovery",
                source_date=source_date,
                company=company,
                role_title=title,
                location=job.get("location"),
                opportunity_type=job.get("opportunity_type") or "unknown",
                deadline_hint=job.get("deadline_hint"),
                target_major=job.get("target_major") or [],
                target_degree_level=job.get("target_degree_level") or [],
                urls=job.get("source_urls") or [],
                raw_text=job.get("source_evidence") or f"{company} | {title}",
                resolution_status="unresolved",
            )
        except Exception:
            continue
        if current_corpus is not None and not signal_supported_by_current_corpus(signal, current_corpus):
            continue
        if source_message_id is not None:
            signal = signal.model_copy(update={"source_message_id": source_message_id})
        recovered.append(signal)
    return recovered


def _load_catalog_payloads() -> list[dict]:
    payloads: list[dict] = []
    for path in LEGACY_BOOTSTRAP_PATHS:
        if not path.exists():
            continue
        try:
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return payloads


def recover_from_existing_catalog(
    *,
    source_key: str,
    source_message_id: str,
    source_name: str,
    source_date,
) -> list[OpportunitySignal]:
    recovered: list[OpportunitySignal] = []
    for payload in _load_catalog_payloads():
        recovered.extend(
            _jobs_from_payload(
                payload,
                source_key=source_key,
                source_name=source_name,
                source_date=source_date,
                source_message_id=source_message_id,
            )
        )
    return _merge_signals(recovered)


def recover_supported_roles_from_any_prior_message(
    *,
    source_key: str,
    source_message_id: str,
    source_name: str,
    source_date,
    current_corpus: str,
) -> list[OpportunitySignal]:
    """Recover only old roles whose company + title are present in current corpus.

    This permits recovery when a Graph message id changes, without importing stale
    jobs from a different weekly newsletter that merely shares the same subject.
    """
    recovered: list[OpportunitySignal] = []
    for payload in _load_catalog_payloads():
        recovered.extend(
            _jobs_from_payload(
                payload,
                source_key=source_key,
                source_name=source_name,
                source_date=source_date,
                source_message_id=None,
                current_corpus=current_corpus,
            )
        )
    rewritten = [
        signal.model_copy(update={"source_message_id": source_message_id})
        for signal in recovered
    ]
    return _merge_signals(rewritten)


def bootstrap_from_existing_catalog(
    *,
    source_key: str,
    source_message_id: str,
    source_name: str,
    source_date,
) -> list[OpportunitySignal]:
    recovered = recover_from_existing_catalog(
        source_key=source_key,
        source_message_id=source_message_id,
        source_name=source_name,
        source_date=source_date,
    )
    if recovered:
        save_snapshot(source_key, source_message_id, recovered)
    return recovered


def load_or_bootstrap_snapshot(
    *,
    source_key: str,
    source_message_id: str,
    source_name: str,
    source_date,
    current_corpus: str | None = None,
) -> list[OpportunitySignal]:
    """Merge snapshot with prior evidence, optionally validated against current mail."""
    cached = load_snapshot(source_key, source_message_id)
    exact = recover_from_existing_catalog(
        source_key=source_key,
        source_message_id=source_message_id,
        source_name=source_name,
        source_date=source_date,
    )
    content_supported: list[OpportunitySignal] = []
    if current_corpus:
        content_supported = recover_supported_roles_from_any_prior_message(
            source_key=source_key,
            source_message_id=source_message_id,
            source_name=source_name,
            source_date=source_date,
            current_corpus=current_corpus,
        )

    merged = _merge_signals(cached, exact, content_supported)
    if merged and (
        len(merged) != len(cached)
        or any(_signal_key(item) not in {_signal_key(old) for old in cached} for item in merged)
    ):
        save_snapshot(source_key, source_message_id, merged)
    return merged
