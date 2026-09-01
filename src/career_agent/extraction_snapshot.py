from __future__ import annotations

import hashlib
import json
from pathlib import Path

from career_agent.models.signal import OpportunitySignal

SNAPSHOT_DIR = Path("data/extraction_cache")
LEGACY_BOOTSTRAP_PATHS = (
    Path("data/job_records/latest_job_records_archive.json"),
    Path("data/job_records/latest_job_catalog.json"),
)


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
        " ".join((signal.company or "").lower().split()),
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


def recover_from_existing_catalog(
    *,
    source_key: str,
    source_message_id: str,
    source_name: str,
    source_date,
) -> list[OpportunitySignal]:
    """Recover prior evidence for this exact Outlook message without writing cache."""
    recovered: list[OpportunitySignal] = []
    for path in LEGACY_BOOTSTRAP_PATHS:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        for job in payload.get("jobs", []):
            if job.get("source_message_id") != source_message_id:
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
                recovered.append(
                    OpportunitySignal(
                        source_type="outlook",
                        source_name=source_name,
                        source_message_id=source_message_id,
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
                )
            except Exception:
                continue
    return _merge_signals(recovered)


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
) -> list[OpportunitySignal]:
    """Merge local snapshot with all prior evidence for the same Outlook message.

    A partial snapshot must never permanently hide roles that existed in a previous
    canonical/archive run of the exact same source message.
    """
    cached = load_snapshot(source_key, source_message_id)
    recovered = recover_from_existing_catalog(
        source_key=source_key,
        source_message_id=source_message_id,
        source_name=source_name,
        source_date=source_date,
    )
    merged = _merge_signals(cached, recovered)
    if merged and len(merged) != len(cached):
        save_snapshot(source_key, source_message_id, merged)
    return merged
