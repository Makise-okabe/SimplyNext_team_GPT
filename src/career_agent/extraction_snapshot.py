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


def bootstrap_from_existing_catalog(
    *,
    source_key: str,
    source_message_id: str,
    source_name: str,
    source_date,
) -> list[OpportunitySignal]:
    """Recover stable extraction evidence from a previous run of the same email.

    This is only a repeat-run bootstrap. It never carries opportunities across a
    different Outlook message id, so a new career email still gets fresh extraction.
    """
    for path in LEGACY_BOOTSTRAP_PATHS:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        recovered: list[OpportunitySignal] = []
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
        if recovered:
            save_snapshot(source_key, source_message_id, recovered)
            return recovered
    return []


def load_or_bootstrap_snapshot(
    *,
    source_key: str,
    source_message_id: str,
    source_name: str,
    source_date,
) -> list[OpportunitySignal]:
    cached = load_snapshot(source_key, source_message_id)
    if cached:
        return cached
    return bootstrap_from_existing_catalog(
        source_key=source_key,
        source_message_id=source_message_id,
        source_name=source_name,
        source_date=source_date,
    )
