from __future__ import annotations

from career_agent.job_normalization import canonical_company_text, clean_company_name, clean_role_title, titles_equivalent
from career_agent.models.job_record import JobRecord

JD_RANK = {
    "unavailable": 0,
    "source_context_only": 1,
    "fetched_secondary": 2,
    "fetched_official": 3,
}
STATUS_RANK = {
    "unresolved": 0,
    "source_verified": 1,
    "secondary_corroborated": 2,
    "verified_exact_job": 3,
}
CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def _same_job(left: JobRecord, right: JobRecord) -> bool:
    if canonical_company_text(left.company) != canonical_company_text(right.company):
        return False
    if left.opportunity_type != right.opportunity_type and "unknown" not in {
        left.opportunity_type,
        right.opportunity_type,
    }:
        return False
    return titles_equivalent(
        left.title,
        right.title,
        left_company=left.company,
        right_company=right.company,
    )


def _strength(job: JobRecord) -> tuple[int, int, int, int]:
    return (
        JD_RANK.get(job.jd_status, 0),
        STATUS_RANK.get(job.research_status, 0),
        CONFIDENCE_RANK.get(job.research_confidence, 0),
        1 if job.primary_source_url else 0,
    )


def _merge_pair(left: JobRecord, right: JobRecord) -> JobRecord:
    winner, other = (left, right) if _strength(left) >= _strength(right) else (right, left)
    clean_winner_title, winner_title_urls = clean_role_title(winner.title)
    clean_other_title, other_title_urls = clean_role_title(other.title)

    title = clean_winner_title
    if not title or (clean_other_title and len(clean_other_title) > len(title) * 1.8):
        title = clean_other_title or title

    source_keys = list(dict.fromkeys([left.source_key, right.source_key]))
    merged_evidence = list(dict.fromkeys([
        *winner.evidence_summary,
        *other.evidence_summary,
        *( [f"same opportunity corroborated across sources: {', '.join(source_keys)}"] if len(source_keys) > 1 else [] ),
    ]))

    availability = winner.availability_status
    if "closed_by_official" in {left.availability_status, right.availability_status}:
        availability = "closed_by_official"
    elif availability == "unknown" and other.availability_status != "unknown":
        availability = other.availability_status

    opportunity_type = winner.opportunity_type
    if opportunity_type == "unknown" and other.opportunity_type != "unknown":
        opportunity_type = other.opportunity_type

    return winner.model_copy(
        update={
            "company": clean_company_name(winner.company) or clean_company_name(other.company),
            "title": title,
            "location": winner.location or other.location,
            "opportunity_type": opportunity_type,
            "deadline_hint": winner.deadline_hint or other.deadline_hint,
            "availability_status": availability,
            "target_major": list(dict.fromkeys([*winner.target_major, *other.target_major])),
            "target_degree_level": list(dict.fromkeys([*winner.target_degree_level, *other.target_degree_level])),
            "source_urls": list(dict.fromkeys([
                *winner.source_urls,
                *other.source_urls,
                *winner_title_urls,
                *other_title_urls,
            ])),
            "primary_source_url": winner.primary_source_url or other.primary_source_url,
            "secondary_source_url": winner.secondary_source_url or other.secondary_source_url,
            "official_job_url": winner.official_job_url or other.official_job_url,
            "application_url": winner.application_url or other.application_url,
            "jd_source_url": winner.jd_source_url or other.jd_source_url,
            "jd_text": winner.jd_text or other.jd_text,
            "source_evidence": winner.source_evidence if len(winner.source_evidence or "") >= len(other.source_evidence or "") else other.source_evidence,
            "evidence_summary": merged_evidence,
            "warnings": list(dict.fromkeys([*winner.warnings, *other.warnings])),
            "errors": list(dict.fromkeys([*winner.errors, *other.errors])),
        }
    )


def consolidate_job_records(jobs: list[JobRecord]) -> list[JobRecord]:
    """Return one canonical record per job while archive can retain raw records."""
    merged: list[JobRecord] = []
    for job in jobs:
        clean_title, title_urls = clean_role_title(job.title)
        normalized = job.model_copy(
            update={
                "company": clean_company_name(job.company),
                "title": clean_title,
                "source_urls": list(dict.fromkeys([*job.source_urls, *title_urls])),
            }
        )

        match_index = next(
            (index for index, existing in enumerate(merged) if _same_job(existing, normalized)),
            None,
        )
        if match_index is None:
            merged.append(normalized)
        else:
            merged[match_index] = _merge_pair(merged[match_index], normalized)
    return merged
