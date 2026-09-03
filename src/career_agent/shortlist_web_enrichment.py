from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from career_agent.company_job_research import ResearchContext, research_company_jobs
from career_agent.matching_dataset import matching_evidence_level, matching_input_text
from career_agent.models.email import EmailMessage
from career_agent.models.job_record import JobRecord
from career_agent.models.signal import OpportunitySignal

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class ShortlistWebEnrichmentMetrics:
    selected: int
    already_full_jd: int
    researched: int
    upgraded_to_full_jd: int
    closed_by_official: int
    still_source_only: int
    search_calls: int
    fetch_calls: int


def _clean_job_payload(raw: dict) -> dict:
    value = dict(raw)
    value.pop("matching_ready", None)
    value.pop("matching_candidate", None)
    value.pop("matching_evidence_level", None)
    value.pop("matching_input_text", None)
    return value


def _record(raw: dict) -> JobRecord:
    return JobRecord.model_validate(_clean_job_payload(raw))


def _job_key(company: str | None, title: str | None) -> tuple[str, str]:
    return (
        " ".join((company or "").lower().split()),
        " ".join((title or "").lower().split()),
    )


def _to_signal(job: JobRecord) -> OpportunitySignal:
    urls = list(
        dict.fromkeys(
            url
            for url in [
                *job.source_urls,
                job.official_job_url,
                job.primary_source_url,
                job.application_url,
            ]
            if url
        )
    )
    return OpportunitySignal(
        source_type="outlook",
        source_name=job.source_sender_email or job.source_key,
        source_message_id=job.source_message_id,
        company=job.company,
        role_title=job.title,
        location=job.location,
        opportunity_type=job.opportunity_type,
        deadline_hint=job.deadline_hint,
        target_major=job.target_major,
        target_degree_level=job.target_degree_level,
        urls=urls,
        raw_text=job.source_evidence,
    )


def _synthetic_email(job: JobRecord) -> EmailMessage:
    return EmailMessage(
        message_id=job.source_message_id,
        sender_email=job.source_sender_email,
        subject=job.source_subject,
        body_text=job.source_evidence,
        links=list(job.source_urls),
    )


def _merge(original: JobRecord, researched: JobRecord) -> JobRecord:
    update: dict = {
        "warnings": list(dict.fromkeys([*original.warnings, *researched.warnings])),
        "errors": list(dict.fromkeys([*original.errors, *researched.errors])),
        "evidence_summary": list(
            dict.fromkeys([*original.evidence_summary, *researched.evidence_summary])
        ),
    }

    if researched.application_url:
        update["application_url"] = researched.application_url
    if researched.primary_source_url:
        update["primary_source_url"] = researched.primary_source_url
        update["official_job_url"] = researched.official_job_url or researched.primary_source_url

    if researched.availability_status == "closed_by_official":
        update.update(
            {
                "availability_status": "closed_by_official",
                "research_status": researched.research_status,
                "research_confidence": researched.research_confidence,
                "research_basis": researched.research_basis,
            }
        )
    elif researched.jd_status == "fetched_official" and researched.jd_text.strip():
        update.update(
            {
                "availability_status": researched.availability_status,
                "research_status": researched.research_status,
                "research_confidence": researched.research_confidence,
                "research_basis": researched.research_basis,
                "jd_status": "fetched_official",
                "jd_source_url": researched.jd_source_url,
                "jd_text": researched.jd_text,
            }
        )

    return original.model_copy(update=update)


def _matching_payload(job: JobRecord) -> dict:
    payload = job.model_dump(mode="json")
    payload["matching_evidence_level"] = matching_evidence_level(job)
    payload["matching_input_text"] = matching_input_text(job)
    return payload


def enrich_stage1_shortlist(
    *,
    all_jobs: list[dict],
    stage1_rankings: list[dict],
    stage1_top_n: int = 20,
    progress: ProgressCallback | None = None,
) -> tuple[list[dict], ShortlistWebEnrichmentMetrics]:
    """Upgrade only the Stage-1 shortlist with official web evidence.

    The broad candidate pool remains deterministic and cheap. Web search/fetch is
    reserved for shortlisted jobs, so a 137-job catalog does not trigger 137 web
    research operations.
    """
    selected = list(stage1_rankings[: max(stage1_top_n, 0)])
    selected_keys = {_job_key(item.get("company"), item.get("title")) for item in selected}

    originals = [_record(raw) for raw in all_jobs]
    context = ResearchContext()
    researched_count = 0
    already_full_jd = 0
    upgraded = 0
    closed = 0

    updated_by_key: dict[tuple[str, str], JobRecord] = {}
    shortlist_records = [job for job in originals if _job_key(job.company, job.title) in selected_keys]

    for index, job in enumerate(shortlist_records, start=1):
        key = _job_key(job.company, job.title)
        if matching_evidence_level(job) == "full_jd":
            already_full_jd += 1
            updated_by_key[key] = job
            if progress:
                progress(f"      [WEB {index:02}/{len(shortlist_records):02}] {job.company} — {job.title}: already full_jd")
            continue
        if job.availability_status in {"expired_by_source_deadline", "closed_by_official"}:
            updated_by_key[key] = job
            continue

        researched_count += 1
        if progress:
            progress(f"      [WEB {index:02}/{len(shortlist_records):02}] {job.company} — {job.title}")

        outcome = research_company_jobs(
            email=_synthetic_email(job),
            source_key=job.source_key,
            company_items=[(1, _to_signal(job))],
            context=context,
            progress=None,
        )
        researched = outcome.job_records[0] if outcome.job_records else job
        merged = _merge(job, researched)
        updated_by_key[key] = merged

        if merged.availability_status == "closed_by_official":
            closed += 1
            if progress:
                progress("          -> closed_by_official")
        elif matching_evidence_level(merged) == "full_jd":
            upgraded += 1
            if progress:
                progress("          -> full_jd")
        elif progress:
            progress("          -> source_only")

    enriched_jobs: list[dict] = []
    for original in originals:
        key = _job_key(original.company, original.title)
        enriched_jobs.append(_matching_payload(updated_by_key.get(key, original)))

    selected_after = [
        _record(raw)
        for raw in enriched_jobs
        if _job_key(raw.get("company"), raw.get("title")) in selected_keys
        and raw.get("availability_status") not in {"expired_by_source_deadline", "closed_by_official"}
    ]
    still_source_only = sum(matching_evidence_level(job) == "source_only" for job in selected_after)

    return enriched_jobs, ShortlistWebEnrichmentMetrics(
        selected=len(shortlist_records),
        already_full_jd=already_full_jd,
        researched=researched_count,
        upgraded_to_full_jd=upgraded,
        closed_by_official=closed,
        still_source_only=still_source_only,
        search_calls=context.search_calls,
        fetch_calls=context.fetch_calls,
    )
