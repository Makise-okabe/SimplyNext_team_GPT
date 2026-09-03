from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from career_agent.company_job_research import (
    MIN_JD_CHARS,
    MIN_JD_TITLE_OVERLAP,
    ResearchContext,
    _company_match,
    _title_overlap,
    research_company_jobs,
)
from career_agent.job_research_quality import (
    clean_jd_text,
    is_aggregator_url,
    is_plausible_official_url,
    page_is_closed,
)
from career_agent.matching_dataset import matching_evidence_level, matching_input_text
from career_agent.models.email import EmailMessage
from career_agent.models.job_record import JobRecord
from career_agent.models.signal import OpportunitySignal
from career_agent.tools.web_search import SearchResult
from career_agent.tools.web_search_aggregate import search_public_web_aggregated

ProgressCallback = Callable[[str], None]
MAX_SECONDARY_FETCH_CANDIDATES = 3
MIN_SECONDARY_SEARCH_TITLE_OVERLAP = 0.45


@dataclass(frozen=True)
class ShortlistWebEnrichmentMetrics:
    selected: int
    already_full_jd: int
    researched: int
    upgraded_to_full_jd: int
    upgraded_official: int
    upgraded_secondary: int
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
    if researched.secondary_source_url:
        update["secondary_source_url"] = researched.secondary_source_url

    if researched.availability_status == "closed_by_official":
        update.update(
            {
                "availability_status": "closed_by_official",
                "research_status": researched.research_status,
                "research_confidence": researched.research_confidence,
                "research_basis": researched.research_basis,
            }
        )
    elif researched.jd_status in {"fetched_official", "fetched_secondary"} and researched.jd_text.strip():
        update.update(
            {
                "availability_status": researched.availability_status,
                "research_status": researched.research_status,
                "research_confidence": researched.research_confidence,
                "research_basis": researched.research_basis,
                "jd_status": researched.jd_status,
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


def _result_text(result: SearchResult) -> str:
    return f"{result.title} {result.snippet} {result.url}"


def _rank_secondary_candidates(job: JobRecord, results: list[SearchResult]) -> list[SearchResult]:
    scored: list[tuple[float, SearchResult]] = []
    for result in results:
        value = _result_text(result)
        overlap = _title_overlap(job.title, value)
        if overlap < MIN_SECONDARY_SEARCH_TITLE_OVERLAP:
            continue
        if not _company_match(job.company, value):
            continue

        official = is_plausible_official_url(result.url, job.company)
        score = overlap * 100.0
        if official:
            score += 30.0
        elif is_aggregator_url(result.url):
            score += 10.0
        scored.append((score, result))

    scored.sort(key=lambda item: item[0], reverse=True)
    seen: set[str] = set()
    ranked: list[SearchResult] = []
    for _, result in scored:
        if result.url in seen:
            continue
        seen.add(result.url)
        ranked.append(result)
    return ranked


def _usable_secondary_jd(job: JobRecord, page) -> str | None:
    value = f"{page.title}\n{page.text}"
    if not _company_match(job.company, value):
        return None
    cleaned = clean_jd_text(page.text)
    if len(cleaned.strip()) < MIN_JD_CHARS:
        return None
    if _title_overlap(job.title, cleaned) < MIN_JD_TITLE_OVERLAP:
        return None
    return cleaned[:30_000]


def _aggregated_fallback_search(context: ResearchContext, query: str) -> list[SearchResult]:
    """Count one logical shortlist search while aggregating several providers."""
    context.search_calls += 1
    try:
        return search_public_web_aggregated(query, max_results=12, min_results=6)
    except Exception as exc:
        context.warnings.append(
            f"aggregated shortlist search failed: {type(exc).__name__}: {exc} | query={query}"
        )
        return []


def _secondary_fallback(job: JobRecord, context: ResearchContext) -> JobRecord | None:
    """Use a verified public secondary JD only after official research fails.

    Official employer/ATS pages remain preferred. Secondary pages are accepted
    only when both search metadata and fetched page content match company/title.
    A secondary page saying the listing is closed may still supply historical JD
    evidence; it does not override the trusted NUS source's availability status.
    """
    company = job.company or ""
    title = job.title or ""
    queries = [
        f'"{company}" "{title}" careers job',
        f'"{company}" "{title}"',
    ]

    tried: set[str] = set()
    for query in queries:
        results = _aggregated_fallback_search(context, query)
        candidates = _rank_secondary_candidates(job, results)
        for result in candidates[:MAX_SECONDARY_FETCH_CANDIDATES]:
            if result.url in tried:
                continue
            tried.add(result.url)
            page = context.fetch(result.url)
            if page is None:
                continue

            final_url = page.final_url or result.url
            official = is_plausible_official_url(final_url, job.company)
            if official and page_is_closed(page.text):
                continue

            cleaned = _usable_secondary_jd(job, page)
            if not cleaned:
                continue

            if official:
                return job.model_copy(
                    update={
                        "availability_status": job.availability_status,
                        "research_status": "verified_exact_job",
                        "research_confidence": "high",
                        "research_basis": "official_company_or_ats_page_broad_search",
                        "primary_source_url": final_url,
                        "official_job_url": final_url,
                        "application_url": job.application_url or final_url,
                        "jd_status": "fetched_official",
                        "jd_source_url": final_url,
                        "jd_text": cleaned,
                        "evidence_summary": list(
                            dict.fromkeys(
                                [
                                    *job.evidence_summary,
                                    "official employer/ATS page matched the shortlisted role",
                                ]
                            )
                        ),
                    }
                )

            return job.model_copy(
                update={
                    "availability_status": job.availability_status,
                    "research_status": "verified_exact_job",
                    "research_confidence": "medium",
                    "research_basis": "verified_secondary_job_page",
                    "secondary_source_url": final_url,
                    "jd_status": "fetched_secondary",
                    "jd_source_url": final_url,
                    "jd_text": cleaned,
                    "evidence_summary": list(
                        dict.fromkeys(
                            [
                                *job.evidence_summary,
                                "verified public secondary page matched company and role title",
                            ]
                        )
                    ),
                }
            )
    return None


def enrich_stage1_shortlist(
    *,
    all_jobs: list[dict],
    stage1_rankings: list[dict],
    stage1_top_n: int = 20,
    progress: ProgressCallback | None = None,
) -> tuple[list[dict], ShortlistWebEnrichmentMetrics]:
    """Upgrade only the Stage-1 shortlist with grounded public web evidence."""
    selected = list(stage1_rankings[: max(stage1_top_n, 0)])
    selected_keys = {_job_key(item.get("company"), item.get("title")) for item in selected}

    originals = [_record(raw) for raw in all_jobs]
    context = ResearchContext()
    researched_count = 0
    already_full_jd = 0
    upgraded = 0
    upgraded_official = 0
    upgraded_secondary = 0
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

        if (
            merged.availability_status != "closed_by_official"
            and matching_evidence_level(merged) != "full_jd"
        ):
            fallback = _secondary_fallback(merged, context)
            if fallback is not None:
                merged = _merge(merged, fallback)

        updated_by_key[key] = merged

        if merged.availability_status == "closed_by_official":
            closed += 1
            if progress:
                progress("          -> closed_by_official")
        elif matching_evidence_level(merged) == "full_jd":
            upgraded += 1
            if merged.jd_status == "fetched_official":
                upgraded_official += 1
            elif merged.jd_status == "fetched_secondary":
                upgraded_secondary += 1
            if progress:
                progress(f"          -> full_jd ({merged.jd_status})")
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
        upgraded_official=upgraded_official,
        upgraded_secondary=upgraded_secondary,
        closed_by_official=closed,
        still_source_only=still_source_only,
        search_calls=context.search_calls,
        fetch_calls=context.fetch_calls,
    )
