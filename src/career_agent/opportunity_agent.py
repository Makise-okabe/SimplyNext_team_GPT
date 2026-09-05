from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import wraps

from career_agent.research_session import research_session
from career_agent.record_identity import record_key
from career_agent.matching_dataset import is_matching_candidate

from career_agent.jd_enricher import enrich_job_description
from career_agent.job_link_resolver import resolve_job_link
from career_agent.matching_dataset import matching_evidence_level, matching_input_text
from career_agent.models.job_record import JobRecord
from career_agent.related_job_discovery import discover_related_jobs
from career_agent.stage1_ranking import rank_jobs


@dataclass(frozen=True)
class OpportunityAgentMetrics:
    active_jobs: int
    web_selected: int
    links_resolved: int
    official_links: int
    secondary_links: int
    full_jd: int
    partial_jd: int
    unresolved_links: int
    related_jobs_discovered: int
    search_calls: int = 0
    page_fetch_calls: int = 0


@dataclass(frozen=True)
class OpportunityAgentResult:
    jobs: list[dict]
    stage1_rankings: list[dict]
    semantic_shortlist: list[dict]
    related_jobs: list[dict]
    related_rankings: list[dict]
    metrics: OpportunityAgentMetrics


def _clean_job_payload(raw: dict) -> dict:
    value = dict(raw)
    for key in (
        "matching_ready",
        "matching_candidate",
        "matching_evidence_level",
        "matching_input_text",
    ):
        value.pop(key, None)
    return value


def _record(raw: dict) -> JobRecord:
    return JobRecord.model_validate(_clean_job_payload(raw))


def _matching_payload(job: JobRecord) -> dict:
    payload = job.model_dump(mode="json")
    payload["matching_evidence_level"] = matching_evidence_level(job)
    payload["matching_input_text"] = matching_input_text(job)
    return payload


def _key(company: str | None, title: str | None) -> tuple[str, str]:
    return (
        " ".join((company or "").lower().split()),
        " ".join((title or "").lower().split()),
    )


def _select_for_web(
    rankings: list[dict],
    *,
    primary_count: int,
    exploration_count: int,
) -> list[dict]:
    primary = list(rankings[: max(primary_count, 0)])
    selected_keys = {record_key(item) for item in primary}

    exploration: list[dict] = []
    for item in rankings[max(primary_count, 0) :]:
        if len(exploration) >= max(exploration_count, 0):
            break
        if item.get("confidence") not in {"low", "medium"}:
            continue
        if float(item.get("score") or 0.0) <= 0:
            continue
        key = record_key(item)
        if key in selected_keys:
            continue
        exploration.append(item)
        selected_keys.add(key)

    return [*primary, *exploration]


def _with_research_session(func):
    @wraps(func)
    def run(**kwargs):
        with research_session() as session:
            result = func(**kwargs)
            from dataclasses import replace
            return replace(result, metrics=replace(result.metrics, search_calls=session.search_calls, page_fetch_calls=session.fetch_calls))
    return run


@_with_research_session
def run_opportunity_agent(
    *,
    student_profile: dict,
    jobs: list[dict],
    web_primary_count: int = 12,
    web_exploration_count: int = 3,
    semantic_shortlist_count: int = 5,
    related_company_count: int = 2,
    related_per_company: int = 1,
    progress=None,
) -> OpportunityAgentResult:
    """Rank broadly first, then enrich only a small high-value shortlist."""
    records = [job for raw in jobs if is_matching_candidate(job := _record(raw))]
    payloads = [_matching_payload(job) for job in records]
    initial_rankings = [item.to_dict() for item in rank_jobs(student_profile, payloads)]
    web_selection = _select_for_web(
        initial_rankings,
        primary_count=web_primary_count,
        exploration_count=web_exploration_count,
    )
    selected_keys = {record_key(item) for item in web_selection}

    updated_by_key: dict[str, JobRecord] = {}
    links_resolved = 0
    official_links = 0
    secondary_links = 0
    full_jd = 0
    partial_jd = 0

    selected_records = [job for job in records if job.record_id in selected_keys]
    for index, job in enumerate(selected_records, start=1):
        if progress:
            progress(f"      [WEB {index:02}/{len(selected_records):02}] {job.company} — {job.title}")

        resolved, resolution = resolve_job_link(job)
        if resolution.url:
            links_resolved += 1
            if resolution.kind.startswith("official"):
                official_links += 1
            elif resolution.kind.startswith("secondary"):
                secondary_links += 1
            if progress:
                progress(f"          link -> {resolution.kind} | {resolution.url}")
        elif progress:
            progress("          link -> unresolved")

        enriched = enrich_job_description(resolved)
        if enriched.jd_status in {"fetched_official", "fetched_secondary"}:
            full_jd += 1
            if progress:
                progress(f"          JD   -> full ({enriched.jd_status})")
        elif enriched.jd_status in {"partial_official", "partial_secondary"}:
            partial_jd += 1
            if progress:
                progress(f"          JD   -> partial ({enriched.jd_status})")
        elif progress:
            progress("          JD   -> source/title evidence")
        updated_by_key[job.record_id] = enriched

    final_records = [updated_by_key.get(job.record_id, job) for job in records]
    final_payloads = [_matching_payload(job) for job in final_records if is_matching_candidate(job)]
    reranked = [item.to_dict() for item in rank_jobs(student_profile, final_payloads)]
    semantic_shortlist = reranked[: max(semantic_shortlist_count, 0)]

    related_records, related_metrics = discover_related_jobs(
        top_rankings=reranked,
        student_profile=student_profile,
        existing_jobs=final_payloads,
        max_companies=related_company_count,
        per_company=related_per_company,
        main_shortlist_count=max(semantic_shortlist_count, 0),
    )
    enriched_related = []
    for job in related_records:
        if job.link_verification_status != "verified":
            job, _ = resolve_job_link(job)
        if is_matching_candidate(job):
            enriched_related.append(enrich_job_description(job))
    related_payloads = [_matching_payload(job) for job in enriched_related]
    related_rankings = [item.to_dict() for item in rank_jobs(student_profile, related_payloads)]

    metrics = OpportunityAgentMetrics(
        active_jobs=len(final_payloads),
        web_selected=len(selected_records),
        links_resolved=links_resolved,
        official_links=official_links,
        secondary_links=secondary_links,
        full_jd=full_jd,
        partial_jd=partial_jd,
        unresolved_links=max(len(selected_records) - links_resolved, 0),
        related_jobs_discovered=related_metrics.roles_discovered,
    )
    return OpportunityAgentResult(
        jobs=final_payloads,
        stage1_rankings=reranked,
        semantic_shortlist=semantic_shortlist,
        related_jobs=related_payloads,
        related_rankings=related_rankings,
        metrics=metrics,
    )
