from __future__ import annotations

import re
from dataclasses import dataclass

from career_agent.hybrid_matching import infer_title_skills
from career_agent.job_research_quality import is_plausible_official_url
from career_agent.models.job_record import JobRecord
from career_agent.tools.web_search_aggregate import search_public_web_aggregated

TITLE_CLEAN = re.compile(r"\s+[-|–—]\s+.*$")


@dataclass(frozen=True)
class RelatedDiscoveryMetrics:
    companies_searched: int
    results_seen: int
    roles_discovered: int


def _normalize(value: str | None) -> str:
    return " ".join((value or "").lower().split())


def _extract_title(result_title: str, company: str) -> str:
    value = result_title.strip()
    value = re.sub(re.escape(company), "", value, flags=re.I).strip(" -|–—")
    value = TITLE_CLEAN.sub("", value).strip()
    value = re.sub(r"\b(careers?|jobs?)\b", "", value, flags=re.I).strip(" -|–—")
    return " ".join(value.split())


def discover_related_jobs(
    *,
    top_rankings: list[dict],
    student_profile: dict,
    existing_jobs: list[dict],
    max_companies: int = 4,
    per_company: int = 2,
) -> tuple[list[JobRecord], RelatedDiscoveryMetrics]:
    """Find a few additional official roles from companies already ranking well."""
    existing_keys = {
        (_normalize(job.get("company")), _normalize(job.get("title"))) for job in existing_jobs
    }
    companies: list[str] = []
    for item in top_rankings:
        company = str(item.get("company") or "").strip()
        if company and company not in companies:
            companies.append(company)
        if len(companies) >= max_companies:
            break

    student_skills = {
        str(skill).lower()
        for skill in [
            *(student_profile.get("explicit_skills") or []),
            *(student_profile.get("course_derived_skills") or []),
        ]
    }
    priority_terms = [
        skill
        for skill in (
            "semiconductor",
            "embedded systems",
            "machine learning",
            "deep learning",
            "software engineering",
            "analog circuits",
            "eda/cadence",
            "computer vision",
        )
        if skill in student_skills
    ][:4]
    skill_query = " ".join(priority_terms) or "engineer"

    discovered: list[JobRecord] = []
    results_seen = 0
    for company in companies:
        query = f'"{company}" careers {skill_query}'
        results = search_public_web_aggregated(query, max_results=18, min_results=8)
        results_seen += len(results)
        added = 0
        for result in results:
            if not is_plausible_official_url(result.url, company):
                continue
            title = _extract_title(result.title, company)
            if len(title) < 4:
                continue
            key = (_normalize(company), _normalize(title))
            if key in existing_keys:
                continue
            title_skills = {skill.lower() for skill in infer_title_skills(title)}
            if title_skills and not (title_skills & student_skills):
                continue

            record = JobRecord(
                source_key="web_discovered",
                source_message_id=f"web:{_normalize(company)}",
                source_subject="Related role discovered from company careers",
                company=company,
                title=title,
                availability_status="unknown",
                opportunity_type="unknown",
                record_kind="job_posting",
                research_status="source_verified",
                research_confidence="medium",
                research_basis="related_company_role_discovery",
                primary_source_url=result.url,
                official_job_url=result.url,
                application_url=result.url,
                job_page_url=result.url,
                job_page_kind="official_probable",
                job_page_confidence="medium",
                source_urls=[result.url],
                source_evidence=f"Search result: {result.title}. {result.snippet}",
                evidence_summary=["discovered from an official company/ATS careers result"],
            )
            discovered.append(record)
            existing_keys.add(key)
            added += 1
            if added >= per_company:
                break

    return discovered, RelatedDiscoveryMetrics(
        companies_searched=len(companies),
        results_seen=results_seen,
        roles_discovered=len(discovered),
    )
