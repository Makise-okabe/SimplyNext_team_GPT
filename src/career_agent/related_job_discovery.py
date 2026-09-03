from __future__ import annotations

import re
from dataclasses import dataclass

from career_agent.hybrid_matching import infer_title_skills
from career_agent.job_link_resolver import _looks_job_like
from career_agent.job_research_quality import is_plausible_official_url
from career_agent.models.job_record import JobRecord
from career_agent.stage1_ranking import rank_job
from career_agent.tools.web_search import search_public_web

TITLE_CLEAN = re.compile(r"\s+[-|–—]\s+.*$")
GENERIC_TITLES = {
    "honor",
    "honor singapore",
    "product",
    "products",
    "cakes",
    "careers",
    "jobs",
    "home",
    "about us",
}
JOB_TITLE_TERMS = (
    "engineer",
    "intern",
    "developer",
    "scientist",
    "analyst",
    "architect",
    "designer",
    "research",
    "manager",
    "specialist",
    "associate",
    "consultant",
    "graduate",
    "technician",
    "operations",
)


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


def _looks_like_job_title(title: str) -> bool:
    normalized = _normalize(title)
    if len(normalized) < 4 or normalized in GENERIC_TITLES:
        return False
    return any(term in normalized for term in JOB_TITLE_TERMS)


def _job_key(company: str | None, title: str | None) -> tuple[str, str]:
    return _normalize(company), _normalize(title)


def _same_company_existing_roles(
    *,
    company: str,
    anchor_title: str,
    student_profile: dict,
    existing_jobs: list[dict],
    per_company: int,
) -> list[JobRecord]:
    candidates: list[tuple[float, dict]] = []
    anchor_key = _job_key(company, anchor_title)
    for raw in existing_jobs:
        if _normalize(raw.get("company")) != _normalize(company):
            continue
        if _job_key(raw.get("company"), raw.get("title")) == anchor_key:
            continue
        if str(raw.get("availability_status") or "") in {"expired_by_source_deadline", "closed_by_official"}:
            continue
        title = str(raw.get("title") or "").strip()
        if not title:
            continue
        try:
            ranked = rank_job(student_profile, raw)
            score = float(ranked.score)
        except Exception:
            score = 0.0
        if score <= 0:
            continue
        candidates.append((score, raw))

    candidates.sort(key=lambda item: (-item[0], _normalize(item[1].get("title"))))
    results: list[JobRecord] = []
    for _, raw in candidates[: max(per_company, 0)]:
        clean = dict(raw)
        for key in (
            "matching_ready",
            "matching_candidate",
            "matching_evidence_level",
            "matching_input_text",
        ):
            clean.pop(key, None)
        try:
            record = JobRecord.model_validate(clean)
        except Exception:
            continue
        results.append(
            record.model_copy(
                update={
                    "research_basis": "related_role_from_existing_email_jobs",
                    "evidence_summary": list(
                        dict.fromkeys(
                            [
                                *record.evidence_summary,
                                "same-company role already present in trusted career-email dataset",
                            ]
                        )
                    ),
                }
            )
        )
    return results


def discover_related_jobs(
    *,
    top_rankings: list[dict],
    student_profile: dict,
    existing_jobs: list[dict],
    max_companies: int = 4,
    per_company: int = 2,
) -> tuple[list[JobRecord], RelatedDiscoveryMetrics]:
    """Recommend same-company alternatives, then lightly search official roles if needed."""
    existing_keys = {_job_key(job.get("company"), job.get("title")) for job in existing_jobs}
    companies: list[tuple[str, str]] = []
    for item in top_rankings:
        company = str(item.get("company") or "").strip()
        title = str(item.get("title") or "").strip()
        if company and all(_normalize(company) != _normalize(existing[0]) for existing in companies):
            companies.append((company, title))
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
    ][:3]
    skill_query = " ".join(priority_terms) or "engineer"

    discovered: list[JobRecord] = []
    results_seen = 0
    companies_searched = 0

    for company, anchor_title in companies:
        same_company = _same_company_existing_roles(
            company=company,
            anchor_title=anchor_title,
            student_profile=student_profile,
            existing_jobs=existing_jobs,
            per_company=per_company,
        )
        for record in same_company:
            key = _job_key(record.company, record.title)
            if key in {_job_key(item.company, item.title) for item in discovered}:
                continue
            discovered.append(record)

        remaining = max(per_company - len(same_company), 0)
        if remaining <= 0:
            continue

        companies_searched += 1
        query = f'"{company}" careers {skill_query} engineer intern developer'
        try:
            results = search_public_web(query, max_results=8)
        except Exception:
            results = []
        results_seen += len(results)

        added = 0
        for result in results:
            if not is_plausible_official_url(result.url, company):
                continue
            if not _looks_job_like(result.url):
                continue

            title = _extract_title(result.title, company)
            if not _looks_like_job_title(title):
                continue

            key = _job_key(company, title)
            if key in existing_keys or key in {_job_key(item.company, item.title) for item in discovered}:
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
                evidence_summary=["discovered from a concrete official company/ATS job page"],
            )
            discovered.append(record)
            added += 1
            if added >= remaining:
                break

    return discovered, RelatedDiscoveryMetrics(
        companies_searched=companies_searched,
        results_seen=results_seen,
        roles_discovered=len(discovered),
    )
