from __future__ import annotations

from dataclasses import asdict, dataclass

from career_agent.hybrid_matching import build_job_skill_profile


GENERIC_SKILLS = {
    "communication",
    "leadership",
    "stakeholder management",
}


@dataclass(frozen=True)
class Stage1RankedJob:
    company: str
    title: str
    score: float
    confidence: str
    evidence_level: str
    matched_resume_skills: tuple[str, ...]
    matched_course_skills: tuple[str, ...]
    missing_skills: tuple[str, ...]
    job_skills: tuple[str, ...]
    inferred_job_skills: tuple[str, ...]
    official_job_url: str | None
    application_url: str | None
    job_page_url: str | None
    job_page_kind: str
    job_page_confidence: str
    source_subject: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def _student_skill_weight(skill: str, explicit: set[str], course: set[str]) -> float:
    if skill in explicit:
        return 1.0
    if skill in course:
        return 0.75
    return 0.0


def _job_skill_weight(skill: str, inferred: set[str]) -> float:
    weight = 0.65 if skill in inferred else 1.0
    if skill in GENERIC_SKILLS:
        weight *= 0.35
    return weight


def _confidence(evidence_level: str, job_skill_count: int, inferred_count: int) -> str:
    if evidence_level == "full_jd":
        return "high"
    if evidence_level == "partial_jd":
        return "medium"
    explicit_job_skills = job_skill_count - inferred_count
    if explicit_job_skills >= 2 or job_skill_count >= 4:
        return "medium"
    return "low"


def _score_cap(matched: set[str], job_skills: set[str]) -> float:
    if not matched:
        return 0.0
    substantive = matched - GENERIC_SKILLS
    generic_only = not substantive
    if generic_only:
        return 28.0 if len(matched) == 1 else 40.0
    if len(substantive) == 1:
        return 68.0 if len(job_skills) <= 2 else 62.0
    if len(substantive) == 2:
        return 84.0
    if len(substantive) == 3:
        return 92.0
    return 100.0


def rank_job(student_profile: dict, job: dict) -> Stage1RankedJob:
    explicit = {str(skill).lower() for skill in student_profile.get("explicit_skills", [])}
    course = {str(skill).lower() for skill in student_profile.get("course_derived_skills", [])}
    job_profile = build_job_skill_profile(job)
    job_skills = {skill.lower() for skill in job_profile.skills}
    inferred = {skill.lower() for skill in job_profile.inferred_skills}
    matched_resume = sorted(skill for skill in job_skills if skill in explicit)
    matched_course = sorted(skill for skill in job_skills if skill not in explicit and skill in course)
    matched = set(matched_resume) | set(matched_course)
    missing = sorted(skill for skill in job_skills if skill not in explicit and skill not in course)

    denominator = sum(_job_skill_weight(skill, inferred) for skill in job_skills)
    substantive_job_skills = job_skills - GENERIC_SKILLS
    denominator = max(denominator, 1.8 if substantive_job_skills else 1.2)
    numerator = sum(
        _job_skill_weight(skill, inferred) * _student_skill_weight(skill, explicit, course)
        for skill in job_skills
        if _student_skill_weight(skill, explicit, course) > 0
    )
    coverage = numerator / denominator if denominator else 0.0
    substantive_matches = matched - GENERIC_SKILLS
    generic_matches = matched & GENERIC_SKILLS
    breadth_bonus = min(10.0, len(substantive_matches) * 2.2 + len(generic_matches) * 0.5)
    score = min(_score_cap(matched, job_skills), round(coverage * 90.0 + breadth_bonus, 1))

    evidence_level = str(job.get("matching_evidence_level") or job_profile.evidence_level)
    confidence = _confidence(evidence_level, len(job_skills), len(inferred))
    job_page_url = job.get("job_page_url") or job.get("application_url") or job.get("official_job_url") or job.get("secondary_source_url")

    return Stage1RankedJob(
        company=str(job.get("company") or ""),
        title=str(job.get("title") or ""),
        score=score,
        confidence=confidence,
        evidence_level=evidence_level,
        matched_resume_skills=tuple(matched_resume),
        matched_course_skills=tuple(matched_course),
        missing_skills=tuple(missing),
        job_skills=tuple(sorted(job_skills)),
        inferred_job_skills=tuple(sorted(inferred)),
        official_job_url=job.get("official_job_url") or job.get("primary_source_url"),
        application_url=job.get("application_url"),
        job_page_url=job_page_url,
        job_page_kind=str(job.get("job_page_kind") or "unresolved"),
        job_page_confidence=str(job.get("job_page_confidence") or "low"),
        source_subject=job.get("source_subject"),
    )


def rank_jobs(student_profile: dict, jobs: list[dict]) -> list[Stage1RankedJob]:
    ranked = [rank_job(student_profile, job) for job in jobs]
    ranked.sort(
        key=lambda item: (
            -item.score,
            {"high": 0, "medium": 1, "low": 2}.get(item.confidence, 3),
            item.company.lower(),
            item.title.lower(),
        )
    )
    return ranked
