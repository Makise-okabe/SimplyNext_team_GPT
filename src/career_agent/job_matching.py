from __future__ import annotations

import os
from typing import Iterable

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from career_agent.models.job_record import JobRecord
from career_agent.models.match_result import MatchResult
from career_agent.models.student_profile import StudentProfile


def _build_llm():
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is missing from .env")
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
    ).with_structured_output(MatchResult)


def _recommendation_for_score(score: int) -> str:
    if score >= 85:
        return "strong_match"
    if score >= 65:
        return "possible_match"
    return "weak_match"


def match_job(profile: StudentProfile, job: JobRecord) -> MatchResult:
    """Evaluate one matching-ready JobRecord against one student profile."""
    if not job.jd_text.strip():
        raise ValueError("JobRecord has no JD text and is not suitable for matching")

    prompt = f"""
You are the SimplyNext job matching agent.

Assess how well the student's ACTUAL evidence matches this job description.
Do not invent skills, coursework, experience, grades, citizenship, or preferences.
Do not penalize the student for requirements that are not stated in the JD.
Distinguish between:
- demonstrated evidence in resume/projects/transcript
- reasonable transferable evidence
- genuinely missing or unsupported requirements

Scoring rubric (0-100):
- 85-100: strong_match — most important requirements are directly supported
- 65-84: possible_match — meaningful fit, but some important gaps remain
- 0-64: weak_match — major requirements are missing or unsupported

The rationale should be concise and evidence-based. Missing requirements must be
requirements actually present in the JD, not generic career advice.

STUDENT PROFILE:
{profile.model_dump_json(indent=2)}

JOB:
Company: {job.company or ''}
Title: {job.title or ''}
Location: {job.location or ''}
Type: {job.opportunity_type}
JD source: {job.jd_source_url or ''}

JOB DESCRIPTION:
{job.jd_text}
""".strip()

    result = _build_llm().invoke(prompt)
    score = max(0, min(100, result.score))
    # URLs and recommendation are deterministic outputs; the LLM only evaluates fit.
    return result.model_copy(
        update={
            "score": score,
            "recommendation": _recommendation_for_score(score),
            "company": job.company,
            "title": job.title,
            "jd_source_url": job.jd_source_url,
            "primary_source_url": job.primary_source_url,
            "secondary_source_url": job.secondary_source_url,
        }
    )


def rank_jobs(
    profile: StudentProfile,
    jobs: Iterable[JobRecord],
) -> list[MatchResult]:
    results = [match_job(profile, job) for job in jobs]
    return sorted(results, key=lambda item: item.score, reverse=True)
