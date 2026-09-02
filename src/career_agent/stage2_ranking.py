from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Literal

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

MAX_RESUME_CHARS = 12000
MAX_JOB_EVIDENCE_CHARS = 3500
LLM_TIMEOUT_SECONDS = 45.0
LLM_MAX_RETRIES = 1
LLM_MODEL = "openai/gpt-oss-120b"


class SemanticAssessment(BaseModel):
    candidate_index: int = Field(ge=0)
    semantic_score: float = Field(ge=0, le=100)
    fit_label: Literal["strong", "good", "possible", "weak"]
    why_match: str
    matched_evidence: list[str] = Field(default_factory=list)
    missing_or_weak_evidence: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]


class SemanticAssessmentBatch(BaseModel):
    assessments: list[SemanticAssessment] = Field(default_factory=list)


@dataclass(frozen=True)
class Stage2RankedJob:
    candidate_index: int
    company: str
    title: str
    final_score: float
    semantic_score: float
    stage1_score: float
    fit_label: str
    confidence: str
    evidence_level: str
    why_match: str
    matched_evidence: tuple[str, ...]
    missing_or_weak_evidence: tuple[str, ...]
    matched_resume_skills: tuple[str, ...]
    matched_course_skills: tuple[str, ...]
    inferred_job_skills: tuple[str, ...]
    official_job_url: str | None
    application_url: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def _build_llm():
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is missing from .env")
    return ChatGroq(
        model=LLM_MODEL,
        temperature=0,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
    ).with_structured_output(
        SemanticAssessmentBatch,
        method="json_schema",
        strict=False,
    )


def _course_summary(student_profile: dict) -> list[dict]:
    items = []
    for course in student_profile.get("course_enrichment") or []:
        if not isinstance(course, dict):
            continue
        items.append(
            {
                "module_code": course.get("module_code"),
                "title": course.get("title"),
                "skills": course.get("skills") or [],
            }
        )
    return items


def _job_evidence(job: dict) -> str:
    evidence_level = str(job.get("matching_evidence_level") or "source_only")
    if evidence_level == "full_jd" and str(job.get("jd_text") or "").strip():
        text = str(job.get("jd_text") or "")
    else:
        text = str(job.get("source_evidence") or "").strip()
        if not text:
            text = str(job.get("matching_input_text") or "").strip()
    return text[:MAX_JOB_EVIDENCE_CHARS]


def build_stage2_prompt(
    *,
    resume_text: str,
    student_profile: dict,
    stage1_top: list[dict],
    source_jobs: list[dict],
) -> str:
    if len(stage1_top) != len(source_jobs):
        raise ValueError("stage1_top and source_jobs must have the same length")

    candidates = []
    for index, (ranked, job) in enumerate(zip(stage1_top, source_jobs)):
        candidates.append(
            {
                "candidate_index": index,
                "company": ranked.get("company") or job.get("company"),
                "title": ranked.get("title") or job.get("title"),
                "stage1_score": ranked.get("score", 0),
                "stage1_confidence": ranked.get("confidence"),
                "job_evidence_level": job.get("matching_evidence_level", "source_only"),
                "matched_resume_skills": ranked.get("matched_resume_skills") or [],
                "matched_course_skills": ranked.get("matched_course_skills") or [],
                "inferred_job_skills": ranked.get("inferred_job_skills") or [],
                "location": job.get("location"),
                "opportunity_type": job.get("opportunity_type"),
                "target_major": job.get("target_major") or [],
                "target_degree_level": job.get("target_degree_level") or [],
                "job_evidence": _job_evidence(job),
            }
        )

    student = {
        "explicit_skills": student_profile.get("explicit_skills") or [],
        "course_derived_skills": student_profile.get("course_derived_skills") or [],
        "courses": _course_summary(student_profile),
        "resume_text": (resume_text or "")[:MAX_RESUME_CHARS],
    }

    return f"""
You are the final semantic career-matching judge for an NUS student.

Evaluate ALL candidates below. Return exactly one assessment for every candidate_index, with no duplicates.
Return one JSON object with this root shape: {{"assessments": [ ... ]}}.
Do not discover new jobs and do not browse the web.

EVIDENCE RULES:
1. Student resume facts are strongest evidence of demonstrated experience.
2. Course-derived skills are supporting evidence, not proof of professional mastery.
3. A full JD is stronger job evidence than email/source context.
4. `inferred_job_skills` are only hypotheses inferred from the title. Never present them as employer-stated requirements.
5. Do not invent skills, requirements, achievements, preferences, or experience.
6. Generic overlaps such as communication, leadership, or data analysis must not outweigh a clear role-family mismatch.
7. Judge role-family fit first: semiconductor/electronics/embedded/software/AI/etc. A marketing or accounting role should not score highly merely because the student has communication or data skills.
8. Seniority matters. Penalize clearly senior roles if the resume does not support that seniority.
9. If job evidence is sparse, the candidate may still be a good match, but confidence must be lower.
10. Use the whole resume, projects, internships and courses. Do not rely only on the Stage-1 matched-skill list.
11. For `source_only` jobs, NEVER invent employer requirements from general industry knowledge. Only treat requirements explicitly present in `job_evidence` as employer requirements.
12. If a `source_only` job lacks enough requirements to identify a specific gap, say that the job requirements are unavailable or evidence is sparse. Do not name speculative certifications, standards, tools, years of experience, or domain requirements.
13. `missing_or_weak_evidence` must be grounded either in explicit job evidence or in an obvious property of the title itself, such as a role explicitly labelled Senior. Do not speculate beyond that.

SCORING CALIBRATION:
90-100: unusually strong fit with multiple direct resume/project/course signals and little role mismatch.
80-89: strong fit; several relevant signals, some gaps or sparse job evidence.
65-79: plausible/good fit but notable gaps, uncertainty, or adjacent role family.
45-64: possible but weak/partial fit.
0-44: poor fit or mostly generic overlap.

For `matched_evidence`, write short concrete evidence items supported by the resume or courses.
For `missing_or_weak_evidence`, identify only grounded gaps or evidence uncertainty.
Keep `why_match` concise (1-3 sentences).

STUDENT:
{json.dumps(student, ensure_ascii=False)}

CANDIDATES:
{json.dumps(candidates, ensure_ascii=False)}
""".strip()


def _find_source_job(ranked: dict, jobs: list[dict]) -> dict:
    company = str(ranked.get("company") or "").strip().lower()
    title = str(ranked.get("title") or "").strip().lower()
    matches = [
        job
        for job in jobs
        if str(job.get("company") or "").strip().lower() == company
        and str(job.get("title") or "").strip().lower() == title
    ]
    return matches[0] if matches else {}


def rerank_stage1(
    *,
    resume_text: str,
    student_profile: dict,
    all_jobs: list[dict],
    stage1_rankings: list[dict],
    stage1_top_n: int = 20,
    llm=None,
) -> list[Stage2RankedJob]:
    selected = list(stage1_rankings[: max(stage1_top_n, 0)])
    source_jobs = [_find_source_job(item, all_jobs) for item in selected]
    prompt = build_stage2_prompt(
        resume_text=resume_text,
        student_profile=student_profile,
        stage1_top=selected,
        source_jobs=source_jobs,
    )

    model = llm or _build_llm()
    batch = model.invoke(prompt)
    if isinstance(batch, dict):
        batch = SemanticAssessmentBatch.model_validate(batch)

    by_index: dict[int, SemanticAssessment] = {}
    for assessment in batch.assessments:
        if 0 <= assessment.candidate_index < len(selected):
            by_index.setdefault(assessment.candidate_index, assessment)

    results: list[Stage2RankedJob] = []
    for index, (ranked, job) in enumerate(zip(selected, source_jobs)):
        assessment = by_index.get(index)
        stage1_score = float(ranked.get("score") or 0.0)

        if assessment is None:
            semantic_score = stage1_score
            final_score = round(stage1_score, 1)
            fit_label = "possible"
            confidence = "low"
            why_match = "Semantic assessment was unavailable; retained the Stage-1 evidence score."
            matched_evidence: tuple[str, ...] = ()
            gaps = ("LLM semantic assessment missing",)
        else:
            semantic_score = float(assessment.semantic_score)
            final_score = round(0.8 * semantic_score + 0.2 * stage1_score, 1)
            fit_label = assessment.fit_label
            confidence = assessment.confidence
            why_match = assessment.why_match.strip()
            matched_evidence = tuple(assessment.matched_evidence[:8])
            gaps = tuple(assessment.missing_or_weak_evidence[:8])

        results.append(
            Stage2RankedJob(
                candidate_index=index,
                company=str(ranked.get("company") or job.get("company") or ""),
                title=str(ranked.get("title") or job.get("title") or ""),
                final_score=final_score,
                semantic_score=round(semantic_score, 1),
                stage1_score=round(stage1_score, 1),
                fit_label=fit_label,
                confidence=confidence,
                evidence_level=str(job.get("matching_evidence_level") or ranked.get("evidence_level") or "source_only"),
                why_match=why_match,
                matched_evidence=matched_evidence,
                missing_or_weak_evidence=gaps,
                matched_resume_skills=tuple(ranked.get("matched_resume_skills") or ()),
                matched_course_skills=tuple(ranked.get("matched_course_skills") or ()),
                inferred_job_skills=tuple(ranked.get("inferred_job_skills") or ()),
                official_job_url=job.get("official_job_url") or job.get("primary_source_url") or ranked.get("official_job_url"),
                application_url=job.get("application_url") or ranked.get("application_url"),
            )
        )

    results.sort(
        key=lambda item: (
            -item.final_score,
            {"high": 0, "medium": 1, "low": 2}.get(item.confidence, 3),
            item.company.lower(),
            item.title.lower(),
        )
    )
    return results
