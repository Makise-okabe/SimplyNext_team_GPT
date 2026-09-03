from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Callable, Literal

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

MAX_RESUME_CHARS = 6000
MAX_JOB_EVIDENCE_CHARS = 1800
MAX_RELEVANT_COURSES = 16
LLM_TIMEOUT_SECONDS = 45.0
LLM_MAX_RETRIES = 1
LLM_MODEL = "openai/gpt-oss-20b"
STAGE2_BATCH_SIZE = 10
STAGE2_RATE_LIMIT_ATTEMPTS = 3
STAGE2_RATE_LIMIT_FALLBACK_SECONDS = 5.0
STAGE2_MAX_RATE_LIMIT_WAIT_SECONDS = 20.0
MISSING_ASSESSMENT_SCORE_CAP = 60.0


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


class Stage2RateLimitUnavailable(RuntimeError):
    """Raised when the provider asks us to wait too long for an interactive run."""


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
    semantic_assessed: bool
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
    model_name = os.getenv("GROQ_STAGE2_MODEL", LLM_MODEL).strip() or LLM_MODEL
    return ChatGroq(
        model=model_name,
        temperature=0,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
    ).with_structured_output(
        SemanticAssessmentBatch,
        method="json_schema",
        strict=False,
    )


def _course_summary(student_profile: dict, relevant_skills: set[str] | None = None) -> list[dict]:
    items: list[dict] = []
    for course in student_profile.get("course_enrichment") or []:
        if not isinstance(course, dict):
            continue
        skills = [str(skill).lower() for skill in (course.get("skills") or []) if str(skill).strip()]
        if not skills:
            continue
        overlap = len(set(skills) & (relevant_skills or set()))
        items.append(
            {
                "module_code": course.get("module_code"),
                "title": course.get("title"),
                "skills": skills,
                "_overlap": overlap,
            }
        )

    items.sort(key=lambda item: (-int(item["_overlap"]), str(item.get("module_code") or "")))
    compact = []
    for item in items[:MAX_RELEVANT_COURSES]:
        compact.append(
            {
                "module_code": item.get("module_code"),
                "title": item.get("title"),
                "skills": item.get("skills") or [],
            }
        )
    return compact


def _job_evidence(job: dict) -> str:
    evidence_level = str(job.get("matching_evidence_level") or "source_only")
    if evidence_level in {"full_jd", "partial_jd"} and str(job.get("jd_text") or "").strip():
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

    relevant_course_skills: set[str] = set()
    candidates = []
    for index, (ranked, job) in enumerate(zip(stage1_top, source_jobs)):
        matched_course_skills = [str(skill).lower() for skill in (ranked.get("matched_course_skills") or [])]
        relevant_course_skills.update(matched_course_skills)
        candidates.append(
            {
                "candidate_index": index,
                "company": ranked.get("company") or job.get("company"),
                "title": ranked.get("title") or job.get("title"),
                "stage1_score": ranked.get("score", 0),
                "job_evidence_level": job.get("matching_evidence_level", "source_only"),
                "matched_resume_skills": ranked.get("matched_resume_skills") or [],
                "matched_course_skills": matched_course_skills,
                "inferred_job_skills": ranked.get("inferred_job_skills") or [],
                "location": job.get("location"),
                "opportunity_type": job.get("opportunity_type"),
                "job_evidence": _job_evidence(job),
            }
        )

    student = {
        "explicit_skills": student_profile.get("explicit_skills") or [],
        "course_derived_skills": student_profile.get("course_derived_skills") or [],
        "relevant_courses": _course_summary(student_profile, relevant_course_skills),
        "resume_text": (resume_text or "")[:MAX_RESUME_CHARS],
    }

    return f"""
You are the final semantic career-matching judge for an NUS student.
Evaluate ALL candidates and return exactly one assessment for every candidate_index.
Return one JSON object: {{"assessments": [ ... ]}}. Do not browse or discover jobs.

RULES:
- Resume facts are strongest evidence. Course-derived skills are supporting evidence.
- Job evidence strength: full JD > partial JD > trusted email/source-only context.
- `inferred_job_skills` are hypotheses from the title, never employer-stated requirements.
- Do not invent student skills, employer requirements, certifications, tools, or experience.
- Judge role-family fit first; generic communication/leadership/data overlap cannot rescue a role mismatch.
- Penalize explicit senior roles when the resume does not support that seniority.
- Sparse job evidence can still yield a good match, but confidence should be lower.
- For source-only/partial-JD jobs, only discuss requirements actually visible in the evidence.
- Keep why_match to 1-2 concise sentences and each evidence/gap item short.

SCORE:
90-100 unusually strong fit; 80-89 strong; 65-79 plausible/good; 45-64 weak/partial; 0-44 poor.

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


def _coerce_batch(value) -> SemanticAssessmentBatch:
    if isinstance(value, SemanticAssessmentBatch):
        return value
    if isinstance(value, dict):
        return SemanticAssessmentBatch.model_validate(value)
    return SemanticAssessmentBatch.model_validate(value)


def _assess_chunk(
    *,
    model,
    resume_text: str,
    student_profile: dict,
    ranked_chunk: list[dict],
    job_chunk: list[dict],
) -> dict[int, SemanticAssessment]:
    prompt = build_stage2_prompt(
        resume_text=resume_text,
        student_profile=student_profile,
        stage1_top=ranked_chunk,
        source_jobs=job_chunk,
    )
    batch = _coerce_batch(model.invoke(prompt))
    found: dict[int, SemanticAssessment] = {}
    for assessment in batch.assessments:
        if 0 <= assessment.candidate_index < len(ranked_chunk):
            found.setdefault(assessment.candidate_index, assessment)
    return found


def _is_rate_limit_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return "ratelimit" in name or "rate limit" in message or "rate_limit" in message


def _retry_after_seconds(exc: Exception, attempt: int) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        value = headers.get("retry-after") or headers.get("Retry-After")
        if value is not None:
            try:
                return max(float(value), 0.0)
            except (TypeError, ValueError):
                pass
    return STAGE2_RATE_LIMIT_FALLBACK_SECONDS * (2 ** max(attempt - 1, 0))


def _assess_chunk_with_rate_limit_retry(
    *,
    model,
    resume_text: str,
    student_profile: dict,
    ranked_chunk: list[dict],
    job_chunk: list[dict],
    sleep_fn: Callable[[float], None],
    show_progress: bool,
    progress_indent: str,
) -> dict[int, SemanticAssessment]:
    for attempt in range(1, STAGE2_RATE_LIMIT_ATTEMPTS + 1):
        try:
            return _assess_chunk(
                model=model,
                resume_text=resume_text,
                student_profile=student_profile,
                ranked_chunk=ranked_chunk,
                job_chunk=job_chunk,
            )
        except Exception as exc:
            if not _is_rate_limit_error(exc):
                raise

            delay = _retry_after_seconds(exc, attempt)
            if delay > STAGE2_MAX_RATE_LIMIT_WAIT_SECONDS:
                raise Stage2RateLimitUnavailable(
                    f"provider retry-after {delay:g}s exceeds interactive cap "
                    f"of {STAGE2_MAX_RATE_LIMIT_WAIT_SECONDS:g}s"
                ) from exc

            if attempt >= STAGE2_RATE_LIMIT_ATTEMPTS:
                raise Stage2RateLimitUnavailable("rate limit persisted after bounded retries") from exc

            if show_progress:
                print(
                    f"{progress_indent}rate limited; waiting {delay:g}s "
                    f"before attempt {attempt + 1}/{STAGE2_RATE_LIMIT_ATTEMPTS}..."
                )
            sleep_fn(delay)
    return {}


def rerank_stage1(
    *,
    resume_text: str,
    student_profile: dict,
    all_jobs: list[dict],
    stage1_rankings: list[dict],
    stage1_top_n: int = 10,
    llm=None,
    batch_size: int = STAGE2_BATCH_SIZE,
    show_progress: bool = False,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> list[Stage2RankedJob]:
    selected = list(stage1_rankings[: max(stage1_top_n, 0)])
    source_jobs = [_find_source_job(item, all_jobs) for item in selected]
    if not selected:
        return []

    model = llm or _build_llm()
    batch_size = max(1, int(batch_size))
    by_index: dict[int, SemanticAssessment] = {}
    total_batches = (len(selected) + batch_size - 1) // batch_size

    for batch_number, start in enumerate(range(0, len(selected), batch_size), start=1):
        ranked_chunk = selected[start : start + batch_size]
        job_chunk = source_jobs[start : start + batch_size]
        skip_individual_retries = False

        if show_progress:
            print(f"      Stage 2 batch {batch_number}/{total_batches}: requesting {len(ranked_chunk)} candidates...")

        try:
            found = _assess_chunk_with_rate_limit_retry(
                model=model,
                resume_text=resume_text,
                student_profile=student_profile,
                ranked_chunk=ranked_chunk,
                job_chunk=job_chunk,
                sleep_fn=sleep_fn,
                show_progress=show_progress,
                progress_indent="        ",
            )
        except Stage2RateLimitUnavailable as exc:
            found = {}
            skip_individual_retries = True
            if show_progress:
                print(
                    f"      Stage 2 batch {batch_number}/{total_batches}: "
                    f"rate-limit window too long; skipping retries ({exc})"
                )
        except Exception as exc:
            found = {}
            if show_progress:
                print(f"      Stage 2 batch {batch_number}/{total_batches}: failed ({type(exc).__name__})")

        for local_index, assessment in found.items():
            by_index.setdefault(start + local_index, assessment)

        missing_local = [index for index in range(len(ranked_chunk)) if index not in found]
        if show_progress:
            print(
                f"      Stage 2 batch {batch_number}/{total_batches}: "
                f"assessed {len(found)}/{len(ranked_chunk)}"
            )

        if skip_individual_retries:
            continue

        for retry_number, original_local in enumerate(missing_local, start=1):
            if show_progress:
                print(
                    f"        retry {retry_number}/{len(missing_local)} "
                    f"for candidate {start + original_local + 1}..."
                )
            try:
                retry_found = _assess_chunk_with_rate_limit_retry(
                    model=model,
                    resume_text=resume_text,
                    student_profile=student_profile,
                    ranked_chunk=[ranked_chunk[original_local]],
                    job_chunk=[job_chunk[original_local]],
                    sleep_fn=sleep_fn,
                    show_progress=show_progress,
                    progress_indent="          ",
                )
            except Stage2RateLimitUnavailable as exc:
                retry_found = {}
                if show_progress:
                    print(f"          rate-limit window too long; skipping ({exc})")
            except Exception as exc:
                retry_found = {}
                if show_progress:
                    print(f"          failed ({type(exc).__name__})")

            assessment = retry_found.get(0)
            if assessment is not None:
                by_index.setdefault(start + original_local, assessment)
                if show_progress:
                    print("          assessed")
            elif show_progress:
                print("          still missing")

    results: list[Stage2RankedJob] = []
    for index, (ranked, job) in enumerate(zip(selected, source_jobs)):
        assessment = by_index.get(index)
        stage1_score = float(ranked.get("score") or 0.0)

        if assessment is None:
            semantic_score = min(stage1_score, MISSING_ASSESSMENT_SCORE_CAP)
            final_score = round(semantic_score, 1)
            fit_label = "possible"
            confidence = "low"
            semantic_assessed = False
            why_match = "Semantic assessment was unavailable after retry; this candidate is not treated as a validated final match."
            matched_evidence: tuple[str, ...] = ()
            gaps = ("LLM semantic assessment missing after retry",)
        else:
            semantic_score = float(assessment.semantic_score)
            final_score = round(0.8 * semantic_score + 0.2 * stage1_score, 1)
            fit_label = assessment.fit_label
            confidence = assessment.confidence
            semantic_assessed = True
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
                semantic_assessed=semantic_assessed,
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
            not item.semantic_assessed,
            -item.final_score,
            {"high": 0, "medium": 1, "low": 2}.get(item.confidence, 3),
            item.company.lower(),
            item.title.lower(),
        )
    )
    return results