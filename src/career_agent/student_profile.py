from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from career_agent.course_enrichment import CourseEnrichment, enrich_courses, extract_module_codes
from career_agent.hybrid_matching import extract_explicit_skills, normalize_skills


@dataclass(frozen=True)
class StudentProfile:
    explicit_skills: tuple[str, ...]
    course_derived_skills: tuple[str, ...]
    all_skills: tuple[str, ...]
    module_codes: tuple[str, ...]
    course_enrichment: tuple[CourseEnrichment, ...]
    resume_text_chars: int
    transcript_text_chars: int

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["course_enrichment"] = [asdict(item) for item in self.course_enrichment]
        return payload


def build_student_profile(
    *,
    resume_text: str,
    transcript_text: str,
    extra_skills: Iterable[str] = (),
    enrich_modules: bool = True,
) -> StudentProfile:
    explicit = extract_explicit_skills(resume_text) | normalize_skills(extra_skills)
    module_codes = extract_module_codes(transcript_text)

    course_items: list[CourseEnrichment] = []
    if enrich_modules:
        course_items = enrich_courses(module_codes)

    course_skills: set[str] = set()
    for item in course_items:
        course_skills.update(item.skills)

    all_skills = explicit | course_skills

    return StudentProfile(
        explicit_skills=tuple(sorted(explicit)),
        course_derived_skills=tuple(sorted(course_skills)),
        all_skills=tuple(sorted(all_skills)),
        module_codes=tuple(module_codes),
        course_enrichment=tuple(course_items),
        resume_text_chars=len(resume_text or ""),
        transcript_text_chars=len(transcript_text or ""),
    )
