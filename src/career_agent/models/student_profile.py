from __future__ import annotations

from pydantic import BaseModel, Field


class EducationRecord(BaseModel):
    institution: str | None = None
    degree: str | None = None
    major: str | None = None
    graduation_date: str | None = None
    gpa: str | None = None
    coursework: list[str] = Field(default_factory=list)


class ExperienceRecord(BaseModel):
    organization: str | None = None
    title: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    bullets: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)


class ProjectRecord(BaseModel):
    name: str
    description: str = ""
    skills: list[str] = Field(default_factory=list)


class StudentProfile(BaseModel):
    """Stable student-side input consumed by the job matching agent."""

    name: str | None = None
    summary: str = ""
    education: list[EducationRecord] = Field(default_factory=list)
    experiences: list[ExperienceRecord] = Field(default_factory=list)
    projects: list[ProjectRecord] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    preferred_roles: list[str] = Field(default_factory=list)
    preferred_locations: list[str] = Field(default_factory=list)
    work_authorization: list[str] = Field(default_factory=list)
    raw_resume_text: str = ""
    raw_transcript_text: str = ""
