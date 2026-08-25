from datetime import date
from typing import Literal
from pydantic import BaseModel, Field, HttpUrl


class Job(BaseModel):
    company: str
    title: str
    location: str | None = None
    opportunity_type: Literal["internship", "full_time", "unknown"] = "unknown"

    official_url: HttpUrl | None = None
    deadline: date | None = None

    degree_requirements: list[str] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)

    visa_information: str | None = None
    raw_description: str = ""

    verification_status: Literal["verified", "partial", "unresolved"] = "unresolved"
    evidence: list[str] = Field(default_factory=list)
