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

    # "verified" means an official/public page was matched. "source_verified"
    # means a trusted NUS career source plus attached JD directly supports the role.
    verification_status: Literal[
        "verified",
        "source_verified",
        "partial",
        "unresolved",
    ] = "unresolved"
    verification_basis: Literal[
        "official_web",
        "trusted_email_attachment",
        "public_web",
        "none",
    ] = "none"
    evidence: list[str] = Field(default_factory=list)
