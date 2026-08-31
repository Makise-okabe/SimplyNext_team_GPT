from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class MatchResult(BaseModel):
    company: str | None = None
    title: str | None = None
    score: int
    recommendation: Literal["strong_match", "possible_match", "weak_match"]
    matched_strengths: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    rationale: str = ""
    jd_source_url: str | None = None
    primary_source_url: str | None = None
    secondary_source_url: str | None = None
