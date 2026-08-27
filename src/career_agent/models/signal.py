from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


OpportunityType = Literal["internship", "full_time", "event", "unknown"]


class OpportunitySignal(BaseModel):
    source_type: Literal["outlook", "eml"]
    source_name: str
    source_message_id: str
    source_date: datetime | None = None

    company: str | None = None
    role_title: str | None = None
    location: str | None = None
    opportunity_type: OpportunityType = "unknown"
    deadline_hint: date | None = None

    target_major: list[str] = Field(default_factory=list)
    target_degree_level: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)

    raw_text: str = ""
    resolution_status: Literal["unresolved", "resolved", "needs_user_action"] = "unresolved"


class ExtractedOpportunity(BaseModel):
    """LLM-only schema before source metadata is attached deterministically."""

    company: str | None = None
    role_title: str | None = None
    location: str | None = None
    opportunity_type: OpportunityType = "unknown"
    deadline_hint: date | None = None
    target_major: list[str] = Field(default_factory=list)
    target_degree_level: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)
    evidence_text: str = ""


class ExtractedOpportunityBatch(BaseModel):
    opportunities: list[ExtractedOpportunity] = Field(default_factory=list)
