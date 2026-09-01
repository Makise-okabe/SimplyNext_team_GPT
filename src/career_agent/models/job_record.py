from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from career_agent.models.opportunity_research import RecordKind, ResearchStatus
from career_agent.models.signal import OpportunitySignal, OpportunityType


JDStatus = Literal[
    "fetched_official",
    "fetched_secondary",
    "source_context_only",
    "unavailable",
]
AvailabilityStatus = Literal[
    "active_candidate",
    "expired_by_source_deadline",
    "closed_by_official",
    "unknown",
]


class SourceDocument(BaseModel):
    label: str
    source_type: Literal["email", "attachment", "linked_pdf"]
    url: str | None = None
    text_chars: int = 0


class JobRecord(BaseModel):
    """Research-complete job contract consumed by the matching agent."""

    # Canonical source tag lets the later matching/ranking agent keep provenance
    # without reopening the original Outlook message.
    source_key: Literal["goh_ze_li", "talentconnect", "unknown"] = "unknown"
    source_message_id: str
    source_sender_email: str | None = None
    source_subject: str

    company: str | None = None
    title: str | None = None
    location: str | None = None
    opportunity_type: OpportunityType = "unknown"
    deadline_hint: date | None = None
    availability_status: AvailabilityStatus = "unknown"
    research_skipped_reason: str | None = None
    target_major: list[str] = Field(default_factory=list)
    target_degree_level: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)

    record_kind: RecordKind = "unknown"
    research_status: ResearchStatus = "unresolved"
    research_confidence: Literal["high", "medium", "low"] = "low"
    research_basis: str = "none"

    # Source hierarchy for downstream matching/explanations. The primary source
    # is always employer/ATS when known. A secondary mirror may supply the JD when
    # an official page is dynamic or otherwise unreadable.
    primary_source_url: str | None = None
    secondary_source_url: str | None = None
    official_job_url: str | None = None
    application_url: str | None = None

    jd_status: JDStatus = "unavailable"
    jd_source_url: str | None = None
    jd_text: str = ""

    source_evidence: str = ""
    evidence_summary: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class EmailOpportunityResearchResult(BaseModel):
    source_key: str
    source_message_id: str
    source_subject: str
    source_documents: list[SourceDocument] = Field(default_factory=list)
    opportunities: list[OpportunitySignal] = Field(default_factory=list)
    company_count: int = 0
    job_records: list[JobRecord] = Field(default_factory=list)

    extraction_llm_calls: int = 0
    extraction_source_chars: int = 0
    web_search_calls: int = 0
    page_fetch_calls: int = 0
    judge_llm_calls: int = 0

    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
