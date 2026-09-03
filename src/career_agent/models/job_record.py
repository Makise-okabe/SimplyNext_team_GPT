from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from career_agent.models.opportunity_research import RecordKind, ResearchStatus
from career_agent.models.signal import OpportunitySignal, OpportunityType


JDStatus = Literal[
    "fetched_official",
    "fetched_secondary",
    "partial_official",
    "partial_secondary",
    "source_context_only",
    "unavailable",
]
AvailabilityStatus = Literal[
    "active_candidate",
    "expired_by_source_deadline",
    "closed_by_official",
    "unknown",
]
JobPageKind = Literal[
    "official_exact",
    "official_probable",
    "secondary_exact",
    "secondary_probable",
    "company_careers",
    "unresolved",
]
JobPageConfidence = Literal["high", "medium", "low"]


class SourceDocument(BaseModel):
    label: str
    source_type: Literal["email", "attachment", "linked_pdf"]
    url: str | None = None
    text_chars: int = 0


class JobRecord(BaseModel):
    """Canonical job contract consumed by the career-opportunity agent.

    Page resolution and JD evidence are intentionally separate. A role can have a
    useful clickable job page even when the page is dynamic, protected, or too
    sparse to yield a full job description.
    """

    source_key: Literal[
        "goh_ze_li",
        "talentconnect",
        "web_discovered",
        "unknown",
    ] = "unknown"
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

    # Legacy/source-hierarchy fields retained for backward compatibility.
    primary_source_url: str | None = None
    secondary_source_url: str | None = None
    official_job_url: str | None = None
    application_url: str | None = None

    # UI-facing page resolution. This is independent from JD extraction.
    job_page_url: str | None = None
    job_page_kind: JobPageKind = "unresolved"
    job_page_confidence: JobPageConfidence = "low"

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
