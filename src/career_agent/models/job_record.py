from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator

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
SearchResolutionStatus = Literal[
    "resolved_job_page",
    "search_fallback_only",
    "not_searched",
]


class SourceDocument(BaseModel):
    label: str
    source_type: Literal["email", "attachment", "linked_pdf"]
    url: str | None = None
    text_chars: int = 0


class JobRecord(BaseModel):
    """Canonical job contract consumed by the career-opportunity agent.

    A concrete job page, a search fallback, and JD evidence are intentionally
    separate. The UI can therefore remain useful even when a search provider
    cannot resolve a direct posting or a dynamic page cannot be scraped.
    """

    source_key: Literal[
        "goh_ze_li",
        "talentconnect",
        "web_discovered",
        "unknown",
    ] = "unknown"
    record_id: str = ""
    source_message_id: str
    source_sender_email: str | None = None
    source_subject: str

    company: str | None = None
    title: str | None = None
    industry: str | None = None
    talentconnect_id: str | None = None
    remarks: str = ""
    job_id: str | None = None
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

    # UI-facing concrete page resolution.
    job_page_url: str | None = None
    job_page_kind: JobPageKind = "unresolved"
    job_page_confidence: JobPageConfidence = "low"
    link_verification_status: str = "not_checked"
    link_checked_at: str | None = None
    link_verification_reason: str = ""
    link_attempts: list[dict] = Field(default_factory=list)
    company_careers_url: str | None = None
    candidate_job_url: str | None = None
    candidate_job_kind: Literal["official_candidate", "secondary_candidate"] | None = None
    candidate_job_reason: str = ""
    responsibilities: list[str] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    qualifications: list[str] = Field(default_factory=list)
    source_provenance: list[dict] = Field(default_factory=list)

    # Search fallback is explicitly NOT treated as a resolved job page. It lets
    # the UI offer "Find job" rather than showing a dead <unresolved> state.
    search_fallback_url: str | None = None
    search_resolution_status: SearchResolutionStatus = "not_searched"

    jd_status: JDStatus = "unavailable"
    jd_source_url: str | None = None
    jd_text: str = ""

    source_evidence: str = ""
    evidence_summary: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def assign_record_id(self):
        if not self.record_id:
            from career_agent.record_identity import record_key
            self.record_id = record_key(self.model_dump())
        return self


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
