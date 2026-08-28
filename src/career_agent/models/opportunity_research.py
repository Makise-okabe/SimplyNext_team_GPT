from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from career_agent.models.job_identity import JobIdentity

RecordKind = Literal[
    "job_posting",
    "programme",
    "recruitment_campaign",
    "event",
    "challenge",
    "unknown",
]
EvidenceTier = Literal["official", "institutional", "secondary", "weak"]
ResearchStatus = Literal[
    "verified_exact_job",
    "official_context_supported",
    "secondary_corroborated",
    "source_verified",
    "ambiguous",
    "unresolved",
]
ResearchRelation = Literal[
    "exact_posting",
    "official_background",
    "secondary_evidence",
    "application",
    "source_email",
]


class ResearchCandidate(BaseModel):
    url: str
    title: str = ""
    snippet: str = ""
    host: str = ""
    tier: EvidenceTier = "weak"
    relation: ResearchRelation = "secondary_evidence"
    score: float = 0.0
    reasons: list[str] = Field(default_factory=list)


class ResearchTraceStep(BaseModel):
    round_number: int
    scope: Literal["official", "secondary"]
    query: str
    results_returned: int = 0
    official_results: int = 0
    secondary_results: int = 0
    elapsed_ms: int = 0


class SourceProvenance(BaseModel):
    message_id: str
    subject: str
    sender_name: str | None = None
    sender_email: str | None = None
    received_at: datetime | None = None
    transport_sender_name: str | None = None
    transport_sender_email: str | None = None
    original_email_url: str | None = None
    attachment_names: list[str] = Field(default_factory=list)


class ResearchMetrics(BaseModel):
    search_calls: int = 0
    fetch_calls: int = 0
    judge_llm_calls: int = 0
    elapsed_ms: int = 0


class OpportunityResearchPackage(BaseModel):
    identity: JobIdentity
    record_kind: RecordKind = "unknown"
    status: ResearchStatus = "unresolved"
    confidence: Literal["high", "medium", "low"] = "low"
    basis: str = "none"

    provenance: SourceProvenance
    official_job_url: str | None = None
    official_background_urls: list[str] = Field(default_factory=list)
    secondary_evidence_urls: list[str] = Field(default_factory=list)
    application_url: str | None = None

    candidates: list[ResearchCandidate] = Field(default_factory=list)
    trace: list[ResearchTraceStep] = Field(default_factory=list)
    evidence_summary: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    metrics: ResearchMetrics = Field(default_factory=ResearchMetrics)
