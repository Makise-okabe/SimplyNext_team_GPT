from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from career_agent.models.signal import OpportunityType

IdentifierType = Literal[
    "job_id",
    "requisition_id",
    "posting_id",
    "reference_number",
    "job_code",
    "position_id",
    "other",
]
IdentityStrength = Literal["strong", "moderate", "weak"]


class JobIdentifier(BaseModel):
    kind: IdentifierType
    label: str
    value: str


class JobIdentity(BaseModel):
    """Compact identity fingerprint used by V2/V3 same-job verification.

    This object deliberately stores identity attributes, not a full job
    description. The full email/PDF remains ephemeral in the graph state.
    """

    source_message_id: str
    signal_index: int

    company: str | None = None
    title: str | None = None
    identifiers: list[JobIdentifier] = Field(default_factory=list)
    location: str | None = None
    opportunity_type: OpportunityType = "unknown"

    business_unit: str | None = None
    team: str | None = None
    employment_type: str | None = None
    duration: str | None = None
    start_period: str | None = None
    end_period: str | None = None
    target_cohort: list[str] = Field(default_factory=list)

    distinctive_phrases: list[str] = Field(default_factory=list)
    direct_urls: list[str] = Field(default_factory=list)
    evidence_snippets: list[str] = Field(default_factory=list)

    identity_strength: IdentityStrength = "weak"
    source_fingerprint: str


class ExtractedJobIdentity(BaseModel):
    """Tolerant LLM-only schema before deterministic grounding."""

    source_index: int
    company: str | None = None
    title: str | None = None
    location: str | None = None
    business_unit: str | None = None
    team: str | None = None
    employment_type: str | None = None
    duration: str | None = None
    start_period: str | None = None
    end_period: str | None = None
    target_cohort: list[str] | None = None
    distinctive_phrases: list[str] | None = None


class ExtractedJobIdentityBatch(BaseModel):
    identities: list[ExtractedJobIdentity] = Field(default_factory=list)


class IdentityExtractionMetrics(BaseModel):
    signals_seen: int = 0
    identities_built: int = 0
    llm_calls: int = 0
    batches: int = 0
    source_chars_sent: int = 0
    elapsed_ms: int = 0


class JobIdentityExtractionResult(BaseModel):
    identities: list[JobIdentity] = Field(default_factory=list)
    metrics: IdentityExtractionMetrics = Field(default_factory=IdentityExtractionMetrics)
    errors: list[str] = Field(default_factory=list)
