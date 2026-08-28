from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CandidateDecision = Literal["same_job", "possible", "reject", "unreadable"]
IdentityStatus = Literal["verified", "source_verified", "ambiguous", "unresolved"]
IdentityBasis = Literal[
    "exact_identifier",
    "direct_official_url",
    "jd_content_match",
    "trusted_nus_attachment",
    "multiple_candidates",
    "none",
]
Confidence = Literal["high", "medium", "low"]


class CandidateEvaluation(BaseModel):
    requested_url: str
    final_url: str | None = None
    host: str = ""
    url_kind: str = "unknown"
    page_title: str = ""
    status_code: int | None = None

    evidence_score: float = 0.0
    decision: CandidateDecision = "unreadable"
    confidence: Confidence = "low"

    identifier_hits: list[str] = Field(default_factory=list)
    company_match: bool = False
    title_overlap: float = 0.0
    location_match: bool = False
    business_unit_match: bool = False
    distinctive_phrase_hits: list[str] = Field(default_factory=list)
    hard_conflicts: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    fetch_error: str | None = None


class SameJobJudgeOutput(BaseModel):
    decision: Literal["same_job", "ambiguous", "different"]
    candidate_url: str | None = None
    reason: str = ""
    supporting_evidence: list[str] = Field(default_factory=list)


class VerificationMetrics(BaseModel):
    fetch_calls: int = 0
    pages_fetched: int = 0
    llm_calls: int = 0
    elapsed_ms: int = 0


class SameJobVerificationResult(BaseModel):
    identity_status: IdentityStatus = "unresolved"
    identity_basis: IdentityBasis = "none"
    confidence: Confidence = "low"

    official_url: str | None = None
    application_url: str | None = None
    matched_candidate_url: str | None = None

    evaluations: list[CandidateEvaluation] = Field(default_factory=list)
    matched_evidence: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    metrics: VerificationMetrics = Field(default_factory=VerificationMetrics)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
