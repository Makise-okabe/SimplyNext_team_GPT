from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from career_agent.models.job_identity import JobIdentity
from career_agent.models.job_verification import SameJobVerificationResult

LivenessStatus = Literal["open", "closed", "unknown"]


class LivenessResult(BaseModel):
    url: str | None = None
    final_url: str | None = None
    status: LivenessStatus = "unknown"
    status_code: int | None = None
    reason: str = ""
    checked_at: str | None = None
    elapsed_ms: int = 0
    warning: str | None = None


class V4IdentityOutcome(BaseModel):
    identity: JobIdentity
    verification: SameJobVerificationResult
    liveness: LivenessResult
    verification_cache_hit: bool = False
    liveness_cache_hit: bool = False


class V4Metrics(BaseModel):
    source_identity_cache_hit: bool = False
    verification_cache_hits: int = 0
    verification_cache_misses: int = 0
    liveness_cache_hits: int = 0
    liveness_cache_misses: int = 0

    signal_llm_calls: int = 0
    identity_llm_calls: int = 0
    search_calls: int = 0
    verification_fetch_calls: int = 0
    judge_llm_calls: int = 0
    liveness_fetch_calls: int = 0
    total_llm_calls: int = 0

    approx_llm_source_chars: int = 0
    elapsed_ms: int = 0


class V4EmailResult(BaseModel):
    outcomes: list[V4IdentityOutcome] = Field(default_factory=list)
    metrics: V4Metrics = Field(default_factory=V4Metrics)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
