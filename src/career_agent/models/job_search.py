from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SearchStrategy = Literal[
    "direct_url",
    "exact_identifier",
    "metadata",
    "distinctive_phrase",
]
UrlKind = Literal[
    "employer_or_ats",
    "application_form",
    "source_page",
    "aggregator",
    "unknown",
]


class SearchCandidate(BaseModel):
    url: str
    title: str = ""
    snippet: str = ""
    host: str
    url_kind: UrlKind = "unknown"

    discovery_score: float = 0.0
    strategies: list[SearchStrategy] = Field(default_factory=list)
    identifier_hits: list[str] = Field(default_factory=list)
    distinctive_phrase_hits: list[str] = Field(default_factory=list)
    metadata_hits: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class SearchTraceStep(BaseModel):
    round_number: int
    strategy: SearchStrategy
    query: str
    results_returned: int = 0
    candidates_after_merge: int = 0
    elapsed_ms: int = 0


class CandidateSearchMetrics(BaseModel):
    search_calls: int = 0
    raw_results_seen: int = 0
    unique_candidates: int = 0
    elapsed_ms: int = 0
    llm_calls: int = 0
    stopped_reason: str = ""


class CandidateDiscoveryResult(BaseModel):
    candidates: list[SearchCandidate] = Field(default_factory=list)
    trace: list[SearchTraceStep] = Field(default_factory=list)
    metrics: CandidateSearchMetrics = Field(default_factory=CandidateSearchMetrics)
    errors: list[str] = Field(default_factory=list)
