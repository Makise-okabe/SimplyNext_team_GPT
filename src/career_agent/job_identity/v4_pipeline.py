from __future__ import annotations

import math
import time

from career_agent.job_identity.discover_candidates import discover_candidates
from career_agent.job_identity.extract_identity import extract_job_identities
from career_agent.job_identity.liveness import check_liveness
from career_agent.job_identity.verify_same_job import verify_same_job
from career_agent.models.email import EmailMessage
from career_agent.models.job_v4 import (
    LivenessResult,
    V4EmailResult,
    V4IdentityOutcome,
    V4Metrics,
)
from career_agent.nodes.extract_signal import (
    MAX_CANDIDATES_PER_LLM_CALL,
    build_candidate_chunks,
    extract_signal,
)
from career_agent.storage.verification_cache import (
    VerificationCacheStore,
    verification_cache_key,
)


def _signal_input_stats(email: EmailMessage) -> tuple[int, int, int]:
    candidates = build_candidate_chunks(email.body_text or "", email.links or [])
    source_chars = sum(len(candidate.context) + len(candidate.url) for candidate in candidates)
    if (email.attachment_text or "").strip():
        # Mirrors the direct attachment candidate cap in extract_signal.py.
        attachment_chars = min(len(email.attachment_text), 7000)
        source_chars += attachment_chars
        candidate_count = len(candidates) + 1
    else:
        candidate_count = len(candidates)

    calls = math.ceil(candidate_count / MAX_CANDIDATES_PER_LLM_CALL) if candidate_count else 0
    return candidate_count, calls, source_chars


def process_email_v4(
    email: EmailMessage,
    cache: VerificationCacheStore,
    force_refresh: bool = False,
) -> V4EmailResult:
    """Run V1->V3 with cycle-safe cache, then refresh liveness when needed."""
    started = time.perf_counter()
    warnings: list[str] = []
    errors: list[str] = []

    source_cache_hit = False
    signal_llm_calls = 0
    identity_llm_calls = 0
    approx_llm_source_chars = 0

    identities = None if force_refresh else cache.get_source_identities(email.message_id)
    if identities is not None:
        source_cache_hit = True
    else:
        _, estimated_signal_calls, signal_source_chars = _signal_input_stats(email)
        state = {
            "email": email.model_dump(mode="json"),
            "normalized_text": email.body_text,
            "extracted_links": email.links,
            "errors": [],
        }
        signal_result = extract_signal(state)
        signal_llm_calls = estimated_signal_calls
        approx_llm_source_chars += signal_source_chars
        errors.extend(signal_result.get("errors", []))

        identity_result = extract_job_identities(
            email,
            signal_result.get("opportunity_signals", []),
        )
        identities = identity_result.identities
        identity_llm_calls = identity_result.metrics.llm_calls
        approx_llm_source_chars += identity_result.metrics.source_chars_sent
        errors.extend(identity_result.errors)
        cache.save_source_identities(email.message_id, identities)

    outcomes: list[V4IdentityOutcome] = []
    verification_hits = 0
    verification_misses = 0
    liveness_hits = 0
    liveness_misses = 0
    search_calls = 0
    verification_fetch_calls = 0
    judge_llm_calls = 0
    liveness_fetch_calls = 0

    for identity in identities or []:
        key = verification_cache_key(identity, email)
        verification = None if force_refresh else cache.get_verification(key)
        verification_cache_hit = verification is not None

        if verification is not None:
            verification_hits += 1
        else:
            verification_misses += 1
            discovery = discover_candidates(identity)
            verification = verify_same_job(identity, discovery, email)
            search_calls += discovery.metrics.search_calls
            verification_fetch_calls += verification.metrics.fetch_calls
            judge_llm_calls += verification.metrics.llm_calls
            warnings.extend(verification.warnings)
            errors.extend(verification.errors)
            cache.save_verification(key, verification)

        official_url = verification.official_url
        liveness_cache_hit = False
        if official_url:
            liveness = None if force_refresh else cache.get_liveness(official_url)
            if liveness is not None:
                liveness_hits += 1
                liveness_cache_hit = True
            else:
                liveness_misses += 1
                liveness_fetch_calls += 1
                liveness = check_liveness(official_url)
                if liveness.warning:
                    warnings.append(
                        f"liveness unavailable for {official_url}: {liveness.warning}"
                    )
                cache.save_liveness(liveness)
        else:
            liveness = LivenessResult(
                url=None,
                status="unknown",
                reason="no verified official URL available",
            )

        outcomes.append(
            V4IdentityOutcome(
                identity=identity,
                verification=verification,
                liveness=liveness,
                verification_cache_hit=verification_cache_hit,
                liveness_cache_hit=liveness_cache_hit,
            )
        )

    total_llm_calls = signal_llm_calls + identity_llm_calls + judge_llm_calls
    return V4EmailResult(
        outcomes=outcomes,
        metrics=V4Metrics(
            source_identity_cache_hit=source_cache_hit,
            verification_cache_hits=verification_hits,
            verification_cache_misses=verification_misses,
            liveness_cache_hits=liveness_hits,
            liveness_cache_misses=liveness_misses,
            signal_llm_calls=signal_llm_calls,
            identity_llm_calls=identity_llm_calls,
            search_calls=search_calls,
            verification_fetch_calls=verification_fetch_calls,
            judge_llm_calls=judge_llm_calls,
            liveness_fetch_calls=liveness_fetch_calls,
            total_llm_calls=total_llm_calls,
            approx_llm_source_chars=approx_llm_source_chars,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        ),
        warnings=list(dict.fromkeys(warnings)),
        errors=list(dict.fromkeys(errors)),
    )
