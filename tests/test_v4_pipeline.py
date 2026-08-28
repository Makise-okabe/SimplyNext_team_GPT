from datetime import datetime, timezone

import career_agent.job_identity.v4_pipeline as v4
from career_agent.models.email import EmailMessage
from career_agent.models.job_identity import JobIdentity
from career_agent.models.job_verification import SameJobVerificationResult
from career_agent.storage.verification_cache import (
    VerificationCacheStore,
    verification_cache_key,
)


def test_warm_cache_skips_signal_identity_search_fetch_and_judge(tmp_path, monkeypatch) -> None:
    store = VerificationCacheStore(tmp_path / "v4.db")
    email = EmailMessage(
        message_id="m1",
        sender_email="zeli.goh@nus.edu.sg",
        subject="Example internship",
        received_at=datetime(2026, 8, 28, tzinfo=timezone.utc),
    )
    identity = JobIdentity(
        source_message_id="m1",
        signal_index=1,
        company="Example Corp",
        title="Validation Intern",
        location="Singapore",
        opportunity_type="internship",
        identity_strength="moderate",
        source_fingerprint="fp",
    )
    verification = SameJobVerificationResult(
        identity_status="source_verified",
        identity_basis="trusted_nus_attachment",
        confidence="high",
        official_url=None,
    )
    store.save_source_identities(email.message_id, [identity])
    store.save_verification(verification_cache_key(identity, email), verification)

    def fail(*args, **kwargs):
        raise AssertionError("warm cache should bypass expensive V1/V2/V3 work")

    monkeypatch.setattr(v4, "extract_signal", fail)
    monkeypatch.setattr(v4, "extract_job_identities", fail)
    monkeypatch.setattr(v4, "discover_candidates", fail)
    monkeypatch.setattr(v4, "verify_same_job", fail)

    result = v4.process_email_v4(email, store)

    assert result.metrics.source_identity_cache_hit is True
    assert result.metrics.verification_cache_hits == 1
    assert result.metrics.total_llm_calls == 0
    assert result.metrics.search_calls == 0
    assert result.metrics.verification_fetch_calls == 0
    assert result.metrics.liveness_fetch_calls == 0
    assert result.outcomes[0].verification.identity_status == "source_verified"
    assert result.errors == []
