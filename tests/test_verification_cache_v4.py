from datetime import datetime, timedelta, timezone

from career_agent.models.email import EmailMessage
from career_agent.models.job_identity import JobIdentity
from career_agent.models.job_v4 import LivenessResult
from career_agent.models.job_verification import SameJobVerificationResult
from career_agent.storage.verification_cache import (
    VerificationCacheStore,
    verification_cache_key,
)


def _identity() -> JobIdentity:
    return JobIdentity(
        source_message_id="m1",
        signal_index=1,
        company="Example Corp",
        title="Graduate Engineer",
        location="Singapore",
        opportunity_type="full_time",
        start_period="July",
        end_period="December",
        distinctive_phrases=["advanced silicon validation platform"],
        evidence_snippets=["RAW-LIKE EVIDENCE MUST NOT BE CACHED"],
        identity_strength="strong",
        source_fingerprint="fingerprint",
    )


def test_source_identity_cache_strips_evidence_snippets(tmp_path) -> None:
    store = VerificationCacheStore(tmp_path / "v4.db")
    store.save_source_identities("m1", [_identity()])

    cached = store.get_source_identities("m1")

    assert cached is not None
    assert cached[0].company == "Example Corp"
    assert cached[0].evidence_snippets == []


def test_verification_cache_key_changes_across_source_years() -> None:
    identity = _identity()
    email_2026 = EmailMessage(
        message_id="a",
        sender_email="zeli.goh@nus.edu.sg",
        subject="Role",
        received_at=datetime(2026, 8, 28, tzinfo=timezone.utc),
    )
    email_2027 = email_2026.model_copy(
        update={
            "message_id": "b",
            "received_at": datetime(2027, 8, 28, tzinfo=timezone.utc),
        }
    )

    assert verification_cache_key(identity, email_2026) != verification_cache_key(identity, email_2027)


def test_verification_ttl_is_status_aware(tmp_path) -> None:
    store = VerificationCacheStore(tmp_path / "v4.db")
    result = SameJobVerificationResult(
        identity_status="source_verified",
        identity_basis="trusted_nus_attachment",
        confidence="high",
    )
    store.save_verification("key", result)

    assert store.get_verification("key") is not None
    future = datetime.now(timezone.utc) + timedelta(days=8)
    assert store.get_verification("key", now=future) is None


def test_liveness_cache_expires_independently(tmp_path) -> None:
    store = VerificationCacheStore(tmp_path / "v4.db")
    result = LivenessResult(
        url="https://careers.example.com/job/1",
        status="open",
        reason="apply now",
    )
    store.save_liveness(result)

    assert store.get_liveness(result.url) is not None
    future = datetime.now(timezone.utc) + timedelta(hours=7)
    assert store.get_liveness(result.url, now=future) is None
