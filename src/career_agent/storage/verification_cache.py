from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from career_agent.models.email import EmailMessage
from career_agent.models.job_identity import JobIdentity
from career_agent.models.job_v4 import LivenessResult
from career_agent.models.job_verification import SameJobVerificationResult

CACHE_VERSION = "v4.1"
VERIFICATION_TTL = {
    "verified": timedelta(days=30),
    "source_verified": timedelta(days=7),
    "ambiguous": timedelta(days=1),
    "unresolved": timedelta(days=1),
}
DEFAULT_LIVENESS_TTL = timedelta(hours=6)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def verification_cache_key(identity: JobIdentity, email: EmailMessage) -> str:
    """Cycle-aware key so recurring annual roles never share verification blindly."""
    source_year = ""
    if email.received_at is not None:
        source_year = str(email.received_at.year)

    canonical = "|".join(
        [
            identity.source_fingerprint,
            source_year,
            (identity.start_period or "").strip().lower(),
            (identity.end_period or "").strip().lower(),
            (identity.duration or "").strip().lower(),
        ]
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class VerificationCacheStore:
    """Privacy-minimised cache for V1 identities, V3 decisions and liveness.

    It intentionally stores no raw email body, PDF text, attachment bytes or full
    fetched-page body. Cached JobIdentity evidence snippets are stripped as well;
    only compact structured identity fields/distinctive phrases remain.
    """

    def __init__(self, path: str | Path = "private_data/job_identity_v4.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS source_identity_cache (
                    source_message_id TEXT PRIMARY KEY,
                    cache_version TEXT NOT NULL,
                    identities_json TEXT NOT NULL,
                    cached_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS verification_cache (
                    cache_key TEXT PRIMARY KEY,
                    cache_version TEXT NOT NULL,
                    identity_status TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    cached_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS liveness_cache (
                    url TEXT PRIMARY KEY,
                    cache_version TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    cached_at TEXT NOT NULL
                )
                """
            )

    def clear_all(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM source_identity_cache")
            connection.execute("DELETE FROM verification_cache")
            connection.execute("DELETE FROM liveness_cache")

    def save_source_identities(self, source_message_id: str, identities: list[JobIdentity]) -> None:
        sanitized: list[dict] = []
        for identity in identities:
            payload = identity.model_dump(mode="json")
            payload["evidence_snippets"] = []
            sanitized.append(payload)

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO source_identity_cache (
                    source_message_id, cache_version, identities_json, cached_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(source_message_id) DO UPDATE SET
                    cache_version = excluded.cache_version,
                    identities_json = excluded.identities_json,
                    cached_at = excluded.cached_at
                """,
                (
                    source_message_id,
                    CACHE_VERSION,
                    json.dumps(sanitized, ensure_ascii=False),
                    _utcnow().isoformat(),
                ),
            )

    def get_source_identities(self, source_message_id: str) -> list[JobIdentity] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT cache_version, identities_json
                FROM source_identity_cache
                WHERE source_message_id = ?
                """,
                (source_message_id,),
            ).fetchone()

        if not row or row["cache_version"] != CACHE_VERSION:
            return None
        return [JobIdentity.model_validate(item) for item in json.loads(row["identities_json"])]

    def save_verification(self, cache_key: str, result: SameJobVerificationResult) -> None:
        payload = result.model_dump(mode="json")
        # Runtime metrics/transient warnings are not identity facts and should not
        # reappear on a warm cache hit.
        payload["metrics"] = {}
        payload["warnings"] = []
        payload["errors"] = []

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO verification_cache (
                    cache_key, cache_version, identity_status, result_json, cached_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    cache_version = excluded.cache_version,
                    identity_status = excluded.identity_status,
                    result_json = excluded.result_json,
                    cached_at = excluded.cached_at
                """,
                (
                    cache_key,
                    CACHE_VERSION,
                    result.identity_status,
                    json.dumps(payload, ensure_ascii=False),
                    _utcnow().isoformat(),
                ),
            )

    def get_verification(
        self,
        cache_key: str,
        now: datetime | None = None,
    ) -> SameJobVerificationResult | None:
        now = now or _utcnow()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT cache_version, identity_status, result_json, cached_at
                FROM verification_cache
                WHERE cache_key = ?
                """,
                (cache_key,),
            ).fetchone()

        if not row or row["cache_version"] != CACHE_VERSION:
            return None

        ttl = VERIFICATION_TTL.get(row["identity_status"], timedelta(days=1))
        if now - _parse_time(row["cached_at"]) > ttl:
            return None
        return SameJobVerificationResult.model_validate(json.loads(row["result_json"]))

    def save_liveness(self, result: LivenessResult) -> None:
        if not result.url:
            return
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO liveness_cache (
                    url, cache_version, result_json, cached_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    cache_version = excluded.cache_version,
                    result_json = excluded.result_json,
                    cached_at = excluded.cached_at
                """,
                (
                    result.url,
                    CACHE_VERSION,
                    json.dumps(result.model_dump(mode="json"), ensure_ascii=False),
                    _utcnow().isoformat(),
                ),
            )

    def get_liveness(
        self,
        url: str,
        ttl: timedelta = DEFAULT_LIVENESS_TTL,
        now: datetime | None = None,
    ) -> LivenessResult | None:
        now = now or _utcnow()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT cache_version, result_json, cached_at
                FROM liveness_cache
                WHERE url = ?
                """,
                (url,),
            ).fetchone()

        if not row or row["cache_version"] != CACHE_VERSION:
            return None
        if now - _parse_time(row["cached_at"]) > ttl:
            return None
        return LivenessResult.model_validate(json.loads(row["result_json"]))
