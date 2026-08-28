from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

VERIFICATION_RANK = {
    "unresolved": 0,
    "partial": 1,
    "source_verified": 2,
    "verified": 3,
}


class OpportunityStore:
    """Minimal durable memory for normalized opportunities only.

    Raw email bodies, PDF bytes and attachment text are intentionally not stored.
    The database therefore acts as a privacy-minimised product memory rather than
    an archive of the student's mailbox.
    """

    def __init__(self, path: str | Path = "private_data/simplynext.db") -> None:
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
                CREATE TABLE IF NOT EXISTS opportunities (
                    fingerprint TEXT PRIMARY KEY,
                    company TEXT NOT NULL,
                    title TEXT NOT NULL,
                    location TEXT,
                    opportunity_type TEXT NOT NULL,
                    official_url TEXT,
                    application_url TEXT,
                    deadline TEXT,
                    verification_status TEXT NOT NULL,
                    verification_basis TEXT NOT NULL,
                    source_name TEXT,
                    source_email TEXT,
                    source_message_id TEXT,
                    source_date TEXT,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL
                )
                """
            )

            # Lightweight migration for databases created by the previous prototype
            # commit. This keeps a user's local memory usable after git pull.
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(opportunities)").fetchall()
            }
            if "application_url" not in columns:
                connection.execute(
                    "ALTER TABLE opportunities ADD COLUMN application_url TEXT"
                )

            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_opportunities_last_seen "
                "ON opportunities(last_seen_at DESC)"
            )

    @staticmethod
    def fingerprint(job: dict) -> str:
        # URLs/deadlines are mutable evidence, not job identity. Excluding them
        # means a source-verified record can later gain an official URL without
        # appearing as a duplicate opportunity.
        canonical = "|".join(
            str(job.get(field) or "").strip().lower()
            for field in ("company", "title", "location", "opportunity_type")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _identity_values(job: dict) -> tuple[str, str, str, str]:
        return (
            job.get("company") or "Unknown",
            job.get("title") or "Unknown role",
            job.get("location") or "",
            job.get("opportunity_type") or "unknown",
        )

    def upsert_job(self, job: dict, source_email: dict | None = None) -> bool:
        source_email = source_email or {}
        company, title, location, opportunity_type = self._identity_values(job)
        new_fingerprint = self.fingerprint(job)
        now = datetime.now(timezone.utc).isoformat()

        with self._connect() as connection:
            # Identity lookup also recognizes rows written by the old fingerprint
            # algorithm, so schema/identity upgrades do not duplicate memory.
            existing = connection.execute(
                """
                SELECT fingerprint, verification_status, verification_basis
                FROM opportunities
                WHERE lower(company) = lower(?)
                  AND lower(title) = lower(?)
                  AND lower(COALESCE(location, '')) = lower(?)
                  AND lower(opportunity_type) = lower(?)
                LIMIT 1
                """,
                (company, title, location, opportunity_type),
            ).fetchone()

            fingerprint = existing["fingerprint"] if existing else new_fingerprint
            status = job.get("verification_status") or "unresolved"
            basis = job.get("verification_basis") or "none"

            if existing:
                old_status = existing["verification_status"] or "unresolved"
                if VERIFICATION_RANK.get(old_status, 0) > VERIFICATION_RANK.get(status, 0):
                    status = old_status
                    basis = existing["verification_basis"] or "none"

            connection.execute(
                """
                INSERT INTO opportunities (
                    fingerprint, company, title, location, opportunity_type,
                    official_url, application_url, deadline,
                    verification_status, verification_basis,
                    source_name, source_email, source_message_id, source_date,
                    first_seen_at, last_seen_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(fingerprint) DO UPDATE SET
                    official_url = COALESCE(excluded.official_url, opportunities.official_url),
                    application_url = COALESCE(excluded.application_url, opportunities.application_url),
                    deadline = COALESCE(excluded.deadline, opportunities.deadline),
                    verification_status = excluded.verification_status,
                    verification_basis = excluded.verification_basis,
                    source_name = excluded.source_name,
                    source_email = excluded.source_email,
                    source_message_id = excluded.source_message_id,
                    source_date = excluded.source_date,
                    last_seen_at = excluded.last_seen_at
                """,
                (
                    fingerprint,
                    company,
                    title,
                    job.get("location"),
                    opportunity_type,
                    str(job.get("official_url")) if job.get("official_url") else None,
                    str(job.get("application_url")) if job.get("application_url") else None,
                    str(job.get("deadline")) if job.get("deadline") else None,
                    status,
                    basis,
                    source_email.get("sender_name"),
                    source_email.get("sender_email"),
                    source_email.get("message_id"),
                    str(source_email.get("received_at")) if source_email.get("received_at") else None,
                    now,
                    now,
                ),
            )

        return existing is None

    def upsert_jobs(self, jobs: list[dict], source_email: dict | None = None) -> tuple[int, int]:
        inserted = 0
        updated = 0
        for job in jobs:
            if self.upsert_job(job, source_email=source_email):
                inserted += 1
            else:
                updated += 1
        return inserted, updated

    def list_recent(self, limit: int = 50) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT company, title, location, opportunity_type,
                       official_url, application_url, deadline,
                       verification_status, verification_basis,
                       source_name, source_email, source_message_id, source_date,
                       first_seen_at, last_seen_at
                FROM opportunities
                ORDER BY last_seen_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM opportunities").fetchone()
        return int(row["count"])
