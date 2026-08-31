from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from career_agent.models.email import EmailMessage


CareerSource = Literal["goh_ze_li", "talentconnect"]


class InboxCheckpoint(BaseModel):
    """Local-only state used to distinguish genuinely new inbox messages."""

    schema_version: int = 1
    seen_message_ids: list[str] = Field(default_factory=list)
    updated_at: datetime | None = None


class CareerEmailRecord(BaseModel):
    """Relevant new career email handed to the next pipeline stage.

    The rich normalized email remains available in-memory for later opportunity
    extraction, while the checkpoint itself stores only message IDs.
    """

    source: CareerSource
    email: EmailMessage


class IncrementalInboxResult(BaseModel):
    scanned_recent: int
    unseen_total: int
    filtered_out: int
    records: list[CareerEmailRecord] = Field(default_factory=list)
    unseen_message_ids: list[str] = Field(default_factory=list)
