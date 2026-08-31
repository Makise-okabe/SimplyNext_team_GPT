from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from career_agent.models.email import EmailMessage


CareerSource = Literal["goh_ze_li", "talentconnect"]


class InboxCheckpoint(BaseModel):
    """Local-only state used to distinguish genuinely new inbox messages.

    ``baseline_at`` anchors the first manual bootstrap so old messages that were
    outside the initial scan window cannot later be mistaken for newly arrived
    mail. ``seen_message_ids`` then provides exact de-duplication after bootstrap.
    No email body, attachment text, or access token is stored here.
    """

    schema_version: int = 1
    baseline_at: datetime
    seen_message_ids: list[str] = Field(default_factory=list)
    updated_at: datetime


class CareerEmailRecord(BaseModel):
    """Relevant new career email handed to the next pipeline stage in memory."""

    source: CareerSource
    email: EmailMessage


class IncrementalInboxResult(BaseModel):
    scanned_recent: int
    unseen_total: int
    filtered_out: int
    records: list[CareerEmailRecord] = Field(default_factory=list)
    unseen_message_ids: list[str] = Field(default_factory=list)
    checkpoint_committed: bool = False
