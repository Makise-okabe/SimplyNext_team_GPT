from datetime import datetime

from pydantic import BaseModel, Field


class EmailMessage(BaseModel):
    message_id: str
    sender_name: str | None = None
    sender_email: str | None = None
    subject: str
    received_at: datetime | None = None

    # Preserve the forwarding envelope so the normalized sender can represent
    # the original NUS career source without losing provenance.
    transport_sender_name: str | None = None
    transport_sender_email: str | None = None

    body_text: str = ""
    body_html: str = ""
    attachment_text: str = ""

    links: list[str] = Field(default_factory=list)
    attachments: list[str] = Field(default_factory=list)
