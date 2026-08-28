from __future__ import annotations

from career_agent.models.email import EmailMessage
from career_agent.nodes.normalize_email import normalize_email


def _normalize(value: str | None) -> str:
    return " ".join((value or "").lower().split())


def target_relevance_score(
    email: EmailMessage,
    company: str | None = None,
    title: str | None = None,
    subject_hint: str | None = None,
) -> int:
    """Rank mailbox messages before --limit is applied.

    The concrete role title in the message body is the strongest signal, followed
    by the target company. A broad newsletter subject is only a weak hint.
    """
    normalized = normalize_email(email)
    body = _normalize(normalized.body_text)
    subject = _normalize(normalized.subject)
    company_value = _normalize(company)
    title_value = _normalize(title)
    subject_value = _normalize(subject_hint)

    score = 0
    if title_value:
        if title_value in body:
            score += 120
        if title_value in subject:
            score += 60
    if company_value:
        if company_value in body:
            score += 45
        if company_value in subject:
            score += 25
    if subject_value and subject_value in subject:
        score += 10
    return score


def select_target_messages(
    messages: list[EmailMessage],
    company: str | None = None,
    title: str | None = None,
    subject_hint: str | None = None,
    limit: int = 1,
) -> list[EmailMessage]:
    ranked = [
        (
            target_relevance_score(
                message,
                company=company,
                title=title,
                subject_hint=subject_hint,
            ),
            message,
        )
        for message in messages
    ]

    # When a concrete target is supplied, do not silently fall back to an email
    # that contains neither the company nor role. This avoids false zero-signal
    # test runs against similarly named newsletters.
    if company or title:
        ranked = [item for item in ranked if item[0] > 0]
    elif subject_hint:
        ranked = [
            item
            for item in ranked
            if _normalize(subject_hint) in _normalize(item[1].subject)
        ]

    ranked.sort(
        key=lambda item: (
            item[0],
            item[1].received_at.timestamp() if item[1].received_at else 0.0,
        ),
        reverse=True,
    )
    return [message for _, message in ranked[: max(1, limit)]]
