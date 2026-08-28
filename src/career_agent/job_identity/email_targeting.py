from __future__ import annotations

from career_agent.models.email import EmailMessage
from career_agent.nodes.normalize_email import normalize_email


def _normalize(value: str | None) -> str:
    return " ".join((value or "").lower().split())


def _target_hits(
    email: EmailMessage,
    company: str | None = None,
    title: str | None = None,
) -> tuple[bool, bool]:
    """Return whether the concrete company/title target occurs in this email."""
    normalized = normalize_email(email)
    body = _normalize(normalized.body_text)
    subject = _normalize(normalized.subject)
    company_value = _normalize(company)
    title_value = _normalize(title)

    company_hit = bool(
        company_value
        and (company_value in body or company_value in subject)
    )
    title_hit = bool(
        title_value
        and (title_value in body or title_value in subject)
    )
    return company_hit, title_hit


def target_relevance_score(
    email: EmailMessage,
    company: str | None = None,
    title: str | None = None,
    subject_hint: str | None = None,
) -> int:
    """Rank mailbox messages before --limit is applied.

    The concrete role title in the message body is the strongest signal, followed
    by the target company. A broad newsletter subject is only a weak ranking hint
    and must never make an otherwise irrelevant email eligible by itself.
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
    ranked: list[tuple[int, EmailMessage]] = []

    for message in messages:
        company_hit, title_hit = _target_hits(
            message,
            company=company,
            title=title,
        )

        # Concrete targeting rule: subject_hint is never sufficient. When a
        # company/title target is supplied, at least one concrete target must
        # actually occur in the email body/subject before the message can enter
        # the ranking pool.
        if (company or title) and not (company_hit or title_hit):
            continue

        if not company and not title and subject_hint:
            if _normalize(subject_hint) not in _normalize(message.subject):
                continue

        ranked.append(
            (
                target_relevance_score(
                    message,
                    company=company,
                    title=title,
                    subject_hint=subject_hint,
                ),
                message,
            )
        )

    ranked.sort(
        key=lambda item: (
            item[0],
            item[1].received_at.timestamp() if item[1].received_at else 0.0,
        ),
        reverse=True,
    )
    return [message for _, message in ranked[: max(1, limit)]]
