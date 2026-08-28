from datetime import datetime, timezone

from career_agent.job_identity.email_targeting import (
    select_target_messages,
    target_relevance_score,
)
from career_agent.models.email import EmailMessage


def _email(message_id: str, subject: str, body: str, day: int) -> EmailMessage:
    return EmailMessage(
        message_id=message_id,
        sender_email="zeli.goh@nus.edu.sg",
        subject=subject,
        body_text=body,
        received_at=datetime(2026, 2, day, tzinfo=timezone.utc),
    )


def test_role_body_match_beats_similar_subject_newsletter() -> None:
    wrong = _email(
        "old",
        "Industry Opportunities and 2025 Career & Internship Status Surveys",
        "General career survey information with no IBM role.",
        20,
    )
    correct = _email(
        "target",
        "From Your CDE Career Advisors: Industry Opportunities + NUS Career Fest Feb 2026",
        "IBM Associate Application Developer-AWS Cloud (Based in Bangkok)",
        8,
    )

    selected = select_target_messages(
        [wrong, correct],
        company="IBM",
        title="Associate Application Developer-AWS Cloud",
        subject_hint="Industry Opportunities",
        limit=1,
    )

    assert selected == [correct]


def test_exact_title_body_match_has_strongest_score() -> None:
    company_only = _email("company", "Jobs", "IBM has opportunities", 10)
    role_match = _email(
        "role",
        "Jobs",
        "IBM Associate Application Developer-AWS Cloud Bangkok",
        9,
    )

    assert target_relevance_score(
        role_match,
        company="IBM",
        title="Associate Application Developer-AWS Cloud",
    ) > target_relevance_score(
        company_only,
        company="IBM",
        title="Associate Application Developer-AWS Cloud",
    )


def test_targeted_selection_does_not_fall_back_to_irrelevant_mail() -> None:
    unrelated = _email("x", "Industry Opportunities", "Marvell role only", 8)

    selected = select_target_messages(
        [unrelated],
        company="IBM",
        title="Associate Application Developer-AWS Cloud",
        subject_hint="Industry Opportunities",
        limit=1,
    )

    assert selected == []
