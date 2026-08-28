from career_agent.job_identity.concrete_job_research import (
    _core_title,
    _official_broad_query,
)
from career_agent.job_identity.official_research import infer_initial_record_kind
from career_agent.job_identity.targeted_signal import build_targeted_signal
from career_agent.models.email import EmailMessage
from career_agent.models.job_identity import JobIdentity


def test_marvell_newsletter_jobs_section_becomes_concrete_full_time_job() -> None:
    title = "Senior Staff Analog Layout Engineer"
    email = EmailMessage(
        message_id="m1",
        sender_name="Goh Ze Li",
        sender_email="zeli.goh@nus.edu.sg",
        subject="Industry Opportunities",
        body_text=(
            "JOBS\n"
            "Engineering and Manufacturing | Marvell Asia Pte Ltd | "
            f"{title} | 280137 | Deadline: 25 Feb 2026\n"
            "INTERNSHIPS\nOther roles"
        ),
    )

    signal = build_targeted_signal(email, company="Marvell", title=title)

    assert signal is not None
    assert signal.opportunity_type == "full_time"

    identity = JobIdentity(
        source_message_id="m1",
        signal_index=1,
        company="Marvell Asia Pte Ltd",
        title=title,
        opportunity_type=signal.opportunity_type,
        identity_strength="moderate",
        source_fingerprint="fp",
    )
    assert infer_initial_record_kind(identity) == "job_posting"


def test_broader_official_query_removes_seniority_but_keeps_role_core() -> None:
    identity = JobIdentity(
        source_message_id="m1",
        signal_index=1,
        company="Marvell Asia Pte Ltd",
        title="Senior Staff Analog Layout Engineer",
        location="Singapore",
        opportunity_type="full_time",
        identity_strength="moderate",
        source_fingerprint="fp",
    )

    assert _core_title(identity.title) == "Analog Layout Engineer"
    query = _official_broad_query(identity)
    assert "Marvell" in query
    assert '"Analog Layout Engineer"' in query
    assert "Singapore" in query
    assert "Senior Staff" not in query
